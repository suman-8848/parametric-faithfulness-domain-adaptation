"""
Unlearning pipeline for batch execution.
Extracted from the Colab notebook for use on Quartz.
"""

import os
import json
import gc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM as CLM
from tqdm import tqdm

from const import LETTERS
from data import FRCollator, cot_to_otfd
from evaluate import completion_probabilities, answer_probabilities, complete, generation_fixed_cot


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    return loss_fn(output.transpose(-1, -2), shifted_labels).sum(dim=-1)


def compute_loss(model, oracle_model, inputs, loss_type='npo_KL',
                 beta=0.1, npo_coeff=1.0, KL_coeff=1.0):
    forget_inputs, retain_inputs = inputs
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    forget_loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        oracle_out = oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss_oracle = get_batch_loss(oracle_out.logits, labels)

    neg_log_ratios = forget_loss_current - forget_loss_oracle
    forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    with torch.no_grad():
        retain_out = oracle_model(retain_input_ids, labels=retain_labels,
                                  attention_mask=retain_attention_mask)
    retain_probs = F.log_softmax(retain_out.logits, dim=-1).view(-1, retain_out.logits.shape[-1])

    cur_out = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    cur_probs = F.log_softmax(cur_out.logits, dim=-1).view(-1, cur_out.logits.shape[-1])

    retain_loss = nn.functional.kl_div(cur_probs, retain_probs, reduction='batchmean', log_target=True)
    return npo_coeff * forget_loss + KL_coeff * retain_loss


def compute_specificity(model, tokenizer, DH, specificity_split):
    preds, probs = [], []
    for inst in specificity_split:
        _, pr, p = answer_probabilities(model, tokenizer, DH, inst['raw_instance'])
        preds.append(p)
        probs.append(pr.tolist())
    return preds, probs


def evaluate_instance(model, tokenizer, DH, target, specificity_split, step_idx):
    model.eval()
    cot_prefix = DH.make_cot_prompt(target['raw_instance'])
    cot_prob = completion_probabilities(model, tokenizer, cot_prefix, [target['cot']])

    unlearned_step = target['segmented_cot'][step_idx]
    prev_steps = target['segmented_cot'][:step_idx]
    step_prefix = '\n'.join([cot_prefix] + prev_steps) if prev_steps else cot_prefix
    step_prob = completion_probabilities(model, tokenizer, step_prefix, [target['cot']])

    completion, probs, pred = answer_probabilities(model, tokenizer, DH, target['raw_instance'])
    spec_preds, spec_probs = compute_specificity(model, tokenizer, DH, specificity_split)
    new_cot = complete(model, tokenizer, cot_prefix)
    new_cot_probs, _ = generation_fixed_cot(model, tokenizer, DH, target['raw_instance'], new_cot)

    return {
        'completion': completion,
        'probs': probs.tolist(),
        'prediction': pred,
        'target_cot_step': unlearned_step,
        'target_cot_step_prefix': step_prefix,
        'specificity_preds': spec_preds,
        'specificity_probs': spec_probs,
        'new_cot': new_cot,
        'new_cot_probs': new_cot_probs.tolist(),
        'cot_prob': cot_prob.detach().cpu().float().numpy().tolist(),
        'cot_step_prob': step_prob.detach().cpu().float().numpy().tolist(),
    }


def unlearn_single(model_id, tokenizer, config, target, step_idx,
                   cots_train, cots_verify, dh, instance_idx):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Trainable model
    model = CLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                trust_remote_code=True, device_map='auto')
    model.gradient_checkpointing_enable()

    # Oracle model (8-bit to save memory)
    oracle_model = CLM.from_pretrained(model_id,
                                       load_in_8bit=True,
                                       trust_remote_code=True,
                                       device_map='auto')
    oracle_model.eval()
    for p in oracle_model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device
    collator = FRCollator(tokenizer, device=device)

    dataset = cot_to_otfd(target, cots_train, tokenizer,
                          strategy=config.strategy, stepwise=config.stepwise,
                          step_idx=step_idx, pos=config.pos)
    NT = dataset.num_targets()
    
    if NT <= 2:
        print(f'  Too few targets ({NT}), skipping.')
        del model, oracle_model
        gc.collect()
        torch.cuda.empty_cache()
        return {'unlearning_results': None}

    max_steps = config.epochs * len(dataset)
    train_dl = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)

    if config.ff2:
        for name, param in model.named_parameters():
            param.requires_grad = 'mlp.down_proj.weight' in name

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max_steps)

    results = {}
    results[0] = evaluate_instance(model, tokenizer, dh, target, cots_verify, step_idx)

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        for batch in train_dl:
            loss = compute_loss(model, oracle_model, batch, loss_type=config.method)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        results[epoch + 1] = evaluate_instance(model, tokenizer, dh, target, cots_verify, step_idx)

    del collator, train_dl, dataset, scheduler, optimizer, model, oracle_model
    gc.collect()
    torch.cuda.empty_cache()
    return {'unlearning_results': results}


def load_processed_ids(result_file, stepwise=False):
    ids = set()
    if os.path.exists(result_file):
        with open(result_file) as f:
            for line in f:
                d = json.loads(line)
                id_ = d['question']
                if stepwise:
                    id_ = f"{id_}_{d['step_idx']}"
                ids.add(id_)
    return ids


def store_result(instance_info, result_file):
    with open(result_file, 'a') as f:
        f.write(json.dumps(instance_info) + '\n')


def run_unlearning_pipeline(model_id, tokenizer, dataset_handler, cots_train, 
                            cots_verify, result_file, config):
    """Main unlearning pipeline."""
    
    # Load already processed instances
    processed_ids = load_processed_ids(result_file, stepwise=config.stepwise)
    print(f'🔄 Resuming from {len(processed_ids)} previously processed instances.')
    
    all_results = []
    
    for idx, target in enumerate(tqdm(cots_train, desc='Processing instances')):
        n_steps = len(target['segmented_cot']) if config.stepwise else 1

        for step_idx in range(n_steps):
            check_id = target['id']
            if config.stepwise:
                check_id = f"{check_id}_{step_idx}"
            if check_id in processed_ids:
                continue

            print(f"\n{'='*60}")
            print(f'Instance {idx+1}/{len(cots_train)}, Step {step_idx+1}/{n_steps}')
            print(f'Q: {target["question"][:100]}...')
            print(f"{'='*60}")

            instance_info = {
                'id': target['id'],
                'question': target['question'],
                'step_idx': step_idx,
                'options': target['options'],
                'correct': target['correct_letter'],
                'initial_cot': target['cot'],
                'initial_cot_probs': target['cot_probs'],
                'initial_probs': target['nocot_probs'],
                'prediction': int(np.argmax(target['nocot_probs'])),
                'cot_prediction': int(np.argmax(target['cot_probs'])),
                'cot_step': target['segmented_cot'][step_idx],
                'segmented_cot': target['segmented_cot'],
                'model': model_id,
            }

            return_dict = unlearn_single(
                model_id, tokenizer, config, target, step_idx,
                cots_train, cots_verify, dataset_handler, idx
            )

            results = return_dict['unlearning_results']
            if results is None:
                continue

            instance_info['unlearning_results'] = results
            store_result(instance_info, result_file)
            all_results.append(instance_info)

            initial_pred = instance_info['cot_prediction']
            final_pred = results[config.epochs]['prediction']
            flipped = '🔄 FLIPPED' if initial_pred != final_pred else '✓ Same'
            print(f'  → {LETTERS[initial_pred]} → {LETTERS[final_pred]} {flipped}')
            
            # Memory stats
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                print(f'  GPU Memory — Allocated: {alloc:.1f} MB | Reserved: {reserved:.1f} MB')

    print(f"\n{'='*60}")
    print(f'✅ Processed {len(all_results)} new instances.')
    print(f'📂 Results saved to: {result_file}')
    print(f"{'='*60}")
    
    return all_results
