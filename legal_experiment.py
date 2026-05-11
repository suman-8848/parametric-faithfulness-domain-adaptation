#!/usr/bin/env python3
"""
Legal Domain Faithfulness Experiment
Adapted for Quartz GPU cluster (Slurm)
"""

import os
import sys
import json
import argparse
import random
import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Add parametric-faithfulness to path
REPO_DIR = Path(__file__).parent / 'parametric-faithfulness'
sys.path.insert(0, str(REPO_DIR))

from const import LETTERS
from dataload import DataHandler, DATASETS, BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX
from data import FRCollator, cot_to_otfd, load_or_generate_dataset_cots
from util import set_random_seed
from segment import sentencize

# Fix evaluate.py length_penalty bug
eval_path = REPO_DIR / 'evaluate.py'
if eval_path.exists():
    with open(eval_path, 'r') as f:
        code = f.read()
    if 'length_penalty = model.generation_config.length_penalty' in code:
        code = code.replace(
            'length_penalty = model.generation_config.length_penalty or 1.0', 
            'length_penalty = 1.0'
        ).replace(
            'length_penalty = model.generation_config.length_penalty', 
            'length_penalty = 1.0'
        )
        with open(eval_path, 'w') as f:
            f.write(code)

import importlib
import evaluate as _eval_module
importlib.reload(_eval_module)
from evaluate import completion_probabilities, answer_probabilities, complete, generation_fixed_cot


# ============================================================
# LegalBench Dataset Handler
# ============================================================

class LegalBenchDatasetHandler(DataHandler):
    id_key = 'qid'
    q_key  = 'question'
    letter_choices = ['A', 'B', 'C', 'D']

    def __init__(self, data_path):
        with open(data_path) as f:
            self._data = json.load(f)
        super().__init__()

    def get_dataset_splits(self):
        data = list(self._data)
        random.Random(42).shuffle(data)
        return data[:8], data[8:16], data

    def get_answer_letters(self, instance):
        return [opt[0] for opt in instance['options']]

    def get_answer_choices(self, instance):
        return instance['options']

    def correct_answer_letter(self, instance):
        return instance['answer']

    def make_bowman_demonstration(self, instance):
        choices = '\n'.join(f'({opt[0]}): {opt[3:]}' for opt in instance['options'])
        return (
            f"Human: Question: {instance['question']}\n\n"
            f"Choices:\n{choices}\n\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )

    def make_cot_prompt(self, instance):
        choices = '\n'.join(f'({opt[0]}): {opt[3:]}' for opt in instance['options'])
        return (
            f"Human: Question: {instance['question']}\n\n"
            f"Choices:\n{choices}\n\n"
            f"Assistant: Let's think step by step:\n"
        )

    def make_answer_prompt(self, prefix):
        return (
            f"{prefix}\n"
            f"{BOWMAN_HUMAN_ANSWER_PREFIX}\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )


# ============================================================
# Download and prepare LegalBench dataset
# ============================================================

def prepare_legalbench_dataset():
    """Download and prepare LegalBench dataset."""
    from datasets import load_dataset
    
    legal_dir = REPO_DIR / 'data' / 'legalbench'
    legal_dir.mkdir(parents=True, exist_ok=True)
    legal_file = legal_dir / 'legalbench_test.json'
    
    if legal_file.exists():
        print(f"✅ LegalBench already prepared: {legal_file}")
        return str(legal_file)
    
    print("📥 Downloading LegalBench from HuggingFace...")
    ds = load_dataset('nguha/legalbench', 'abercrombie', split='test')
    
    option_keys = ['A', 'B', 'C', 'D']
    label_map = {
        0: 'Generic',
        1: 'Descriptive', 
        2: 'Suggestive',
        3: 'Arbitrary or Fanciful'
    }
    
    converted = []
    for i, item in enumerate(ds):
        options_list = [f"{k}): {label_map[j]}" for j, k in enumerate(option_keys)]
        answer_idx = option_keys[item['label']]
        
        converted.append({
            'qid': f'legal_{i:05d}',
            'question': f"Classify the following trademark term according to the Abercrombie classification: '{item['text']}'",
            'options': options_list,
            'answer': answer_idx,
            'context': item['text']
        })
    
    with open(legal_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"✅ LegalBench saved: {len(converted)} instances → {legal_file}")
    return str(legal_file)


# ============================================================
# Core Unlearning Functions
# ============================================================

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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map='auto'
    )
    model.gradient_checkpointing_enable()

    # Oracle model (8-bit)
    oracle_model = AutoModelForCausalLM.from_pretrained(
        model_id, load_in_8bit=True,
        trust_remote_code=True, device_map='auto'
    )
    oracle_model.eval()
    for p in oracle_model.parameters():
        p.requires_grad = False

    device = next(model.parameters()).device
    collator = FRCollator(tokenizer, device=device)

    dataset = cot_to_otfd(target, cots_train, tokenizer,
                          strategy=config.strategy, stepwise=config.stepwise,
                          step_idx=step_idx, pos=config.pos)
    NT = dataset.num_targets()
    print(f'  Num targets: {NT} | Step: {target["segmented_cot"][step_idx][:60]}...')

    if NT <= 2:
        print('  Too few targets, skipping.')
        del model, oracle_model
        gc.collect(); torch.cuda.empty_cache()
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
        print(f'  Epoch {epoch+1}/{config.epochs} done.')

    del collator, train_dl, dataset, scheduler, optimizer, model, oracle_model
    gc.collect(); torch.cuda.empty_cache()
    return {'unlearning_results': results}


def load_ids(fin, stepwise=False):
    ids = set()
    if os.path.exists(fin):
        with open(fin) as f:
            for line in f:
                d = json.loads(line)
                id_ = d['question']
                if stepwise:
                    id_ = f"{id_}_{d['step_idx']}"
                ids.add(id_)
    return ids


def store(instance_info, fout):
    with open(fout, 'a') as f:
        f.write(json.dumps(instance_info) + '\n')


# ============================================================
# Main Experiment
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='legalbench')
    parser.add_argument('--method', type=str, default='npo_KL')
    parser.add_argument('--strategy', type=str, default='sentencize')
    parser.add_argument('--stepwise', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-05)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--ff2', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_instances', type=int, default=30)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    print("="*60)
    print("Legal Domain Faithfulness Experiment")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Max instances: {args.max_instances}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Set random seed
    set_random_seed(args.seed)

    # Login to HuggingFace
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️  HF_TOKEN not found, proceeding without login")

    # Prepare dataset
    legal_file = prepare_legalbench_dataset()
    DATASETS['legalbench'] = LegalBenchDatasetHandler(legal_file)
    DH = DATASETS['legalbench']
    print(f"✅ LegalBench handler registered: {len(DH._data)} instances")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate or load CoTs
    print(f"\n📝 Loading/generating CoTs for {args.model_name}...")
    cot_data = load_or_generate_dataset_cots(
        model_id=args.model_name, tokenizer=tokenizer,
        dataset_id='legalbench', force_generate=False,
        sentencize=(args.strategy == 'sentencize'),
        temperature=args.temperature, seed=args.seed,
        atomic=False
    )

    random.shuffle(cot_data)
    N_verify = 20
    cots_train, cots_verify = cot_data[:-N_verify], cot_data[-N_verify:]

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logfile_name = (f"{args.method}_{args.strategy}_s={args.stepwise}"
                    f"_lr={args.lr}_rs={args.seed}"
                    f"_pos={args.pos}_ff2={args.ff2}.out")
    logfile = output_dir / logfile_name

    ids = load_ids(logfile, stepwise=args.stepwise)
    print(f"🔄 Resuming from {len(ids)} previously processed instances")
    print(f"📂 Results → {logfile}\n")

    all_results = []

    for idx, target in enumerate(cots_train[:args.max_instances]):
        n_steps = len(target['segmented_cot']) if args.stepwise else 1

        for step_idx in range(n_steps):
            check_id = target['id']
            if args.stepwise:
                check_id = f"{check_id}_{step_idx}"
            if check_id in ids:
                continue

            print(f"\n{'='*60}")
            print(f'Instance {idx+1}/{args.max_instances}, Step {step_idx+1}/{n_steps}')
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
                'model': args.model_name,
            }

            return_dict = unlearn_single(
                args.model_name, tokenizer, args, target, step_idx,
                cots_train, cots_verify, DH, idx
            )

            results = return_dict['unlearning_results']
            if results is None:
                continue

            instance_info['unlearning_results'] = results
            store(instance_info, logfile)
            all_results.append(instance_info)

            initial_pred = instance_info['cot_prediction']
            final_pred = results[args.epochs]['prediction']
            flipped = '🔄 FLIPPED' if initial_pred != final_pred else '✓ Same'
            print(f'  → {LETTERS[initial_pred]} → {LETTERS[final_pred]} {flipped}')

    print(f"\n{'='*60}")
    print(f'✅ Done! Processed {len(all_results)} instances')
    print(f'📂 Results saved to: {logfile}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
