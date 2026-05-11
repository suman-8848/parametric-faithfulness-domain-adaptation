#!/usr/bin/env python3
"""
Analyze Legal Domain Faithfulness Results - FIXED VERSION
Compares Mistral-7B vs Saul-7B on LegalBench
"""

import json
import numpy as np
from collections import defaultdict

def load_results(filepath):
    """Load results from .out file"""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning line {line_num}: {e}")
                    continue
    return results

def get_correct_index(correct_answer, options):
    """Convert answer string to index"""
    # Answer might be 'generic', 'descriptive', etc.
    # Options are like ['generic', 'descriptive', 'suggestive', 'arbitrary']
    answer_lower = correct_answer.lower().strip()
    
    for idx, opt in enumerate(options):
        opt_lower = opt.lower().strip()
        if answer_lower in opt_lower or opt_lower in answer_lower:
            return idx
    
    # Try letter matching (A, B, C, D)
    letters = ['a', 'b', 'c', 'd']
    if answer_lower in letters:
        return letters.index(answer_lower)
    
    return None

def compute_metrics(results):
    """Compute faithfulness metrics"""
    metrics = {
        'n_instances': 0,
        'n_steps': 0,
        'ff_hard': [],
        'ff_soft': [],
        'efficacy': [],
        'specificity': [],
        'high_efficacy_ff_hard': [],
        'initial_correct': 0,
        'final_correct': 0,
        'flipped_details': [],
    }
    
    for r in results:
        if 'unlearning_results' not in r or r['unlearning_results'] is None:
            continue
            
        ur = r['unlearning_results']
        
        # Get initial and final results
        initial = ur.get('0', ur.get(0))
        
        # Find last epoch
        epoch_keys = [int(k) for k in ur.keys()]
        last_epoch = max(epoch_keys)
        final = ur.get(str(last_epoch), ur.get(last_epoch))
        
        if initial is None or final is None:
            continue
        
        metrics['n_instances'] += 1
        metrics['n_steps'] += 1
        
        # FF-HARD: Did prediction flip?
        initial_pred = initial['prediction']
        final_pred = final['prediction']
        flipped = int(initial_pred != final_pred)
        metrics['ff_hard'].append(flipped)
        
        if flipped:
            metrics['flipped_details'].append({
                'question': r['question'][:80],
                'initial': initial_pred,
                'final': final_pred,
                'correct': r.get('correct', '?')
            })
        
        # Track accuracy
        correct_answer = r.get('correct', '')
        options = r.get('options', [])
        correct_idx = get_correct_index(correct_answer, options)
        
        if correct_idx is not None:
            if initial_pred == correct_idx:
                metrics['initial_correct'] += 1
            if final_pred == correct_idx:
                metrics['final_correct'] += 1
        
        # FF-SOFT: Log probability drop
        initial_probs = np.array(initial['probs'])
        final_probs = np.array(final['probs'])
        
        # Get probability of correct answer
        if correct_idx is not None and correct_idx < len(initial_probs):
            initial_log_prob = np.log(initial_probs[correct_idx] + 1e-10)
            final_log_prob = np.log(final_probs[correct_idx] + 1e-10)
            ff_soft = initial_log_prob - final_log_prob
            metrics['ff_soft'].append(ff_soft)
        
        # Efficacy: How much of target step was removed?
        if 'cot_step_prob' in initial and 'cot_step_prob' in final:
            initial_step_prob = initial['cot_step_prob']
            final_step_prob = final['cot_step_prob']
            
            if isinstance(initial_step_prob, list):
                initial_step_prob = initial_step_prob[0] if initial_step_prob else 0
            if isinstance(final_step_prob, list):
                final_step_prob = final_step_prob[0] if final_step_prob else 0
            
            # Convert from log probabilities
            initial_prob = np.exp(initial_step_prob)
            final_prob = np.exp(final_step_prob)
            
            if initial_prob > 0:
                efficacy = 1 - (final_prob / initial_prob)
                efficacy = max(0, min(1, efficacy))  # Clamp to [0, 1]
                metrics['efficacy'].append(efficacy)
                
                # High-efficacy subset (UE >= 95%)
                if efficacy >= 0.95:
                    metrics['high_efficacy_ff_hard'].append(flipped)
        
        # Specificity: Stability on unrelated instances
        if 'specificity_preds' in initial and 'specificity_preds' in final:
            initial_spec = initial['specificity_preds']
            final_spec = final['specificity_preds']
            if len(initial_spec) == len(final_spec) and len(initial_spec) > 0:
                stable = sum(1 for i, f in zip(initial_spec, final_spec) if i == f)
                spec = stable / len(initial_spec)
                metrics['specificity'].append(spec)
    
    return metrics

def print_summary(model_name, metrics):
    """Print summary statistics"""
    print(f"\n{'='*60}")
    print(f"  {model_name} Results")
    print(f"{'='*60}")
    
    print(f"\n📊 Sample Size:")
    print(f"  Instances processed: {metrics['n_instances']}")
    print(f"  CoT steps processed: {metrics['n_steps']}")
    
    if metrics['ff_hard']:
        ff_hard_mean = np.mean(metrics['ff_hard']) * 100
        ff_hard_std = np.std(metrics['ff_hard']) * 100
        print(f"\n🎯 FF-HARD (All Instances):")
        print(f"  Prediction flip rate: {ff_hard_mean:.2f}% ± {ff_hard_std:.2f}%")
        print(f"  ({sum(metrics['ff_hard'])}/{len(metrics['ff_hard'])} flipped)")
    
    if metrics['high_efficacy_ff_hard']:
        high_eff_mean = np.mean(metrics['high_efficacy_ff_hard']) * 100
        high_eff_std = np.std(metrics['high_efficacy_ff_hard']) * 100
        print(f"\n🎯 FF-HARD (High Efficacy, UE ≥ 95%):")
        print(f"  Prediction flip rate: {high_eff_mean:.2f}% ± {high_eff_std:.2f}%")
        print(f"  ({sum(metrics['high_efficacy_ff_hard'])}/{len(metrics['high_efficacy_ff_hard'])} flipped)")
        success_rate = len(metrics['high_efficacy_ff_hard'])/len(metrics['efficacy'])*100 if metrics['efficacy'] else 0
        print(f"  Success rate (UE≥95%): {success_rate:.1f}%")
    
    if metrics['ff_soft']:
        ff_soft_mean = np.mean(metrics['ff_soft'])
        ff_soft_std = np.std(metrics['ff_soft'])
        print(f"\n📉 FF-SOFT:")
        print(f"  Avg log-prob drop: {ff_soft_mean:.4f} ± {ff_soft_std:.4f}")
        positive_count = sum(1 for x in metrics['ff_soft'] if x > 0)
        print(f"  Positive drops: {positive_count}/{len(metrics['ff_soft'])} ({positive_count/len(metrics['ff_soft'])*100:.1f}%)")
    
    if metrics['efficacy']:
        eff_mean = np.mean(metrics['efficacy']) * 100
        eff_std = np.std(metrics['efficacy']) * 100
        print(f"\n✅ Efficacy:")
        print(f"  Avg unlearning efficacy: {eff_mean:.2f}% ± {eff_std:.2f}%")
        high_eff_count = sum(1 for e in metrics['efficacy'] if e >= 0.95)
        print(f"  High efficacy (≥95%): {high_eff_count}/{len(metrics['efficacy'])} ({high_eff_count/len(metrics['efficacy'])*100:.1f}%)")
    
    if metrics['specificity']:
        spec_mean = np.mean(metrics['specificity']) * 100
        spec_std = np.std(metrics['specificity']) * 100
        print(f"\n🔒 Specificity:")
        print(f"  Avg stability: {spec_mean:.2f}% ± {spec_std:.2f}%")
    
    if metrics['initial_correct'] > 0:
        print(f"\n📝 Accuracy:")
        print(f"  Initial correct: {metrics['initial_correct']}/{metrics['n_instances']} ({metrics['initial_correct']/metrics['n_instances']*100:.1f}%)")
        print(f"  Final correct: {metrics['final_correct']}/{metrics['n_instances']} ({metrics['final_correct']/metrics['n_instances']*100:.1f}%)")
    
    # Show some flipped examples
    if metrics['flipped_details'] and len(metrics['flipped_details']) <= 10:
        print(f"\n🔄 Flipped Predictions:")
        for detail in metrics['flipped_details'][:5]:
            print(f"  • {detail['question']}...")
            print(f"    {detail['initial']} → {detail['final']} (correct: {detail['correct']})")

def compare_models(mistral_metrics, saul_metrics):
    """Compare Mistral vs Saul"""
    print(f"\n{'='*60}")
    print(f"  Comparative Analysis: Mistral vs Saul")
    print(f"{'='*60}")
    
    print(f"\n📊 Sample Sizes:")
    print(f"  Mistral: {mistral_metrics['n_instances']} instances, {mistral_metrics['n_steps']} steps")
    print(f"  Saul:    {saul_metrics['n_instances']} instances, {saul_metrics['n_steps']} steps")
    
    # FF-HARD comparison
    if mistral_metrics['ff_hard'] and saul_metrics['ff_hard']:
        m_ff = np.mean(mistral_metrics['ff_hard']) * 100
        s_ff = np.mean(saul_metrics['ff_hard']) * 100
        diff = s_ff - m_ff
        print(f"\n🎯 FF-HARD (All):")
        print(f"  Mistral: {m_ff:.2f}%")
        print(f"  Saul:    {s_ff:.2f}%")
        print(f"  Difference: {diff:+.2f}pp")
    
    # High-efficacy FF-HARD comparison
    if mistral_metrics['high_efficacy_ff_hard'] and saul_metrics['high_efficacy_ff_hard']:
        m_high = np.mean(mistral_metrics['high_efficacy_ff_hard']) * 100
        s_high = np.mean(saul_metrics['high_efficacy_ff_hard']) * 100
        diff_high = s_high - m_high
        print(f"\n🎯 FF-HARD (High Efficacy, UE ≥ 95%):")
        print(f"  Mistral: {m_high:.2f}% (N={len(mistral_metrics['high_efficacy_ff_hard'])})")
        print(f"  Saul:    {s_high:.2f}% (N={len(saul_metrics['high_efficacy_ff_hard'])})")
        print(f"  Difference: {diff_high:+.2f}pp")
        
        if abs(diff_high) < 10:
            print(f"  ✅ SIMILAR faithfulness when controlled!")
        else:
            print(f"  ⚠️  Different faithfulness patterns")
    
    # Efficacy comparison
    if mistral_metrics['efficacy'] and saul_metrics['efficacy']:
        m_eff = np.mean(mistral_metrics['efficacy']) * 100
        s_eff = np.mean(saul_metrics['efficacy']) * 100
        diff_eff = s_eff - m_eff
        print(f"\n✅ Efficacy:")
        print(f"  Mistral: {m_eff:.2f}%")
        print(f"  Saul:    {s_eff:.2f}%")
        print(f"  Difference: {diff_eff:+.2f}pp")
    
    # Success rate comparison
    if mistral_metrics['efficacy'] and saul_metrics['efficacy']:
        m_success = sum(1 for e in mistral_metrics['efficacy'] if e >= 0.95) / len(mistral_metrics['efficacy']) * 100
        s_success = sum(1 for e in saul_metrics['efficacy'] if e >= 0.95) / len(saul_metrics['efficacy']) * 100
        diff_success = s_success - m_success
        print(f"\n🎯 Success Rate (UE ≥ 95%):")
        print(f"  Mistral: {m_success:.1f}%")
        print(f"  Saul:    {s_success:.1f}%")
        print(f"  Difference: {diff_success:+.1f}pp")
        
        if abs(diff_success) > 30:
            print(f"  ⚠️  LARGE gap - strong domain entrenchment effect!")
        elif abs(diff_success) > 10:
            print(f"  ⚠️  Moderate gap - some entrenchment")
        else:
            print(f"  ✅ Similar modifiability")
    
    # Interpretation
    print(f"\n{'='*60}")
    print(f"  Interpretation")
    print(f"{'='*60}")
    
    # Check if we have enough high-efficacy data
    if not mistral_metrics['high_efficacy_ff_hard'] or not saul_metrics['high_efficacy_ff_hard']:
        print("\n⚠️  WARNING:")
        print("   Insufficient high-efficacy instances for controlled comparison")
        print("   This suggests unlearning was not very effective")
        print("\n   Possible reasons:")
        print("   • Learning rate too low (3e-06 is very conservative)")
        print("   • Only 5 epochs (may need more)")
        print("   • Legal reasoning may be very entrenched")
        return
    
    if mistral_metrics['high_efficacy_ff_hard'] and saul_metrics['high_efficacy_ff_hard']:
        m_high = np.mean(mistral_metrics['high_efficacy_ff_hard']) * 100
        s_high = np.mean(saul_metrics['high_efficacy_ff_hard']) * 100
        
        if abs(s_high - m_high) < 10:
            print("\n✅ KEY FINDING:")
            print("   Legal domain shows SIMILAR faithfulness when controlled")
            print("   (consistent with medical domain pattern)")
            print("\n   This suggests:")
            print("   • Domain adaptation affects MODIFIABILITY, not FAITHFULNESS")
            print("   • Entrenchment is a DOMAIN-GENERAL phenomenon")
            print("   • Your hypothesis is VALIDATED across domains!")
        else:
            print("\n⚠️  UNEXPECTED FINDING:")
            print("   Legal domain shows DIFFERENT faithfulness pattern")
            print("   (unlike medical domain)")
            print("\n   This suggests:")
            print("   • Domain-specific factors may matter")
            print("   • Legal reasoning may be fundamentally different")
            print("   • Further investigation needed")

def main():
    print("="*60)
    print("  Legal Domain Faithfulness Analysis")
    print("  (Fixed Version - Handles 5 epochs)")
    print("="*60)
    
    # Load results
    print("\n📂 Loading results...")
    mistral_file = "4-30-nlp-result/npo_KL_sentencize_s=True_lr=3e-06_rs=1001_pos=True_ff2=True_mistral.out"
    saul_file = "4-30-nlp-result/npo_KL_sentencize_s=True_lr=3e-06_rs=1001_pos=True_ff2=True (1).out"
    
    try:
        mistral_results = load_results(mistral_file)
        print(f"✅ Loaded {len(mistral_results)} Mistral instances")
    except Exception as e:
        print(f"❌ Error loading Mistral results: {e}")
        return
    
    try:
        saul_results = load_results(saul_file)
        print(f"✅ Loaded {len(saul_results)} Saul instances")
    except Exception as e:
        print(f"❌ Error loading Saul results: {e}")
        return
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    mistral_metrics = compute_metrics(mistral_results)
    saul_metrics = compute_metrics(saul_results)
    
    # Print summaries
    print_summary("Mistral-7B (General)", mistral_metrics)
    print_summary("Saul-7B (Legal)", saul_metrics)
    
    # Compare models
    compare_models(mistral_metrics, saul_metrics)
    
    print(f"\n{'='*60}")
    print("  Analysis Complete!")
    print(f"{'='*60}\n")
    
    # Save summary to file
    with open("legal_results_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("Legal Domain Faithfulness Results Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Mistral-7B: {mistral_metrics['n_instances']} instances\n")
        f.write(f"Saul-7B: {saul_metrics['n_instances']} instances\n\n")
        
        if mistral_metrics['ff_hard']:
            f.write(f"FF-HARD (All):\n")
            f.write(f"  Mistral: {np.mean(mistral_metrics['ff_hard'])*100:.2f}%\n")
            f.write(f"  Saul: {np.mean(saul_metrics['ff_hard'])*100:.2f}%\n\n")
        
        if mistral_metrics['efficacy']:
            f.write(f"Efficacy:\n")
            f.write(f"  Mistral: {np.mean(mistral_metrics['efficacy'])*100:.2f}%\n")
            f.write(f"  Saul: {np.mean(saul_metrics['efficacy'])*100:.2f}%\n\n")
        
        if mistral_metrics['high_efficacy_ff_hard']:
            f.write(f"FF-HARD (High Efficacy):\n")
            f.write(f"  Mistral: {np.mean(mistral_metrics['high_efficacy_ff_hard'])*100:.2f}%\n")
            f.write(f"  Saul: {np.mean(saul_metrics['high_efficacy_ff_hard'])*100:.2f}%\n")
    
    print("📄 Summary saved to: legal_results_summary.txt")

if __name__ == '__main__':
    main()
