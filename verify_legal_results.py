#!/usr/bin/env python3
"""
Verify and compare legal domain faithfulness results.
Analyzes Mistral-7B vs Saul-7B on LegalBench.
"""
import json
import numpy as np
from scipy import stats
import os

def load_results(path):
    """Load results from .out file."""
    results = []
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        return results
    
    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping line {line_num} (JSON parse error): {e}")
    return results

def analyze(results, label, final_epoch='5'):
    """Compute all metrics from raw results."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {label}")
    print(f"{'='*70}")
    
    n_total = len(results)
    print(f"  Total result lines: {n_total}")
    
    # Count unique questions
    unique_questions = set()
    for r in results:
        qid = r.get('id', r.get('question', ''))
        unique_questions.add(qid)
    print(f"  Unique questions: {len(unique_questions)}")
    
    # Valid instances
    valid = [r for r in results if r.get('unlearning_results')]
    print(f"  Instances with unlearning results: {len(valid)}")
    
    if not valid:
        print("  No valid results to analyze!")
        return None
    
    # ---- FF-HARD ----
    flips = []
    for r in valid:
        ur = r['unlearning_results']
        initial_pred = r['cot_prediction']
        final_key = str(final_epoch)
        if final_key not in ur:
            final_key = str(max(int(k) for k in ur.keys()))
        final_pred = ur[final_key]['prediction']
        flips.append(int(initial_pred != final_pred))
    
    flips = np.array(flips)
    ff_hard = np.mean(flips) * 100
    print(f"\n  FF-HARD: {ff_hard:.2f}% ({sum(flips)}/{len(flips)})")
    
    # ---- Efficacy ----
    efficacy_values = []
    for r in valid:
        ur = r['unlearning_results']
        initial_step_prob = ur['0'].get('cot_step_prob', None)
        final_key = str(final_epoch)
        if final_key not in ur:
            final_key = str(max(int(k) for k in ur.keys()))
        final_step_prob = ur[final_key].get('cot_step_prob', None)
        
        if initial_step_prob is not None and final_step_prob is not None:
            if isinstance(initial_step_prob, list):
                init_lp = initial_step_prob[0] if len(initial_step_prob) > 0 else None
            else:
                init_lp = initial_step_prob
            if isinstance(final_step_prob, list):
                final_lp = final_step_prob[0] if len(final_step_prob) > 0 else None
            else:
                final_lp = final_step_prob
            
            if init_lp is not None and final_lp is not None:
                ue = 1 - np.exp(final_lp - init_lp)
                efficacy_values.append(ue)
    
    efficacy = np.array(efficacy_values)
    print(f"\n  Efficacy: {np.mean(efficacy)*100:.2f}%")
    n_high = sum(efficacy >= 0.95)
    print(f"  High efficacy (UE >= 95%): {n_high}/{len(efficacy)} ({n_high/len(efficacy)*100:.1f}%)")
    
    # ---- Specificity ----
    spec_values = []
    for r in valid:
        ur = r['unlearning_results']
        initial_spec = ur['0'].get('specificity_preds', [])
        final_key = str(final_epoch)
        if final_key not in ur:
            final_key = str(max(int(k) for k in ur.keys()))
        final_spec = ur[final_key].get('specificity_preds', [])
        
        if initial_spec and final_spec:
            matches = sum(1 for a, b in zip(initial_spec, final_spec) if a == b)
            spec_values.append(matches / len(initial_spec))
    
    specificity = np.array(spec_values) if spec_values else None
    if specificity is not None:
        print(f"\n  Specificity: {np.mean(specificity)*100:.2f}%")
    
    return {
        'flips': flips,
        'ff_hard': ff_hard,
        'efficacy': efficacy,
        'specificity': specificity,
        'n_valid': len(valid),
        'n_unique_q': len(unique_questions),
    }

# ============================================================
# Main analysis
# ============================================================
print("="*70)
print("  LEGAL DOMAIN FAITHFULNESS ANALYSIS")
print("  Mistral-7B vs Saul-7B on LegalBench")
print("="*70)

# Find result files
mistral_path = "results/legalbench/mistral_7b_instruct_v0.2/npo_KL_sentencize_s=True_lr=5e-06_rs=1001_pos=True_ff2=True.out"
saul_path = "results/legalbench/saul_7b_instruct_v1/npo_KL_sentencize_s=True_lr=5e-06_rs=1001_pos=True_ff2=True.out"

print(f"\nLoading Mistral results from: {mistral_path}")
mistral_results = load_results(mistral_path)
print(f"  Loaded {len(mistral_results)} lines")

print(f"\nLoading Saul results from: {saul_path}")
saul_results = load_results(saul_path)
print(f"  Loaded {len(saul_results)} lines")

if not mistral_results and not saul_results:
    print("\n❌ No results found! Make sure the experiment has completed.")
    exit(1)

mistral_stats = analyze(mistral_results, "Mistral-7B-Instruct-v0.2") if mistral_results else None
saul_stats = analyze(saul_results, "Saul-7B-Instruct-v1") if saul_results else None

# ============================================================
# Comparative Analysis
# ============================================================
if mistral_stats and saul_stats:
    print(f"\n{'='*70}")
    print("  COMPARATIVE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n--- FF-HARD Comparison ---")
    print(f"  Mistral:  {mistral_stats['ff_hard']:.2f}%")
    print(f"  Saul:     {saul_stats['ff_hard']:.2f}%")
    print(f"  Difference: {saul_stats['ff_hard'] - mistral_stats['ff_hard']:.2f}%")
    
    t_stat, p_val = stats.ttest_ind(mistral_stats['flips'], saul_stats['flips'])
    print(f"  t-test p-value: {p_val:.4f}")
    
    print(f"\n--- Efficacy Comparison ---")
    print(f"  Mistral:  {np.mean(mistral_stats['efficacy'])*100:.2f}%")
    print(f"  Saul:     {np.mean(saul_stats['efficacy'])*100:.2f}%")
    print(f"  Difference: {(np.mean(saul_stats['efficacy']) - np.mean(mistral_stats['efficacy']))*100:.2f}%")
    
    t_stat_eff, p_val_eff = stats.ttest_ind(mistral_stats['efficacy'], saul_stats['efficacy'])
    print(f"  t-test p-value: {p_val_eff:.6f}")
    
    if mistral_stats['specificity'] is not None and saul_stats['specificity'] is not None:
        print(f"\n--- Specificity Comparison ---")
        print(f"  Mistral:  {np.mean(mistral_stats['specificity'])*100:.2f}%")
        print(f"  Saul:     {np.mean(saul_stats['specificity'])*100:.2f}%")

print(f"\n{'='*70}")
print("  ANALYSIS COMPLETE")
print(f"{'='*70}")
