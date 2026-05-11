#!/usr/bin/env python3
"""
Run faithfulness experiment on Quartz GPU cluster.
Adapted from the Colab notebook for batch execution.
"""

import os
import sys
import json
import argparse
import importlib.util

# Setup paths
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def register_custom_dataset(register_file):
    """Register a custom dataset handler."""
    if not os.path.exists(register_file):
        print(f"Warning: {register_file} not found, skipping registration")
        return
    
    spec = importlib.util.spec_from_file_location("register_dataset", register_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"✅ Custom dataset registered from {register_file}")

def main():
    parser = argparse.ArgumentParser(description='Run faithfulness experiment')
    
    # Model and dataset
    parser.add_argument('--model_name', type=str, required=True,
                       help='HuggingFace model name')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (must be registered in dataload.py)')
    parser.add_argument('--register_dataset', type=str, default=None,
                       help='Python file to register custom dataset')
    
    # Unlearning method
    parser.add_argument('--method', type=str, default='npo_KL',
                       choices=['npo_KL', 'npo', 'grad_ascent', 'grad_diff'],
                       help='Unlearning method')
    parser.add_argument('--strategy', type=str, default='sentencize',
                       choices=['sentencize', 'atomic', 'full'],
                       help='Segmentation strategy')
    parser.add_argument('--stepwise', action='store_true',
                       help='Unlearn each step separately')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=1001,
                       help='Random seed')
    
    # Model modifications
    parser.add_argument('--pos', action='store_true',
                       help='Only unlearn content words (POS filtering)')
    parser.add_argument('--ff2', action='store_true',
                       help='Only update FF2 (MLP down_proj) layers')
    
    # Experiment settings
    parser.add_argument('--max_instances', type=int, default=30,
                       help='Maximum number of instances to process')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature for CoT generation')
    parser.add_argument('--new_cot', action='store_true',
                       help='Force regenerate CoTs')
    parser.add_argument('--atomic', action='store_true',
                       help='Use atomic segmentation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Faithfulness Experiment Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*60)
    
    # Register custom dataset if provided
    if args.register_dataset:
        register_custom_dataset(args.register_dataset)
    
    # Import after dataset registration
    import torch
    from transformers import AutoTokenizer
    from dataload import DATASETS
    from data import load_or_generate_dataset_cots
    from util import set_random_seed
    from run_unlearning import run_unlearning_pipeline
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'\n✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)\n')
    else:
        print('\n⚠️ No GPU detected! This will be very slow.\n')
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset handler
    if args.dataset not in DATASETS:
        print(f"❌ Error: Dataset '{args.dataset}' not found in DATASETS")
        print(f"Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)
    
    DH = DATASETS[args.dataset]
    print(f"✅ Dataset handler loaded: {args.dataset}")
    
    # Generate or load CoTs
    print(f"\n📝 Loading/generating CoTs for {args.model_name} on {args.dataset}...")
    cot_data = load_or_generate_dataset_cots(
        model_id=args.model_name,
        tokenizer=tokenizer,
        dataset_id=args.dataset,
        force_generate=args.new_cot,
        sentencize=(args.strategy == 'sentencize'),
        temperature=args.temperature,
        seed=args.seed,
        atomic=args.atomic
    )
    
    print(f"✅ Loaded {len(cot_data)} instances with CoTs")
    
    # Split data
    import random
    random.shuffle(cot_data)
    N_verify = 20
    cots_train = cot_data[:-N_verify]
    cots_verify = cot_data[-N_verify:]
    
    # Setup results directory
    short_model = args.model_name.split('/')[-1].replace('-', '_').lower()
    resdir = f'results/{args.dataset}/{short_model}/'
    os.makedirs(resdir, exist_ok=True)
    
    logfile_name = (f"{args.method}_{args.strategy}_s={args.stepwise}"
                    f"_lr={args.lr}_rs={args.seed}"
                    f"_pos={args.pos}_ff2={args.ff2}.out")
    
    result_file = os.path.join(resdir, logfile_name)
    
    print(f"\n📂 Results will be saved to: {result_file}")
    print(f"🎯 Processing up to {args.max_instances} instances\n")
    
    # Run unlearning pipeline
    from run_unlearning import run_unlearning_pipeline
    
    run_unlearning_pipeline(
        model_id=args.model_name,
        tokenizer=tokenizer,
        dataset_handler=DH,
        cots_train=cots_train[:args.max_instances],
        cots_verify=cots_verify,
        result_file=result_file,
        config=args
    )
    
    print("\n" + "="*60)
    print("✅ Experiment completed successfully!")
    print(f"📂 Results saved to: {result_file}")
    print("="*60)

if __name__ == '__main__':
    main()
