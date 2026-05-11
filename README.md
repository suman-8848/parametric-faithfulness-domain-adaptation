# Parametric Faithfulness in Domain-Adapted Models

Final project for NLP course, Spring 2026

## Team
- Madhavan Balaji
- Suman Lamsal
- Samuditha Wijayasundara

Indiana University Indianapolis

## What is this?

We extended the FUR (Faithfulness by Unlearning Reasoning Steps) method to domain-adapted language models. Basically, we wanted to see if models trained on medical or legal text have different reasoning patterns than general models.

## Main Finding

Domain adaptation doesn't make reasoning less faithful, but it does make it harder to modify. We call this "reasoning entrenchment" - the reasoning is still causally linked to predictions, but it's more deeply embedded in the parameters.

## Files

- `parametric_faithfulness_medical_extension.ipynb` - MedQA experiments
- `parametric_faithfulness_legal_extension.ipynb` - LegalBench experiments  
- `scripts/` - Shell scripts for running on compute clusters
- `report/Team_3_NLP_Final_Project_Report.pdf` - Full report
- `results/` - Experimental results and visualizations

## Results

**MedQA:** When we control for unlearning success, BioMistral-7B and Mistral-7B have identical faithfulness (~67%, p=0.96). But BioMistral's reasoning is much harder to unlearn.

**LegalBench:** Saul-7B works fine, but Mistral-7B fails catastrophically on legal text with the same hyperparameters.

## Setup

```bash
pip install torch transformers datasets accelerate
```

Run the notebooks. Models download automatically from HuggingFace.

## Credits

Built on the FUR methodology from [Tutek et al. (2025)](https://github.com/technion-cs-nlp/parametric-faithfulness)

Compute: Indiana University Big Red 200
