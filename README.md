# Agentic Explainable AI for Multiple Sclerosis Disability Classification

This repository contains the implementation developed for my MSc Artificial Intelligence dissertation at Birmingham City University.

The project investigates an Agentic Explainable AI framework for Multiple Sclerosis (MS) disability classification. It combines clinical variables and MRI-derived lesion biomarkers with machine learning, traditional explainability methods and an Agentic AI explanation framework.

## Project Overview

The research investigates whether Agentic AI can complement traditional Explainable AI (XAI) techniques by translating model outputs and feature-level explanations into structured, evidence-supported natural-language explanations.

The framework incorporates:

- Clinical and MRI-derived biomarker processing
- Machine learning model comparison and optimisation
- SHAP
- LIME
- Permutation importance
- Retrieval-Augmented Generation (RAG)
- Llama 3 8B for local explanation generation
- Faithfulness validation
- Validator-guided self-correction

## Experimental Framework

The project was evaluated through six experiments:

1. Baseline Model and Modality Comparison
2. Model Refinement and Hyperparameter Optimisation
3. Explainable Artificial Intelligence Analysis
4. Agentic AI Explanation Framework
5. Agentic AI RAG Ablation Evaluation
6. Self-Correcting Agentic AI Evaluation

## Repository Structure

```text
MS_Dissertation_Project/
├── docs/                  # Supporting project documentation
├── outputs/               # Experimental results, figures and tables
├── src/                   # Main implementation
├── .gitignore
├── README.md
└── requirements.txt

