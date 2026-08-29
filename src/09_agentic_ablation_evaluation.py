# ============================================================
# EXPERIMENT 5
# COMPARATIVE AND ABLATION EVALUATION OF AGENTIC XAI
# ============================================================

from pathlib import Path
import json
import pandas as pd
import requests

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_04_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "experiment_04_agentic_explanation_comparison"
)

EXPERIMENT_05_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "experiment_05_agentic_ablation_comparison"
)

WITH_RAG_DIR = EXPERIMENT_05_DIR / "with_rag"
WITHOUT_RAG_DIR = EXPERIMENT_05_DIR / "without_rag"
EVALUATION_DIR = EXPERIMENT_05_DIR / "evaluation"
FIGURES_DIR = EXPERIMENT_05_DIR / "figures"
TABLES_DIR = EXPERIMENT_05_DIR / "tables"
LOGS_DIR = EXPERIMENT_05_DIR / "logs"


# ============================================================
# CREATE / VERIFY OUTPUT DIRECTORIES
# ============================================================

experiment_05_directories = [
    WITH_RAG_DIR,
    WITHOUT_RAG_DIR,
    EVALUATION_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOGS_DIR,
]

for directory in experiment_05_directories:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

PATIENT_IDS = [7, 11, 37]

OLLAMA_MODEL = "llama3.1:8b"
GENERATION_TEMPERATURE = 0.2
GENERATION_SEED = 42
TOP_K_EVIDENCE = 3


# ============================================================
# INITIAL EXPERIMENT CHECK
# ============================================================

print("=" * 65)
print("EXPERIMENT 5: AGENTIC XAI ABLATION EVALUATION")
print("=" * 65)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nExperiment 4 source directory:")
print(EXPERIMENT_04_DIR)

print("\nExperiment 5 output directory:")
print(EXPERIMENT_05_DIR)

print("\nRepresentative patient IDs:")
print(PATIENT_IDS)

print("\nGeneration settings:")
print(f"Model: {OLLAMA_MODEL}")
print(f"Temperature: {GENERATION_TEMPERATURE}")
print(f"Seed: {GENERATION_SEED}")
print(f"Top-k evidence: {TOP_K_EVIDENCE}")

print("\nExperiment 5 directories:")

for directory in experiment_05_directories:
    print(f"  [OK] {directory.name}")

print("\n" + "=" * 65)
print("EXPERIMENT 5 INITIALISATION COMPLETE")
print("=" * 65)

# ============================================================
# STEP 2: LOAD AND VERIFY EXPERIMENT 4 PATIENT OUTPUTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: VERIFYING EXPERIMENT 4 PATIENT OUTPUTS")
print("=" * 65)

experiment_04_patient_data = {}

for patient_id in PATIENT_IDS:

    patient_dir = EXPERIMENT_04_DIR / f"patient_{patient_id}"

    explanation_file = (
        patient_dir
        / f"patient_{patient_id}_agentic_explanation.txt"
    )

    prompt_file = (
        patient_dir
        / f"patient_{patient_id}_agentic_prompt.txt"
    )

    metadata_file = (
        patient_dir
        / f"patient_{patient_id}_generation_metadata.json"
    )

    shap_file = (
        patient_dir
        / f"patient_{patient_id}_top_shap_features.csv"
    )

    evidence_file = (
        patient_dir
        / f"patient_{patient_id}_top_clinical_evidence.csv"
    )

    validation_file = (
        patient_dir
        / f"patient_{patient_id}_faithfulness_validation.json"
    )

    required_files = {
        "explanation": explanation_file,
        "prompt": prompt_file,
        "metadata": metadata_file,
        "shap": shap_file,
        "evidence": evidence_file,
        "validation": validation_file,
    }

    print(f"\nPatient {patient_id}:")

    all_files_present = True

    for file_type, file_path in required_files.items():

        if file_path.exists():
            print(f"  [OK] {file_type}: {file_path.name}")

        else:
            print(f"  [MISSING] {file_type}: {file_path.name}")
            all_files_present = False

    if not all_files_present:
        raise FileNotFoundError(
            f"Required Experiment 4 files are missing "
            f"for Patient {patient_id}."
        )

    # Load explanation
    explanation = explanation_file.read_text(
        encoding="utf-8"
    )

    # Load original Agentic AI prompt
    prompt = prompt_file.read_text(
        encoding="utf-8"
    )

    # Load generation metadata
    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    # Load SHAP features
    shap_df = pd.read_csv(shap_file)

    # Load retrieved clinical evidence
    evidence_df = pd.read_csv(evidence_file)

    # Load faithfulness validation
    with open(
        validation_file,
        "r",
        encoding="utf-8"
    ) as file:
        validation = json.load(file)

    experiment_04_patient_data[patient_id] = {
        "explanation": explanation,
        "prompt": prompt,
        "metadata": metadata,
        "shap": shap_df,
        "evidence": evidence_df,
        "validation": validation,
    }

    print(
        f"  Loaded SHAP features: {len(shap_df)}"
    )

    print(
        f"  Loaded clinical evidence chunks: {len(evidence_df)}"
    )

    print(
        "  Faithfulness status:",
        validation.get("Validation_Status", "Not found")
    )


# ============================================================
# STEP 2 COMPLETION CHECK
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT 4 DATA LOADING SUMMARY")
print("=" * 65)

print(
    "Patients successfully loaded:",
    list(experiment_04_patient_data.keys())
)

print(
    "Number of patient cases:",
    len(experiment_04_patient_data)
)

print("\n" + "=" * 65)
print("STEP 2 COMPLETE")
print("=" * 65)

# ============================================================
# STEP 3: BUILD CONTROLLED NO-RAG PROMPTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: BUILDING NO-RAG PROMPTS")
print("=" * 65)


def build_no_rag_prompt(
    patient_id,
    metadata,
    shap_df
):
    """
    Build an Agentic AI explanation prompt without retrieved
    clinical evidence.

    This creates the ablation condition for Experiment 5.
    """

    predicted_probability = float(
        metadata["Prediction_Probability"]
    )

    predicted_class = int(
        metadata["Predicted_Class"]
    )

    actual_class = int(
        metadata["Actual_Class"]
    )

    # --------------------------------------------------------
    # Build SHAP context
    # --------------------------------------------------------

    shap_lines = []

    for _, row in shap_df.iterrows():

        feature = row["Feature"]
        shap_value = float(
            row["SHAP_Value"]
        )

        if shap_value > 0:
            direction = (
                "towards moderate/high disability"
            )

        elif shap_value < 0:
            direction = (
                "towards low disability"
            )

        else:
            direction = (
                "with no directional contribution"
            )

        shap_lines.append(
            f"{feature}: SHAP contribution "
            f"{shap_value:.4f}, {direction}"
        )

    shap_context = "\n".join(
        shap_lines
    )

    # --------------------------------------------------------
    # Construct controlled No-RAG prompt
    # --------------------------------------------------------

    no_rag_prompt = f"""
You are an AI explanation agent supporting interpretation of a
machine-learning prediction for Multiple Sclerosis disability.

IMPORTANT RULES:

1. Do not diagnose the patient.
2. Do not change or override the machine-learning prediction.
3. Use only the model information and SHAP contributions provided below.
4. Do not invent clinical evidence, guidelines or research findings.
5. Clearly distinguish model-derived information from interpretation.
6. Acknowledge uncertainty when the prediction probability is close
   to the classification threshold.
7. For probabilities within +/- 0.05 of 0.50, explicitly describe
   the prediction as borderline.
8. Explain conflicting SHAP feature contributions where relevant.
9. Keep the explanation concise and suitable for decision support.
10. State that the output is intended to support, not replace,
    professional clinical judgement.
11. Do not interpret the predicted disability class as a diagnosis,
    disease progression assessment or direct measure of neurological
    impairment.
12. Do not recommend diagnosis, treatment or management.
13. When describing SHAP contributions, use the supplied direction
    exactly.
14. Do not infer whether an underlying feature value is clinically
    high, low, better, worse, younger or older from SHAP direction alone.

MODEL PREDICTION

Patient ID: {patient_id}
Actual class: {actual_class}
Predicted class: {predicted_class}
Predicted probability of moderate/high disability:
{predicted_probability:.4f}

PATIENT-SPECIFIC MODEL EXPLANATION

{shap_context}

IMPORTANT ABLATION CONDITION

No retrieved clinical evidence has been supplied for this explanation.

Do not introduce clinical guidelines, research evidence or other
external clinical information.

TASK

Generate a structured explanation containing:

1. Prediction Summary
2. Main Factors Supporting Moderate/High Disability
3. Main Factors Supporting Low Disability
4. Model-Based Context
5. Uncertainty and Limitations
6. Decision-Support Summary

The explanation must remain faithful to the supplied model output
and SHAP contributions.
""".strip()

    return no_rag_prompt


# ============================================================
# GENERATE AND SAVE NO-RAG PROMPTS
# ============================================================

no_rag_prompts = {}

for patient_id in PATIENT_IDS:

    patient_data = (
        experiment_04_patient_data[
            patient_id
        ]
    )

    no_rag_prompt = build_no_rag_prompt(
        patient_id=patient_id,
        metadata=patient_data["metadata"],
        shap_df=patient_data["shap"],
    )

    no_rag_prompts[patient_id] = (
        no_rag_prompt
    )

    prompt_output_file = (
        WITHOUT_RAG_DIR
        / f"patient_{patient_id}_no_rag_prompt.txt"
    )

    prompt_output_file.write_text(
        no_rag_prompt,
        encoding="utf-8"
    )

    print(
        f"\nPatient {patient_id} No-RAG prompt created."
    )

    print(
        f"Saved to: {prompt_output_file}"
    )


# ============================================================
# STEP 3 COMPLETION CHECK
# ============================================================

print("\n" + "=" * 65)
print("NO-RAG PROMPT SUMMARY")
print("=" * 65)

print(
    "No-RAG prompts created for:",
    list(no_rag_prompts.keys())
)

print(
    "Number of prompts:",
    len(no_rag_prompts)
)

print("\n" + "=" * 65)
print("STEP 3 COMPLETE")
print("=" * 65)

# ============================================================
# STEP 4: GENERATE NO-RAG AGENTIC AI EXPLANATIONS
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: GENERATING NO-RAG EXPLANATIONS")
print("=" * 65)


def generate_no_rag_explanation(prompt):
    """
    Generate a controlled No-RAG explanation using the same
    Ollama model and generation settings as Experiment 4.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": GENERATION_TEMPERATURE,
            "seed": GENERATION_SEED,
        },
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    response_data = response.json()

    explanation = (
        response_data["response"].strip()
    )

    return explanation


# ============================================================
# GENERATE EXPLANATIONS FOR ALL THREE PATIENTS
# ============================================================

no_rag_explanations = {}

for patient_id in PATIENT_IDS:

    print(
        f"\nGenerating No-RAG explanation "
        f"for Patient {patient_id}..."
    )

    explanation = generate_no_rag_explanation(
        no_rag_prompts[patient_id]
    )

    no_rag_explanations[patient_id] = (
        explanation
    )

    explanation_output_file = (
        WITHOUT_RAG_DIR
        / f"patient_{patient_id}_no_rag_explanation.txt"
    )

    explanation_output_file.write_text(
        explanation,
        encoding="utf-8"
    )

    print(
        f"Patient {patient_id} explanation generated."
    )

    print(
        f"Saved to: {explanation_output_file}"
    )


# ============================================================
# STEP 4 COMPLETION CHECK
# ============================================================

print("\n" + "=" * 65)
print("NO-RAG GENERATION SUMMARY")
print("=" * 65)

print(
    "No-RAG explanations generated for:",
    list(no_rag_explanations.keys())
)

print(
    "Number of explanations:",
    len(no_rag_explanations)
)

print("\n" + "=" * 65)
print("STEP 4 COMPLETE")
print("=" * 65)

# ============================================================
# STEP 5A: LOAD WITH-RAG EXPLANATIONS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5A: LOADING WITH-RAG EXPLANATIONS")
print("=" * 65)

with_rag_explanations = {}

for patient_id in PATIENT_IDS:

    patient_data = experiment_04_patient_data[patient_id]

    explanation = patient_data["explanation"]

    with_rag_explanations[patient_id] = explanation

    output_file = (
        WITH_RAG_DIR
        / f"patient_{patient_id}_with_rag_explanation.txt"
    )

    output_file.write_text(
        explanation,
        encoding="utf-8"
    )

    print(
        f"[OK] Patient {patient_id}: "
        f"{output_file.name}"
    )


print("\nWith-RAG explanations loaded for:")
print(list(with_rag_explanations.keys()))

print(
    "Number of With-RAG explanations:",
    len(with_rag_explanations)
)

print("\n" + "=" * 65)
print("STEP 5A COMPLETE")
print("=" * 65)

# ============================================================
# STEP 5B: QUANTITATIVE WITH-RAG VS NO-RAG COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("STEP 5B: QUANTITATIVE ABLATION COMPARISON")
print("=" * 65)


def normalise_feature_name(feature):
    """
    Convert feature names into a simple form for checking
    whether they appear in generated explanation text.
    """
    return (
        str(feature)
        .lower()
        .replace("_", " ")
        .strip()
    )


def calculate_shap_feature_coverage(explanation, shap_df):
    """
    Calculate the proportion of supplied top SHAP features
    explicitly mentioned in an explanation.
    """

    explanation_normalised = (
        explanation.lower().replace("_", " ")
    )

    features = [
        normalise_feature_name(feature)
        for feature in shap_df["Feature"].tolist()
    ]

    matched_features = [
        feature
        for feature in features
        if feature in explanation_normalised
    ]

    total_features = len(features)

    coverage = (
        len(matched_features) / total_features
        if total_features > 0
        else 0.0
    )

    return coverage, matched_features


def calculate_section_coverage(explanation):
    """
    Measure presence of the six expected Agentic AI
    decision-support sections.
    """

    expected_sections = [
        "prediction summary",
        "patient-specific factors",
        "clinical evidence",
        "integrated interpretation",
        "limitations",
        "decision-support summary",
    ]

    explanation_lower = explanation.lower()

    matched_sections = [
        section
        for section in expected_sections
        if section in explanation_lower
    ]

    coverage = (
        len(matched_sections) / len(expected_sections)
    )

    return coverage, matched_sections


def count_evidence_mentions(explanation, evidence_df):
    """
    Count unique retrieved evidence identifiers explicitly
    mentioned in the generated explanation.
    """

    explanation_lower = explanation.lower()

    evidence_ids = []

    if "Chunk_ID" in evidence_df.columns:
        evidence_ids = (
            evidence_df["Chunk_ID"]
            .astype(str)
            .tolist()
        )

    matched_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id.lower() in explanation_lower
    ]

    return len(set(matched_ids)), list(set(matched_ids))


def count_faithfulness_flags(explanation):
    """
    Count predefined potentially unsupported directional
    interpretations, using the same principle as Experiment 4.
    """

    unsupported_terms = [
        "younger",
        "older",
        "better",
        "worse",
        "higher age",
        "lower age",
        "improved",
        "poorer",
        "increased",
        "decreased",
    ]

    explanation_lower = explanation.lower()

    matched_terms = [
        term
        for term in unsupported_terms
        if term in explanation_lower
    ]

    return len(matched_terms), matched_terms


# ============================================================
# BUILD PATIENT-LEVEL COMPARISON TABLE
# ============================================================

ablation_results = []

for patient_id in PATIENT_IDS:

    patient_data = experiment_04_patient_data[patient_id]

    shap_df = patient_data["shap"]
    evidence_df = patient_data["evidence"]

    conditions = {
        "With-RAG": with_rag_explanations[patient_id],
        "No-RAG": no_rag_explanations[patient_id],
    }

    for condition, explanation in conditions.items():

        word_count = len(explanation.split())

        shap_coverage, matched_features = (
            calculate_shap_feature_coverage(
                explanation,
                shap_df
            )
        )

        section_coverage, matched_sections = (
            calculate_section_coverage(
                explanation
            )
        )

        evidence_count, matched_evidence = (
            count_evidence_mentions(
                explanation,
                evidence_df
            )
        )

        flag_count, matched_flags = (
            count_faithfulness_flags(
                explanation
            )
        )

        ablation_results.append(
            {
                "Patient_ID": patient_id,
                "Condition": condition,
                "Word_Count": word_count,
                "SHAP_Feature_Coverage": round(
                    shap_coverage, 3
                ),
                "SHAP_Features_Matched": "; ".join(
                    matched_features
                ),
                "Section_Coverage": round(
                    section_coverage, 3
                ),
                "Sections_Matched": "; ".join(
                    matched_sections
                ),
                "Evidence_ID_Mentions": evidence_count,
                "Evidence_IDs_Matched": "; ".join(
                    matched_evidence
                ),
                "Faithfulness_Flags": flag_count,
                "Flagged_Terms": "; ".join(
                    matched_flags
                ),
            }
        )


ablation_results_df = pd.DataFrame(
    ablation_results
)


# ============================================================
# SAVE RESULTS
# ============================================================

patient_results_file = (
    TABLES_DIR
    / "experiment_05_patient_level_ablation_results.csv"
)

ablation_results_df.to_csv(
    patient_results_file,
    index=False
)


print("\nPatient-level ablation results:")
print(
    ablation_results_df.to_string(
        index=False
    )
)

print(
    "\nPatient-level results saved to:"
)
print(patient_results_file)


print("\n" + "=" * 65)
print("STEP 5B COMPLETE")
print("=" * 65)

# ============================================================
# STEP 5C: AGGREGATE WITH-RAG VS NO-RAG RESULTS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5C: AGGREGATE ABLATION RESULTS")
print("=" * 65)


# ============================================================
# AGGREGATE NUMERIC METRICS BY CONDITION
# ============================================================

aggregate_results_df = (
    ablation_results_df
    .groupby("Condition", as_index=False)
    .agg(
        Mean_Word_Count=(
            "Word_Count",
            "mean"
        ),
        Mean_SHAP_Feature_Coverage=(
            "SHAP_Feature_Coverage",
            "mean"
        ),
        Mean_Section_Coverage=(
            "Section_Coverage",
            "mean"
        ),
        Mean_Evidence_ID_Mentions=(
            "Evidence_ID_Mentions",
            "mean"
        ),
        Mean_Faithfulness_Flags=(
            "Faithfulness_Flags",
            "mean"
        ),
        Total_Evidence_ID_Mentions=(
            "Evidence_ID_Mentions",
            "sum"
        ),
        Total_Faithfulness_Flags=(
            "Faithfulness_Flags",
            "sum"
        ),
    )
)


# ============================================================
# ROUND MEAN VALUES FOR REPORTING
# ============================================================

mean_columns = [
    "Mean_Word_Count",
    "Mean_SHAP_Feature_Coverage",
    "Mean_Section_Coverage",
    "Mean_Evidence_ID_Mentions",
    "Mean_Faithfulness_Flags",
]

aggregate_results_df[mean_columns] = (
    aggregate_results_df[mean_columns]
    .round(3)
)


# ============================================================
# ADD PERCENTAGE VERSIONS OF COVERAGE METRICS
# ============================================================

aggregate_results_df[
    "Mean_SHAP_Feature_Coverage_Percent"
] = (
    aggregate_results_df[
        "Mean_SHAP_Feature_Coverage"
    ] * 100
).round(1)

aggregate_results_df[
    "Mean_Section_Coverage_Percent"
] = (
    aggregate_results_df[
        "Mean_Section_Coverage"
    ] * 100
).round(1)


# ============================================================
# SAVE AGGREGATE TABLE
# ============================================================

aggregate_results_file = (
    TABLES_DIR
    / "experiment_05_aggregate_ablation_results.csv"
)

aggregate_results_df.to_csv(
    aggregate_results_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nAggregate With-RAG vs No-RAG results:\n")

print(
    aggregate_results_df.to_string(
        index=False
    )
)

print(
    "\nAggregate results saved to:"
)

print(
    aggregate_results_file
)


print("\n" + "=" * 65)
print("STEP 5C COMPLETE")
print("=" * 65)

# ============================================================
# STEP 5D: PATIENT-LEVEL PAIRED DIFFERENCE ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("STEP 5D: PATIENT-LEVEL PAIRED DIFFERENCES")
print("=" * 65)


# ============================================================
# SPLIT RESULTS BY CONDITION
# ============================================================

with_rag_df = (
    ablation_results_df[
        ablation_results_df["Condition"] == "With-RAG"
    ]
    .set_index("Patient_ID")
)

no_rag_df = (
    ablation_results_df[
        ablation_results_df["Condition"] == "No-RAG"
    ]
    .set_index("Patient_ID")
)


# ============================================================
# CALCULATE WITH-RAG MINUS NO-RAG DIFFERENCES
# ============================================================

paired_difference_rows = []

for patient_id in PATIENT_IDS:

    with_row = with_rag_df.loc[patient_id]
    no_row = no_rag_df.loc[patient_id]

    paired_difference_rows.append(
        {
            "Patient_ID": patient_id,

            "Word_Count_Difference":
                with_row["Word_Count"]
                - no_row["Word_Count"],

            "SHAP_Coverage_Difference":
                with_row["SHAP_Feature_Coverage"]
                - no_row["SHAP_Feature_Coverage"],

            "Section_Coverage_Difference":
                with_row["Section_Coverage"]
                - no_row["Section_Coverage"],

            "Evidence_Mention_Difference":
                with_row["Evidence_ID_Mentions"]
                - no_row["Evidence_ID_Mentions"],

            "Faithfulness_Flag_Difference":
                with_row["Faithfulness_Flags"]
                - no_row["Faithfulness_Flags"],
        }
    )


paired_differences_df = pd.DataFrame(
    paired_difference_rows
)


# ============================================================
# ADD PERCENTAGE-POINT DIFFERENCES
# ============================================================

paired_differences_df[
    "SHAP_Coverage_Difference_Percentage_Points"
] = (
    paired_differences_df[
        "SHAP_Coverage_Difference"
    ] * 100
).round(1)

paired_differences_df[
    "Section_Coverage_Difference_Percentage_Points"
] = (
    paired_differences_df[
        "Section_Coverage_Difference"
    ] * 100
).round(1)


# ============================================================
# SAVE PAIRED DIFFERENCE TABLE
# ============================================================

paired_results_file = (
    TABLES_DIR
    / "experiment_05_paired_ablation_differences.csv"
)

paired_differences_df.to_csv(
    paired_results_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nPatient-level With-RAG minus No-RAG differences:\n")

print(
    paired_differences_df.to_string(
        index=False
    )
)

print(
    "\nPaired difference results saved to:"
)

print(
    paired_results_file
)


print("\n" + "=" * 65)
print("STEP 5D COMPLETE")
print("=" * 65)

# ============================================================
# STEP 6A: CREATE AGGREGATE ABLATION COMPARISON FIGURE
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


print("\n" + "=" * 65)
print("STEP 6A: CREATING AGGREGATE ABLATION FIGURE")
print("=" * 65)


# ------------------------------------------------------------
# Prepare percentage-based comparison
# ------------------------------------------------------------

comparison_metrics = [
    "SHAP Feature Coverage",
    "Section Coverage",
]

no_rag_values = [
    100.0,
    61.1,
]

with_rag_values = [
    100.0,
    66.7,
]


x = np.arange(len(comparison_metrics))
width = 0.35


fig, ax = plt.subplots(figsize=(9, 6))

bars_no_rag = ax.bar(
    x - width / 2,
    no_rag_values,
    width,
    label="No-RAG"
)

bars_with_rag = ax.bar(
    x + width / 2,
    with_rag_values,
    width,
    label="With-RAG"
)


ax.set_ylabel("Mean Coverage (%)")
ax.set_title(
    "Experiment 5: With-RAG vs No-RAG Explanation Coverage"
)

ax.set_xticks(x)
ax.set_xticklabels(comparison_metrics)

ax.set_ylim(0, 110)

ax.legend()


# Add values above bars
for bars in [bars_no_rag, bars_with_rag]:
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.1f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )


fig.tight_layout()


coverage_figure_file = (
    FIGURES_DIR
    / "experiment_05_rag_coverage_comparison.png"
)

plt.savefig(
    coverage_figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nCoverage comparison figure saved to:"
)

print(coverage_figure_file)


print("\n" + "=" * 65)
print("STEP 6A COMPLETE")
print("=" * 65)


# ============================================================
# STEP 6B: CREATE PATIENT-LEVEL FAITHFULNESS COMPARISON FIGURE
# ============================================================

print("\n" + "=" * 65)
print("STEP 6B: CREATING PATIENT-LEVEL FAITHFULNESS FIGURE")
print("=" * 65)


# ------------------------------------------------------------
# Faithfulness flags observed for each patient
# ------------------------------------------------------------

patient_labels = ["Patient 7", "Patient 11", "Patient 37"]

no_rag_flags = [1, 0, 2]
with_rag_flags = [0, 3, 0]


x = np.arange(len(patient_labels))
width = 0.35


fig, ax = plt.subplots(figsize=(9, 6))

bars_no_rag = ax.bar(
    x - width / 2,
    no_rag_flags,
    width,
    label="No-RAG"
)

bars_with_rag = ax.bar(
    x + width / 2,
    with_rag_flags,
    width,
    label="With-RAG"
)


ax.set_ylabel("Number of Faithfulness Flags")

ax.set_title(
    "Experiment 5: Patient-Level Faithfulness Comparison"
)

ax.set_xticks(x)
ax.set_xticklabels(patient_labels)

ax.set_ylim(0, 4)

ax.legend()


# ------------------------------------------------------------
# Add numerical labels
# ------------------------------------------------------------

for bars in [bars_no_rag, bars_with_rag]:
    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )


fig.tight_layout()


faithfulness_figure_file = (
    FIGURES_DIR
    / "experiment_05_patient_faithfulness_comparison.png"
)


plt.savefig(
    faithfulness_figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nPatient-level faithfulness comparison figure saved to:"
)

print(faithfulness_figure_file)


print("\n" + "=" * 65)
print("STEP 6B COMPLETE")
print("=" * 65)

# ============================================================
# STEP 7: CREATE FINAL EXPERIMENT 5 SUMMARY TABLE
# ============================================================

print("\n" + "=" * 65)
print("STEP 7: CREATING FINAL EXPERIMENT 5 SUMMARY")
print("=" * 65)


# ------------------------------------------------------------
# Create dissertation-ready summary
# ------------------------------------------------------------

final_summary_df = pd.DataFrame(
    {
        "Metric": [
            "Mean word count",
            "Mean SHAP feature coverage (%)",
            "Mean section coverage (%)",
            "Mean evidence ID mentions",
            "Total faithfulness flags",
        ],
        "No-RAG": [
            295.667,
            100.0,
            61.1,
            0.0,
            3,
        ],
        "With-RAG": [
            297.333,
            100.0,
            66.7,
            3.0,
            3,
        ],
    }
)


# ------------------------------------------------------------
# Calculate With-RAG minus No-RAG difference
# ------------------------------------------------------------

final_summary_df["Difference"] = (
    final_summary_df["With-RAG"]
    - final_summary_df["No-RAG"]
)


print("\nFinal Experiment 5 summary:")
print(
    final_summary_df.to_string(index=False)
)


# ------------------------------------------------------------
# Save table
# ------------------------------------------------------------

final_summary_file = (
    TABLES_DIR
    / "experiment_05_final_summary.csv"
)

final_summary_df.to_csv(
    final_summary_file,
    index=False
)


print(
    "\nFinal Experiment 5 summary saved to:"
)

print(final_summary_file)


print("\n" + "=" * 65)
print("EXPERIMENT 5 COMPLETE")
print("=" * 65)

