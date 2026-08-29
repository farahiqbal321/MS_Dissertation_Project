"""
Experiment 4: RAG-Enabled Agentic AI Explanation Framework.

This experiment uses the selected Clinical Random Forest model,
patient-level explainability outputs and retrieved clinical evidence
to generate structured, evidence-grounded explanations.

The AI agent does not make a diagnosis or replace the predictive model.
Its role is to interpret existing model outputs for decision support.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import requests
import ollama
import json
import logging
from pathlib import Path

import pandas as pd

from config import (
    EXPERIMENT_02_DIR,
    EXPERIMENT_03_DIR,
    EXPERIMENT_04_DIR,
)

from utils import (
    check_file_exists,
    save_dataframe,
    print_section,
    setup_logger,
)


# =========================================================
# EXPERIMENT 4 DIRECTORIES
# =========================================================

experiment_04_tables_dir = EXPERIMENT_04_DIR / "tables"
experiment_04_figures_dir = EXPERIMENT_04_DIR / "figures"
experiment_04_explanations_dir = EXPERIMENT_04_DIR / "explanations"
experiment_04_evidence_dir = EXPERIMENT_04_DIR / "retrieved_evidence"
experiment_04_knowledge_base_dir = EXPERIMENT_04_DIR / "knowledge_base"
experiment_04_logs_dir = EXPERIMENT_04_DIR / "logs"

for directory in [
    experiment_04_tables_dir,
    experiment_04_figures_dir,
    experiment_04_explanations_dir,
    experiment_04_evidence_dir,
    experiment_04_knowledge_base_dir,
    experiment_04_logs_dir,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================

logger = setup_logger(
    logger_name="experiment_04",
    log_file=experiment_04_logs_dir / "experiment_04.log",
)


print_section(
    "Experiment 4: RAG-Enabled Agentic AI Explanation Framework"
)

logger.info("Experiment 4 started.")


# =========================================================
# REQUIRED INPUT FILES
# =========================================================

selected_model_file = (
    EXPERIMENT_02_DIR
    / "selected_final_model.json"
)

selected_patients_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "selected_patients.csv"
)

prediction_summary_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "training_prediction_summary.csv"
)

global_shap_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "global_shap_feature_importance.csv"
)

local_shap_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "training_patient_shap_values.csv"
)

shap_lime_comparison_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "shap_lime_local_comparison.csv"
)

shap_lime_agreement_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "shap_lime_agreement_summary.csv"
)

combined_lime_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "combined_lime_explanations.csv"
)


required_files = [
    selected_model_file,
    selected_patients_file,
    prediction_summary_file,
    global_shap_file,
    local_shap_file,
    shap_lime_comparison_file,
    shap_lime_agreement_file,
    combined_lime_file,
]

for file_path in required_files:
    check_file_exists(file_path)


print("\nAll required Experiment 2 and Experiment 3 files were found.")

logger.info(
    "Required Experiment 4 input files validated successfully."
)


# =========================================================
# LOAD INPUTS
# =========================================================

with open(
    selected_model_file,
    "r",
    encoding="utf-8",
) as file:
    selected_model_information = json.load(file)


selected_patients_df = pd.read_csv(
    selected_patients_file
)

prediction_summary_df = pd.read_csv(
    prediction_summary_file
)

global_shap_df = pd.read_csv(
    global_shap_file
)

local_shap_df = pd.read_csv(
    local_shap_file
)

shap_lime_comparison_df = pd.read_csv(
    shap_lime_comparison_file
)

shap_lime_agreement_df = pd.read_csv(
    shap_lime_agreement_file
)

combined_lime_df = pd.read_csv(
    combined_lime_file
)


print("\nSelected model:")
print(
    selected_model_information[
        "Selected_Model"
    ]
)

print("\nSelected patients:")
print(selected_patients_df)

print("\nSHAP-LIME agreement summary:")
print(shap_lime_agreement_df)


logger.info(
    "Experiment 2 and Experiment 3 outputs loaded successfully."
)


print_section(
    "Experiment 4 Input Validation Complete"
)

print(
    "\nNo RAG retrieval or LLM explanation "
    "generation has been performed yet."
)

# =========================================================
# LOAD CLINICAL KNOWLEDGE BASE
# =========================================================

print_section(
    "Loading Clinical Knowledge Base"
)

knowledge_base_sources_file = (
    experiment_04_knowledge_base_dir
    / "knowledge_base_sources.csv"
)

clinical_evidence_chunks_file = (
    experiment_04_knowledge_base_dir
    / "clinical_evidence_chunks.csv"
)

check_file_exists(
    knowledge_base_sources_file
)

check_file_exists(
    clinical_evidence_chunks_file
)

knowledge_base_sources_df = pd.read_csv(
    knowledge_base_sources_file
)

clinical_evidence_chunks_df = pd.read_csv(
    clinical_evidence_chunks_file
)

print("\nKnowledge base sources:")
print(knowledge_base_sources_df)

print("\nClinical evidence chunks:")
print(clinical_evidence_chunks_df)

print(
    "\nNumber of included sources:"
)

print(
    len(knowledge_base_sources_df)
)

print(
    "\nNumber of clinical evidence chunks:"
)

print(
    len(clinical_evidence_chunks_df)
)

logger.info(
    "Clinical knowledge base loaded successfully."
)

print_section(
    "Clinical Knowledge Base Validation Complete"
)

# =========================================================
# INITIAL EVIDENCE RETRIEVAL TEST
# =========================================================

print_section(
    "Initial Evidence Retrieval Test"
)

# Patient 11 was selected in Experiment 3 as the
# representative borderline prediction.
test_patient_id = 11

print(
    f"\nTesting clinical evidence retrieval "
    f"for Patient {test_patient_id}"
)


# ---------------------------------------------------------
# Define clinically relevant search terms
# ---------------------------------------------------------

retrieval_terms = [
    "neurological",
    "disability",
    "motor",
    "coordination",
    "sensory",
    "visual",
]


# ---------------------------------------------------------
# Search knowledge-base chunks
# ---------------------------------------------------------

def retrieve_evidence_by_keywords(
    evidence_df: pd.DataFrame,
    search_terms: list[str],
) -> pd.DataFrame:
    """
    Retrieve clinical evidence chunks using transparent
    keyword matching across topic and evidence text.
    """

    results = evidence_df.copy()

    results["Search_Text"] = (
        results["Topic"].fillna("").astype(str)
        + " "
        + results["Evidence_Text"].fillna("").astype(str)
    ).str.lower()

    results["Matched_Terms"] = results["Search_Text"].apply(
        lambda text: [
            term
            for term in search_terms
            if term.lower() in text
        ]
    )

    results["Retrieval_Score"] = results[
        "Matched_Terms"
    ].apply(len)

    results = results[
        results["Retrieval_Score"] > 0
    ].copy()

    results = results.sort_values(
        by="Retrieval_Score",
        ascending=False,
    )

    return results[
        [
            "Chunk_ID",
            "Source_ID",
            "Topic",
            "Evidence_Text",
            "Matched_Terms",
            "Retrieval_Score",
        ]
    ]


retrieved_evidence_df = retrieve_evidence_by_keywords(
    clinical_evidence_chunks_df,
    retrieval_terms,
)


# ---------------------------------------------------------
# Display retrieval results
# ---------------------------------------------------------

print("\nRetrieval terms:")
print(retrieval_terms)

print("\nRetrieved clinical evidence:")
print(
    retrieved_evidence_df[
        [
            "Chunk_ID",
            "Topic",
            "Matched_Terms",
            "Retrieval_Score",
        ]
    ]
)


# ---------------------------------------------------------
# Save retrieval results
# ---------------------------------------------------------

retrieved_evidence_output_file = (
    experiment_04_evidence_dir
    / "patient_11_initial_retrieved_evidence.csv"
)

retrieved_evidence_df.to_csv(
    retrieved_evidence_output_file,
    index=False,
)

logger.info(
    "Initial evidence retrieval test completed "
    "for Patient 11."
)

print(
    "\nRetrieved evidence saved to:"
)

print(
    retrieved_evidence_output_file
)

print_section(
    "Initial Evidence Retrieval Test Complete"
)

# =========================================================
# SHAP-DRIVEN CLINICAL EVIDENCE RETRIEVAL
# =========================================================

print_section(
    "SHAP-Driven Clinical Evidence Retrieval"
)

borderline_shap_file = (
    EXPERIMENT_03_DIR
    / "tables"
    / "borderline_top_shap_features.csv"
)

check_file_exists(
    borderline_shap_file
)

borderline_shap_df = pd.read_csv(
    borderline_shap_file
)

print("\nPatient 11 SHAP data loaded successfully.")

print(
    borderline_shap_df.head(10)
)

# ---------------------------------------------------------
# Select strongest patient-specific SHAP features
# ---------------------------------------------------------

top_n_features = 6

patient_11_top_shap_df = (
    borderline_shap_df
    .sort_values(
        by="Absolute_SHAP",
        ascending=False,
    )
    .head(top_n_features)
    .copy()
)

print(
    "\nTop SHAP features selected "
    "for Patient 11:"
)

print(
    patient_11_top_shap_df[
        [
            "Feature",
            "SHAP_Value",
            "Absolute_SHAP",
        ]
    ]
)

# ---------------------------------------------------------
# Convert SHAP feature names into retrieval concepts
# ---------------------------------------------------------

feature_to_retrieval_terms = {
    "Total_Abnormal_Neuro_Findings": [
        "neurological",
        "disability",
    ],
    "Age_of_onset": [
        "age",
        "onset",
    ],
    "Age": [
        "age",
    ],
    "Coordination": [
        "coordination",
        "balance",
    ],
    "Motor_System": [
        "motor",
        "mobility",
    ],
    "Cerebella": [
        "coordination",
        "balance",
    ],
    "Sensory": [
        "sensory",
    ],
    "Visual": [
        "visual",
    ],
    "Gait": [
        "mobility",
        "gait",
    ],
    "Disease_Duration": [
        "disease",
        "duration",
    ],
}

patient_11_retrieval_terms = []

for feature in patient_11_top_shap_df["Feature"]:

    mapped_terms = feature_to_retrieval_terms.get(
        feature,
        [
            feature
            .replace("_", " ")
            .lower()
        ],
    )

    patient_11_retrieval_terms.extend(
        mapped_terms
    )


# Remove duplicate terms while preserving order
patient_11_retrieval_terms = list(
    dict.fromkeys(
        patient_11_retrieval_terms
    )
)


print(
    "\nAutomatically generated retrieval terms:"
)

print(
    patient_11_retrieval_terms
)

# ---------------------------------------------------------
# Retrieve evidence using SHAP-derived concepts
# ---------------------------------------------------------

patient_11_rag_retrieval_df = (
    retrieve_evidence_by_keywords(
        clinical_evidence_chunks_df,
        patient_11_retrieval_terms,
    )
)


print(
    "\nPatient 11 SHAP-driven "
    "retrieval results:"
)

print(
    patient_11_rag_retrieval_df[
        [
            "Chunk_ID",
            "Topic",
            "Matched_Terms",
            "Retrieval_Score",
        ]
    ]
)

# ---------------------------------------------------------
# Save SHAP-driven retrieval output
# ---------------------------------------------------------

patient_11_rag_output_file = (
    experiment_04_evidence_dir
    / "patient_11_shap_driven_retrieval.csv"
)

patient_11_rag_retrieval_df.to_csv(
    patient_11_rag_output_file,
    index=False,
)

logger.info(
    "SHAP-driven evidence retrieval completed "
    "for Patient 11."
)

print(
    "\nSHAP-driven retrieval saved to:"
)

print(
    patient_11_rag_output_file
)

print_section(
    "SHAP-Driven Retrieval Complete"
)

# ============================================================
# SEMANTIC RETRIEVAL USING SENTENCE TRANSFORMERS
# ============================================================

print_section(
    "Semantic Evidence Retrieval"
)

# Load a lightweight pretrained sentence-transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print(
    "\nSentence-transformer model loaded successfully."
)

# Build a patient-specific semantic query from the
# clinically meaningful concepts derived from SHAP
patient_11_semantic_query = (
    "multiple sclerosis neurological disability "
    "age age of onset coordination balance "
    "motor function mobility"
)

print(
    "\nPatient 11 semantic retrieval query:"
)

print(
    patient_11_semantic_query
)

# ============================================================
# EMBED CLINICAL EVIDENCE AND CALCULATE SEMANTIC SIMILARITY
# ============================================================

print_section(
    "Patient 11 Semantic Similarity Retrieval"
)

# Convert the clinical evidence passages to text
evidence_texts = (
    clinical_evidence_chunks_df["Evidence_Text"]
    .fillna("")
    .astype(str)
    .tolist()
)

# Generate dense embeddings for all clinical evidence chunks
evidence_embeddings = embedding_model.encode(
    evidence_texts
)

# Generate an embedding for the patient-specific query
query_embedding = embedding_model.encode(
    [patient_11_semantic_query]
)

# Calculate cosine similarity between the patient query
# and every clinical evidence chunk
semantic_scores = cosine_similarity(
    query_embedding,
    evidence_embeddings
)[0]

# Create a copy so the original knowledge base is unchanged
patient_11_semantic_results = (
    clinical_evidence_chunks_df.copy()
)

# Add semantic similarity scores
patient_11_semantic_results[
    "Semantic_Similarity"
] = semantic_scores

# Rank evidence from most to least semantically relevant
patient_11_semantic_results = (
    patient_11_semantic_results
    .sort_values(
        "Semantic_Similarity",
        ascending=False
    )
    .reset_index(drop=True)
)

patient_11_semantic_results[
    "Semantic_Rank"
] = range(
    1,
    len(patient_11_semantic_results) + 1
)

print(
    "\nPatient 11 semantic retrieval results:"
)

print(
    patient_11_semantic_results[
        [
            "Chunk_ID",
            "Evidence_Text",
            "Semantic_Similarity",
            "Semantic_Rank"
        ]
    ]
)

# ============================================================
# SAVE SEMANTIC RETRIEVAL RESULTS
# ============================================================

patient_11_semantic_output_file = (
    experiment_04_evidence_dir
    / "patient_11_semantic_retrieval.csv"
)

patient_11_semantic_results.to_csv(
    patient_11_semantic_output_file,
    index=False
)

logger.info(
    "Patient 11 semantic retrieval results saved successfully."
)

print(
    "\nSemantic retrieval results saved to:"
)

print(
    patient_11_semantic_output_file
)

print_section(
    "PATIENT 11 SEMANTIC RETRIEVAL COMPLETE"
)

# ============================================================
# BUILD TOP-K CLINICAL EVIDENCE BUNDLE
# ============================================================

TOP_K_EVIDENCE = 3

patient_11_top_evidence = (
    patient_11_semantic_results
    .sort_values("Semantic_Rank")
    .head(TOP_K_EVIDENCE)
    .copy()
)

print_section(
    "PATIENT 11 TOP-K CLINICAL EVIDENCE"
)

print(
    patient_11_top_evidence[
        [
            "Chunk_ID",
            "Source_ID",
            "Topic",
            "Evidence_Text",
            "Semantic_Similarity",
            "Semantic_Rank",
        ]
    ]
)

# Save selected evidence used by the Agentic AI stage
patient_11_top_evidence_file = (
    experiment_04_evidence_dir
    / "patient_11_top3_clinical_evidence.csv"
)

patient_11_top_evidence.to_csv(
    patient_11_top_evidence_file,
    index=False
)

logger.info(
    "Top-3 clinical evidence bundle created for Patient 11."
)

print(
    "\nTop-3 clinical evidence saved to:"
)

print(
    patient_11_top_evidence_file
)

print_section(
    "PATIENT 11 TOP-K EVIDENCE BUNDLE COMPLETE"
)

# ============================================================
# BUILD STRUCTURED PATIENT CONTEXT FOR AGENTIC EXPLANATION
# ============================================================

print_section(
    "BUILDING PATIENT 11 AGENTIC EXPLANATION CONTEXT"
)

# ------------------------------------------------------------
# 1. Retrieve Patient 11 prediction information
# ------------------------------------------------------------

patient_11_prediction = prediction_summary_df[
    prediction_summary_df["Patient_ID"] == 11
].iloc[0]

# ------------------------------------------------------------
# 2. Retrieve Patient 11 local SHAP explanation
# ------------------------------------------------------------

patient_11_shap_context = borderline_shap_df.copy()

# Rank by absolute SHAP contribution and retain the five
# strongest patient-specific prediction drivers
patient_11_shap_context = (
    patient_11_shap_context
    .sort_values("Absolute_SHAP", ascending=False)
    .head(5)
)

# ------------------------------------------------------------
# 3. Convert SHAP results into structured text
# ------------------------------------------------------------

shap_context_lines = []

for _, row in patient_11_shap_context.iterrows():

    direction = (
        "towards moderate/high disability"
        if row["SHAP_Value"] > 0
        else "towards low disability"
    )

    shap_context_lines.append(
        f"{row['Feature']}: "
        f"SHAP contribution {row['SHAP_Value']:.4f}, "
        f"{direction}"
    )

shap_context_text = "\n".join(shap_context_lines)

# ------------------------------------------------------------
# 4. Convert retrieved NICE evidence into structured text
# ------------------------------------------------------------

evidence_context_lines = []

for _, row in patient_11_top_evidence.iterrows():

    evidence_context_lines.append(
        f"[{row['Chunk_ID']}] "
        f"{row['Topic']}: "
        f"{row['Evidence_Text']}"
    )

evidence_context_text = "\n".join(evidence_context_lines)

# ------------------------------------------------------------
# 5. Display structured context
# ------------------------------------------------------------

print("\nPatient 11 prediction information:")
print(patient_11_prediction)

print("\nPatient-specific SHAP context:")
print(shap_context_text)

print("\nRetrieved clinical evidence context:")
print(evidence_context_text)

print_section(
    "PATIENT 11 AGENTIC EXPLANATION CONTEXT COMPLETE"
)

# ============================================================
# BUILD GROUNDED AGENTIC AI PROMPT
# ============================================================

print_section(
    "BUILDING PATIENT 11 GROUNDED AGENTIC AI PROMPT"
)

patient_11_agentic_prompt = f"""
You are an AI explanation agent supporting interpretation of a
machine-learning prediction for Multiple Sclerosis disability.

IMPORTANT RULES:
1. Do not diagnose the patient.
2. Do not change or override the machine-learning prediction.
3. Use only the model information, SHAP contributions and retrieved
   clinical evidence provided below.
4. Clearly distinguish model-derived information from clinical evidence.
5. Do not invent clinical facts or evidence.
6. Acknowledge uncertainty when the prediction probability is close
   to the classification threshold. For probabilities close to 0.50,
   explicitly describe the prediction as borderline and avoid language
   implying strong confidence in either class.
7. Explain conflicting feature contributions where relevant.
8. Keep the explanation concise, clinically understandable and suitable
   for decision support.
9. Cite retrieved evidence using the supplied evidence identifiers
   such as [NG220_003].
10. State that the output is intended to support, not replace,
    professional clinical judgement.
11. Do not interpret the predicted disability class as a diagnosis,
    disease progression assessment or direct measure of neurological impairment.
12. Do not recommend or make claims about diagnosis, treatment or management
    unless explicitly supported by the retrieved evidence.
13. When describing SHAP contributions, use the supplied direction exactly.
    A negative SHAP value should be described as contributing towards low
    disability, and a positive SHAP value as contributing towards
    moderate/high disability. Do not describe SHAP values as clinical
    associations or causal relationships.
14. Do not state that the model prediction creates a need for further
    clinical evaluation, confirmation or refutation. Instead, state that
    the prediction should be interpreted within the patient's existing
    clinical assessment and alongside the retrieved clinical evidence.


MODEL PREDICTION
----------------
Patient ID: {int(patient_11_prediction["Patient_ID"])}
Actual class: {int(patient_11_prediction["Actual_Class"])}
Predicted class: {int(patient_11_prediction["Predicted_Class"])}
Predicted probability of moderate/high disability:
{patient_11_prediction["Predicted_Probability"]:.4f}

PATIENT-SPECIFIC MODEL EXPLANATION
----------------------------------
{shap_context_text}

RETRIEVED CLINICAL EVIDENCE
---------------------------
{evidence_context_text}

TASK
----
Generate a structured explanation containing:

1. Prediction Summary
2. Main Factors Supporting Moderate/High Disability
3. Main Factors Supporting Low Disability
4. Clinical Context from Retrieved Evidence
5. Uncertainty and Limitations
6. Decision-Support Summary

The explanation must remain faithful to the supplied model output and
retrieved evidence.
"""

print("\nGenerated Patient 11 Agentic AI prompt:")
print(patient_11_agentic_prompt)

print_section(
    "PATIENT 11 GROUNDED AGENTIC AI PROMPT COMPLETE"
)

# ============================================================
# LEGACY PATIENT 11 GENERATION PIPELINE
# ============================================================

# This switch prevents the original one-off Patient 11
# generation pipeline from running while the reusable
# Experiment 4 pipeline is developed and tested.
RUN_LEGACY_PATIENT_11_PIPELINE = False


if RUN_LEGACY_PATIENT_11_PIPELINE:

    # ========================================================
    # GENERATE PATIENT 11 GROUNDED AGENTIC AI EXPLANATION
    # ========================================================

    print_section(
        "GENERATING PATIENT 11 AGENTIC AI EXPLANATION"
    )

    agentic_response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": patient_11_agentic_prompt,
            }
        ],
        options={
            "temperature": 0.2
        },
    )

    patient_11_agentic_explanation = (
        agentic_response["message"]["content"]
    )

    print(
        "\nGenerated Patient 11 Agentic AI explanation:"
    )

    print(
        patient_11_agentic_explanation
    )

    print_section(
        "PATIENT 11 AGENTIC AI EXPLANATION COMPLETE"
    )


    # ========================================================
    # SAVE PATIENT 11 AGENTIC AI EXPLANATION
    # ========================================================

    patient_11_explanation_file = (
        experiment_04_explanations_dir
        / "patient_11_agentic_explanation.txt"
    )

    with open(
        patient_11_explanation_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            patient_11_agentic_explanation
        )


    # ========================================================
    # SAVE PATIENT 11 AGENTIC AI PROMPT
    # ========================================================

    patient_11_prompt_file = (
        experiment_04_explanations_dir
        / "patient_11_agentic_prompt.txt"
    )

    with open(
        patient_11_prompt_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            patient_11_agentic_prompt
        )


    # ========================================================
    # SAVE PATIENT 11 GENERATION METADATA
    # ========================================================

    generation_metadata = {
        "Patient_ID": 11,
        "Scenario": "Borderline",
        "LLM": "llama3.1:8b",
        "LLM_Runtime": "Ollama",
        "Temperature": 0.2,
        "Embedding_Model": "all-MiniLM-L6-v2",
        "Top_K_Evidence": 3,
        "Prediction_Probability": float(
            patient_11_prediction["Predicted_Probability"]
        ),
        "Predicted_Class": int(
            patient_11_prediction["Predicted_Class"]
        ),
        "Actual_Class": int(
            patient_11_prediction["Actual_Class"]
        ),
    }

    metadata_file = (
        experiment_04_explanations_dir
        / "patient_11_generation_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            generation_metadata,
            file,
            indent=4
        )


    logger.info(
        "Patient 11 Agentic AI prompt, explanation "
        "and generation metadata saved successfully."
    )

    print_section(
        "PATIENT 11 AGENTIC AI OUTPUTS SAVED"
    )


# ============================================================
# EXPERIMENT 4: REPRODUCIBLE GENERATION SETTINGS
# ============================================================

OLLAMA_MODEL = "llama3.1:8b"

GENERATION_TEMPERATURE = 0.2

GENERATION_SEED = 42

TOP_K_EVIDENCE = 3

print_section(
    "EXPERIMENT 4 REPRODUCIBLE GENERATION SETTINGS"
)

print("Ollama model:", OLLAMA_MODEL)
print("Temperature:", GENERATION_TEMPERATURE)
print("Seed:", GENERATION_SEED)
print("Top-K evidence:", TOP_K_EVIDENCE)

logger.info(
    "Experiment 4 reproducible generation settings configured: "
    f"model={OLLAMA_MODEL}, "
    f"temperature={GENERATION_TEMPERATURE}, "
    f"seed={GENERATION_SEED}, "
    f"top_k={TOP_K_EVIDENCE}."
)

# ============================================================
# REUSABLE OLLAMA EXPLANATION GENERATION FUNCTION
# ============================================================

def generate_agentic_explanation(prompt):
    """
    Generate a reproducible grounded explanation using the
    local Llama 3.1 8B model through Ollama.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": GENERATION_TEMPERATURE,
            "seed": GENERATION_SEED
        }
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    response_data = response.json()

    explanation = response_data["response"].strip()

    return explanation


# ============================================================
# REUSABLE PATIENT SHAP CONTEXT FUNCTION
# ============================================================

def build_patient_shap_context(patient_id, shap_df, top_n=5):
    """
    Extract the strongest patient-specific SHAP contributions
    from the wide-format training SHAP table.
    """

    patient_row = shap_df[
        shap_df["Patient_ID"] == patient_id
    ].copy()

    if patient_row.empty:
        raise ValueError(
            f"No SHAP values found for Patient {patient_id}."
        )

    # Columns that are identifiers rather than SHAP features
    excluded_columns = [
        "Patient_ID",
        "Actual_Class"
    ]

    feature_columns = [
        column
        for column in shap_df.columns
        if column not in excluded_columns
    ]

    shap_records = []

    for feature in feature_columns:

        shap_value = float(
            patient_row.iloc[0][feature]
        )

        shap_records.append(
            {
                "Feature": feature,
                "SHAP_Value": shap_value,
                "Absolute_SHAP": abs(shap_value)
            }
        )

    patient_shap = pd.DataFrame(
        shap_records
    )

    patient_shap = patient_shap.sort_values(
        by="Absolute_SHAP",
        ascending=False
    ).head(top_n)

    shap_context_lines = []

    for _, row in patient_shap.iterrows():

        feature = row["Feature"]
        shap_value = float(row["SHAP_Value"])

        if shap_value > 0:
            direction = "towards moderate/high disability"

        elif shap_value < 0:
            direction = "towards low disability"

        else:
            direction = "with no directional contribution"

        shap_context_lines.append(
            f"{feature}: SHAP contribution "
            f"{shap_value:.4f}, {direction}"
        )

    shap_context_text = "\n".join(
        shap_context_lines
    )

    return patient_shap, shap_context_text

# ============================================================
# TEST REUSABLE SHAP CONTEXT FUNCTION
# ============================================================

test_patient_11_shap_df, test_patient_11_shap_text = (
    build_patient_shap_context(
        patient_id=11,
        shap_df=local_shap_df,
        top_n=5
    )
)

print_section(
    "REUSABLE SHAP FUNCTION TEST - PATIENT 11"
)

print(
    test_patient_11_shap_df[
        [
            "Feature",
            "SHAP_Value",
            "Absolute_SHAP"
        ]
    ]
)

print(
    "\nGenerated SHAP context:"
)

print(
    test_patient_11_shap_text
)

# ============================================================
# REUSABLE SEMANTIC EVIDENCE RETRIEVAL FUNCTION
# ============================================================

def retrieve_semantic_evidence(
    patient_shap_df,
    evidence_df,
    embedding_model,
    top_k=3
):
    """
    Build a semantic query from a patient's strongest SHAP
    features and retrieve the most relevant clinical evidence.
    """

    feature_to_retrieval_terms = {
        "Total_Abnormal_Neuro_Findings": [
            "neurological",
            "disability",
        ],
        "Age_of_onset": [
            "age",
            "onset",
        ],
        "Age": [
            "age",
        ],
        "Coordination": [
            "coordination",
            "balance",
        ],
        "Motor_System": [
            "motor",
            "mobility",
        ],
        "Cerebella": [
            "coordination",
            "balance",
        ],
        "Sensory": [
            "sensory",
        ],
        "Visual": [
            "visual",
        ],
        "Gait": [
            "gait",
            "mobility",
        ],
        "Disease_Duration": [
            "disease duration",
        ],
        "Pyramidal": [
            "motor",
            "neurological",
        ],
    }

    retrieval_terms = []

    for feature in patient_shap_df["Feature"]:

        mapped_terms = feature_to_retrieval_terms.get(
            feature,
            [
                feature
                .replace("_", " ")
                .lower()
            ],
        )

        retrieval_terms.extend(
            mapped_terms
        )

    # Remove duplicate terms while preserving order
    retrieval_terms = list(
        dict.fromkeys(retrieval_terms)
    )

    semantic_query = (
        "multiple sclerosis disability "
        + " ".join(retrieval_terms)
    )

    evidence_texts = (
        evidence_df["Evidence_Text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    evidence_embeddings = embedding_model.encode(
        evidence_texts
    )

    query_embedding = embedding_model.encode(
        [semantic_query]
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        evidence_embeddings
    )[0]

    semantic_results = evidence_df.copy()

    semantic_results[
        "Semantic_Similarity"
    ] = similarity_scores

    semantic_results = (
        semantic_results
        .sort_values(
            by="Semantic_Similarity",
            ascending=False
        )
        .reset_index(drop=True)
    )

    semantic_results[
        "Semantic_Rank"
    ] = range(
        1,
        len(semantic_results) + 1
    )

    top_evidence = (
        semantic_results
        .head(top_k)
        .copy()
    )

    return (
        retrieval_terms,
        semantic_query,
        semantic_results,
        top_evidence
    )

# ============================================================
# TEST REUSABLE SEMANTIC RETRIEVAL - PATIENT 11
# ============================================================

(
    test_patient_11_terms,
    test_patient_11_query,
    test_patient_11_semantic_results,
    test_patient_11_top_evidence,
) = retrieve_semantic_evidence(
    patient_shap_df=test_patient_11_shap_df,
    evidence_df=clinical_evidence_chunks_df,
    embedding_model=embedding_model,
    top_k=TOP_K_EVIDENCE
)

print_section(
    "REUSABLE SEMANTIC RETRIEVAL TEST - PATIENT 11"
)

print(
    "\nRetrieval terms:"
)

print(
    test_patient_11_terms
)

print(
    "\nSemantic query:"
)

print(
    test_patient_11_query
)

print(
    "\nTop clinical evidence:"
)

print(
    test_patient_11_top_evidence[
        [
            "Chunk_ID",
            "Topic",
            "Semantic_Similarity",
            "Semantic_Rank",
        ]
    ]
)

# ============================================================
# REUSABLE GROUNDED PROMPT FUNCTION
# ============================================================

def build_grounded_agentic_prompt(
    patient_id,
    prediction_row,
    shap_context_text,
    top_evidence_df
):
    """
    Build a consistent grounded Agentic AI prompt for a
    selected patient using prediction information, SHAP
    contributions and retrieved clinical evidence.
    """

    evidence_context_lines = []

    for _, row in top_evidence_df.iterrows():

        evidence_context_lines.append(
            f"[{row['Chunk_ID']}] "
            f"{row['Topic']}: "
            f"{row['Evidence_Text']}"
        )

    evidence_context_text = "\n".join(
        evidence_context_lines
    )

    prompt = f"""
You are an AI explanation agent supporting interpretation of a
machine-learning prediction for Multiple Sclerosis disability.

IMPORTANT RULES:
1. Do not diagnose the patient.
2. Do not change or override the machine-learning prediction.
3. Use only the model information, SHAP contributions and retrieved
   clinical evidence provided below.
4. Clearly distinguish model-derived information from clinical evidence.
5. Do not invent clinical facts or evidence.
6. Acknowledge uncertainty when the prediction probability is close
   to the classification threshold. For probabilities close to 0.50,
   explicitly describe the prediction as borderline and avoid language
   implying strong confidence in either class.
7. Explain conflicting feature contributions where relevant.
8. Keep the explanation concise, clinically understandable and suitable
   for decision support.
9. Cite retrieved evidence using the supplied evidence identifiers.
10. State that the output is intended to support, not replace,
    professional clinical judgement.
11. Do not interpret the predicted disability class as a diagnosis,
    disease progression assessment or direct measure of neurological
    impairment.
12. Do not recommend or make claims about diagnosis, treatment or
    management unless explicitly supported by retrieved evidence.
13. When describing SHAP contributions, use the supplied direction
    exactly. A negative SHAP value should be described as contributing
    towards low disability, and a positive SHAP value as contributing
    towards moderate/high disability. Do not describe SHAP values as
    clinical associations or causal relationships.
14. Do not state that the model prediction creates a need for further
    clinical evaluation, confirmation or refutation. Instead, state
    that the prediction should be interpreted within the patient's
    existing clinical assessment and alongside the retrieved clinical
    evidence.

MODEL PREDICTION
----------------
Patient ID: {int(patient_id)}
Actual class: {int(prediction_row["Actual_Class"])}
Predicted class: {int(prediction_row["Predicted_Class"])}
Predicted probability of moderate/high disability:
{float(prediction_row["Predicted_Probability"]):.4f}

PATIENT-SPECIFIC MODEL EXPLANATION
----------------------------------
{shap_context_text}

RETRIEVED CLINICAL EVIDENCE
---------------------------
{evidence_context_text}

TASK
----
Generate a structured explanation containing:

1. Prediction Summary
2. Main Factors Supporting Moderate/High Disability
3. Main Factors Supporting Low Disability
4. Clinical Context from Retrieved Evidence
5. Uncertainty and Limitations
6. Decision-Support Summary

The explanation must remain faithful to the supplied model output and
retrieved evidence.
"""

    return prompt

# ============================================================
# TEST REUSABLE GROUNDED PROMPT - PATIENT 11
# ============================================================

test_patient_11_prediction = prediction_summary_df[
    prediction_summary_df["Patient_ID"] == 11
].iloc[0]

test_patient_11_prompt = build_grounded_agentic_prompt(
    patient_id=11,
    prediction_row=test_patient_11_prediction,
    shap_context_text=test_patient_11_shap_text,
    top_evidence_df=test_patient_11_top_evidence
)

print_section(
    "REUSABLE GROUNDED PROMPT TEST - PATIENT 11"
)

print(
    test_patient_11_prompt
)

# ============================================================
# REUSABLE END-TO-END AGENTIC PATIENT CASE FUNCTION
# ============================================================

def run_agentic_patient_case(
    patient_id,
    prediction_df,
    shap_df,
    evidence_df,
    embedding_model,
    output_dir,
    top_n_shap=5,
    top_k_evidence=3
):
    """
    Run the complete grounded Agentic AI explanation pipeline
    for one selected patient.

    Pipeline:
    prediction -> SHAP -> semantic retrieval -> grounded prompt
    -> LLM explanation -> saved outputs
    """

    print_section(
        f"RUNNING AGENTIC AI CASE - PATIENT {patient_id}"
    )

    # --------------------------------------------------------
    # 1. Retrieve prediction information
    # --------------------------------------------------------

    patient_prediction_matches = prediction_df[
        prediction_df["Patient_ID"] == patient_id
    ]

    if patient_prediction_matches.empty:
        raise ValueError(
            f"Patient {patient_id} not found in prediction dataframe."
        )

    patient_prediction = patient_prediction_matches.iloc[0]

    print("\nPrediction information:")
    print(patient_prediction)


    # --------------------------------------------------------
    # 2. Build patient-specific SHAP context
    # --------------------------------------------------------

    patient_shap, shap_context_text = build_patient_shap_context(
        patient_id=patient_id,
        shap_df=shap_df,
        top_n=top_n_shap
    )

    print("\nTop SHAP features:")
    print(patient_shap)

    print("\nGenerated SHAP context:")
    print(shap_context_text)


    # --------------------------------------------------------
    # 3. Retrieve patient-specific clinical evidence
    # --------------------------------------------------------

    (
        retrieval_terms,
        semantic_query,
        semantic_results,
        top_evidence,
    ) = retrieve_semantic_evidence(
        patient_shap_df=patient_shap,
        evidence_df=evidence_df,
        embedding_model=embedding_model,
        top_k=top_k_evidence
    )

    print("\nSemantic query:")
    print(semantic_query)

    print("\nTop retrieved clinical evidence:")

    print(
        top_evidence[
            [
                "Chunk_ID",
                "Topic",
                "Semantic_Similarity",
                "Semantic_Rank",
            ]
        ]
    )


    # --------------------------------------------------------
    # 4. Build grounded Agentic AI prompt
    # --------------------------------------------------------

    grounded_prompt = build_grounded_agentic_prompt(
        patient_id=patient_id,
        prediction_row=patient_prediction,
        shap_context_text=shap_context_text,
        top_evidence_df=top_evidence
    )


    # --------------------------------------------------------
    # 5. Generate explanation using local Llama model
    # --------------------------------------------------------

    print_section(
        f"GENERATING AGENTIC AI EXPLANATION - PATIENT {patient_id}"
    )

    explanation = generate_agentic_explanation(
        grounded_prompt
    )

    print(
        f"\nGenerated Patient {patient_id} Agentic AI explanation:"
    )

    print(explanation)


    # --------------------------------------------------------
    # 6. Create patient output directory
    # --------------------------------------------------------

    patient_output_dir = (
        output_dir / f"patient_{patient_id}"
    )

    patient_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 7. Save SHAP results
    # --------------------------------------------------------

    shap_output_file = (
        patient_output_dir
        / f"patient_{patient_id}_top_shap_features.csv"
    )

    patient_shap.to_csv(
        shap_output_file,
        index=False
    )


    # --------------------------------------------------------
    # 8. Save semantic retrieval results
    # --------------------------------------------------------

    semantic_output_file = (
        patient_output_dir
        / f"patient_{patient_id}_semantic_retrieval.csv"
    )

    semantic_results.to_csv(
        semantic_output_file,
        index=False
    )

    top_evidence_file = (
        patient_output_dir
        / f"patient_{patient_id}_top_clinical_evidence.csv"
    )

    top_evidence.to_csv(
        top_evidence_file,
        index=False
    )


    # --------------------------------------------------------
    # 9. Save grounded prompt
    # --------------------------------------------------------

    prompt_output_file = (
        patient_output_dir
        / f"patient_{patient_id}_agentic_prompt.txt"
    )

    with open(
        prompt_output_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(grounded_prompt)


    # --------------------------------------------------------
    # 10. Save generated explanation
    # --------------------------------------------------------

    explanation_output_file = (
        patient_output_dir
        / f"patient_{patient_id}_agentic_explanation.txt"
    )

    with open(
        explanation_output_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(explanation)


    # --------------------------------------------------------
    # 11. Save reproducibility metadata
    # --------------------------------------------------------

    metadata = {
        "Patient_ID": int(patient_id),
        "LLM": OLLAMA_MODEL,
        "LLM_Runtime": "Ollama",
        "Temperature": GENERATION_TEMPERATURE,
        "Seed": GENERATION_SEED,
        "Embedding_Model": "all-MiniLM-L6-v2",
        "Top_N_SHAP": int(top_n_shap),
        "Top_K_Evidence": int(top_k_evidence),
        "Prediction_Probability": float(
            patient_prediction["Predicted_Probability"]
        ),
        "Predicted_Class": int(
            patient_prediction["Predicted_Class"]
        ),
        "Actual_Class": int(
            patient_prediction["Actual_Class"]
        ),
        "Retrieval_Terms": retrieval_terms,
        "Semantic_Query": semantic_query,
    }

    metadata_output_file = (
        patient_output_dir
        / f"patient_{patient_id}_generation_metadata.json"
    )

    with open(
        metadata_output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )


    logger.info(
        f"Agentic AI case completed successfully for "
        f"Patient {patient_id}."
    )

    print_section(
        f"AGENTIC AI CASE COMPLETE - PATIENT {patient_id}"
    )


    # --------------------------------------------------------
    # 12. Return results for later comparison
    # --------------------------------------------------------

    return {
        "patient_id": patient_id,
        "prediction": patient_prediction,
        "shap": patient_shap,
        "retrieval_terms": retrieval_terms,
        "semantic_query": semantic_query,
        "semantic_results": semantic_results,
        "top_evidence": top_evidence,
        "prompt": grounded_prompt,
        "explanation": explanation,
        "metadata": metadata,
    }

# ============================================================
# VALIDATE END-TO-END PIPELINE USING PATIENT 11
# ============================================================

patient_11_reusable_result = run_agentic_patient_case(
    patient_id=11,
    prediction_df=prediction_summary_df,
    shap_df=local_shap_df,
    evidence_df=clinical_evidence_chunks_df,
    embedding_model=embedding_model,
    output_dir=EXPERIMENT_04_DIR,
    top_n_shap=5,
    top_k_evidence=TOP_K_EVIDENCE
)

# ============================================================
# EXPERIMENT 4: AGENTIC EXPLANATION FAITHFULNESS VALIDATION
# ============================================================

def validate_agentic_explanation(
    explanation,
    shap_df,
    predicted_probability
):
    """
    Validate an Agentic AI explanation for potentially unsupported
    interpretations of patient-specific SHAP contributions and
    probability-related uncertainty statements.

    The validator does not alter the generated explanation.
    It records potential faithfulness issues for experimental analysis.
    """

    validation_flags = []

    explanation_lower = explanation.lower()

    # --------------------------------------------------------
    # 1. Unsupported SHAP interpretation checks
    # --------------------------------------------------------

    unsupported_directional_terms = [
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

    for term in unsupported_directional_terms:

        if term in explanation_lower:

            validation_flags.append(
                {
                    "Flag_Type": "Unsupported_SHAP_Interpretation",
                    "Term": term,
                    "Reason": (
                        "SHAP direction indicates contribution towards a "
                        "prediction class but does not independently justify "
                        "this interpretation of the underlying feature value."
                    ),
                }
            )


    # --------------------------------------------------------
    # 2. Probability and uncertainty consistency checks
    # --------------------------------------------------------

    predicted_probability = float(
        predicted_probability
    )

    distance_from_threshold = abs(
        predicted_probability - 0.50
    )

    # Define a borderline region around the 0.50 threshold.
    # A probability within +/- 0.05 is treated as borderline
    # for this experimental validation.
    BORDERLINE_MARGIN = 0.05

    is_borderline = (
        distance_from_threshold <= BORDERLINE_MARGIN
    )

    borderline_terms = [
        "borderline",
        "close to the classification threshold",
        "close to 0.50",
        "close to 0.5",
        "near the classification threshold",
    ]

    explanation_calls_borderline = any(
        term in explanation_lower
        for term in borderline_terms
    )

    # If the explanation describes the case as borderline,
    # the probability should genuinely be close to 0.50.
    if explanation_calls_borderline and not is_borderline:

        validation_flags.append(
            {
                "Flag_Type": "Unsupported_Uncertainty_Interpretation",
                "Term": "borderline/close to threshold",
                "Reason": (
                    f"The predicted probability is "
                    f"{predicted_probability:.4f}, which is "
                    f"{distance_from_threshold:.4f} away from the "
                    "0.50 classification threshold and therefore falls "
                    "outside the predefined borderline margin of +/- 0.05."
                ),
            }
        )

    # If the probability is genuinely borderline, the explanation
    # should acknowledge uncertainty.
    if is_borderline and not explanation_calls_borderline:

        validation_flags.append(
            {
                "Flag_Type": "Missing_Uncertainty_Acknowledgement",
                "Term": "borderline uncertainty",
                "Reason": (
                    f"The predicted probability is "
                    f"{predicted_probability:.4f}, which falls within "
                    "the predefined +/- 0.05 borderline region around "
                    "the 0.50 classification threshold, but the "
                    "explanation does not explicitly acknowledge this."
                ),
            }
        )


    # --------------------------------------------------------
    # 3. SHAP direction audit
    # --------------------------------------------------------

    shap_direction_records = []

    for _, row in shap_df.iterrows():

        feature = row["Feature"]
        shap_value = float(
            row["SHAP_Value"]
        )

        if shap_value > 0:
            expected_direction = (
                "towards moderate/high disability"
            )

        elif shap_value < 0:
            expected_direction = (
                "towards low disability"
            )

        else:
            expected_direction = (
                "with no directional contribution"
            )

        shap_direction_records.append(
            {
                "Feature": feature,
                "SHAP_Value": shap_value,
                "Expected_Direction": expected_direction,
            }
        )


    # --------------------------------------------------------
    # 4. Final validation status
    # --------------------------------------------------------

    validation_status = (
        "PASS"
        if len(validation_flags) == 0
        else "FLAGGED"
    )

    return {
        "Validation_Status": validation_status,
        "Number_of_Flags": len(validation_flags),
        "Flags": validation_flags,
        "Predicted_Probability": predicted_probability,
        "Distance_From_Threshold": distance_from_threshold,
        "Borderline_Margin": BORDERLINE_MARGIN,
        "Expected_Borderline_Status": is_borderline,
        "SHAP_Direction_Audit": shap_direction_records,
    }

# ============================================================
# VALIDATE PATIENT 11 GENERATED EXPLANATION
# ============================================================

patient_11_validation = validate_agentic_explanation(
    explanation=patient_11_reusable_result["explanation"],
    shap_df=patient_11_reusable_result["shap"],
    predicted_probability=(
        patient_11_reusable_result[
            "prediction"
        ]["Predicted_Probability"]
    ),
)


print_section(
    "PATIENT 11 AGENTIC AI FAITHFULNESS VALIDATION"
)

print(
    "Validation status:",
    patient_11_validation["Validation_Status"]
)

print(
    "Number of flags:",
    patient_11_validation["Number_of_Flags"]
)

if patient_11_validation["Flags"]:
    print("\nPotential faithfulness issues:")

    for flag in patient_11_validation["Flags"]:
        print(
            f"- {flag['Term']}: {flag['Reason']}"
        )
else:
    print(
        "\nNo predefined unsupported SHAP interpretations detected."
    )

# ============================================================
# SAVE PATIENT 11 FAITHFULNESS VALIDATION RESULTS
# ============================================================

patient_11_validation_file = (
    EXPERIMENT_04_DIR
    / "patient_11"
    / "patient_11_faithfulness_validation.json"
)

with open(
    patient_11_validation_file,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        patient_11_validation,
        file,
        indent=4
    )

logger.info(
    "Patient 11 faithfulness validation results saved successfully."
)

print(
    "\nFaithfulness validation saved to:"
)

print(
    patient_11_validation_file
)

print_section(
    "PATIENT 11 FAITHFULNESS VALIDATION SAVED"
)

# ============================================================
# REUSABLE FAITHFULNESS VALIDATION RUNNER
# ============================================================

def run_faithfulness_validation(
    patient_id,
    agentic_result,
    output_dir
):
    """
    Validate and save the faithfulness results for one
    Agentic AI patient explanation.
    """

    validation = validate_agentic_explanation(
        explanation=agentic_result["explanation"],
        shap_df=agentic_result["shap"],
        predicted_probability=(
            agentic_result["prediction"][
                "Predicted_Probability"
            ]
        ),
    )

    print_section(
        f"PATIENT {patient_id} AGENTIC AI FAITHFULNESS VALIDATION"
    )

    print(
        "Validation status:",
        validation["Validation_Status"]
    )

    print(
        "Number of flags:",
        validation["Number_of_Flags"]
    )

    if validation["Flags"]:
        print(
            "\nPotential faithfulness issues:"
        )

        for flag in validation["Flags"]:
            print(
                f"- {flag['Term']}: {flag['Reason']}"
            )

    else:
        print(
            "\nNo predefined unsupported SHAP interpretations detected."
        )

    patient_output_dir = (
        output_dir / f"patient_{patient_id}"
    )

    patient_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    validation_file = (
        patient_output_dir
        / f"patient_{patient_id}_faithfulness_validation.json"
    )

    with open(
        validation_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            validation,
            file,
            indent=4
        )

    logger.info(
        f"Patient {patient_id} faithfulness validation results "
        "saved successfully."
    )

    print(
        "\nFaithfulness validation saved to:"
    )

    print(
        validation_file
    )

    print_section(
        f"PATIENT {patient_id} FAITHFULNESS VALIDATION SAVED"
    )

    return validation

# ============================================================
# RUN PATIENT 7 - MODERATE/HIGH DISABILITY CASE
# ============================================================

patient_7_reusable_result = run_agentic_patient_case(
    patient_id=7,
    prediction_df=prediction_summary_df,
    shap_df=local_shap_df,
    evidence_df=clinical_evidence_chunks_df,
    embedding_model=embedding_model,
    output_dir=EXPERIMENT_04_DIR,
    top_n_shap=5,
    top_k_evidence=TOP_K_EVIDENCE
)

patient_7_validation = run_faithfulness_validation(
    patient_id=7,
    agentic_result=patient_7_reusable_result,
    output_dir=EXPERIMENT_04_DIR
)

# ============================================================
# RUN PATIENT 37 - LOW DISABILITY CASE
# ============================================================

patient_37_reusable_result = run_agentic_patient_case(
    patient_id=37,
    prediction_df=prediction_summary_df,
    shap_df=local_shap_df,
    evidence_df=clinical_evidence_chunks_df,
    embedding_model=embedding_model,
    output_dir=EXPERIMENT_04_DIR,
    top_n_shap=5,
    top_k_evidence=TOP_K_EVIDENCE
)

patient_37_validation = run_faithfulness_validation(
    patient_id=37,
    agentic_result=patient_37_reusable_result,
    output_dir=EXPERIMENT_04_DIR
)

# ============================================================
# EXPERIMENT 4: AGENTIC XAI CASE COMPARISON SUMMARY
# ============================================================

comparison_records = []

representative_cases = [
    ("Moderate/High Disability", 7, patient_7_reusable_result, patient_7_validation),
    ("Borderline", 11, patient_11_reusable_result, patient_11_validation),
    ("Low Disability", 37, patient_37_reusable_result, patient_37_validation),
]

for scenario, patient_id, result, validation in representative_cases:

    prediction_row = prediction_summary_df[
        prediction_summary_df["Patient_ID"] == patient_id
    ].iloc[0]

    top_shap = result["shap"]
    top_evidence = result["top_evidence"]

    comparison_records.append(
        {
            "Patient_ID": patient_id,
            "Scenario": scenario,
            "Actual_Class": int(prediction_row["Actual_Class"]),
            "Predicted_Class": int(prediction_row["Predicted_Class"]),
            "Prediction_Probability": float(
                prediction_row["Predicted_Probability"]
            ),
            "Prediction_Correct": (
                int(prediction_row["Actual_Class"])
                == int(prediction_row["Predicted_Class"])
            ),
            "Top_SHAP_Features": "; ".join(
                top_shap["Feature"].astype(str).tolist()
            ),
            "Number_of_SHAP_Features": len(top_shap),
            "Retrieved_Evidence_Count": len(top_evidence),
            "Retrieved_Evidence_IDs": "; ".join(
                top_evidence["Chunk_ID"].astype(str).tolist()
            ),
            "Agentic_Explanation_Generated": bool(
                result["explanation"].strip()
            ),
            "Faithfulness_Status": validation["Validation_Status"],
            "Faithfulness_Flags": validation["Number_of_Flags"],
        }
    )

agentic_xai_comparison_df = pd.DataFrame(
    comparison_records
)

comparison_file = (
    EXPERIMENT_04_DIR
    / "agentic_xai_comparison_summary.csv"
)

agentic_xai_comparison_df.to_csv(
    comparison_file,
    index=False
)

print_section(
    "EXPERIMENT 4 AGENTIC XAI CASE COMPARISON SUMMARY"
)

print(
    agentic_xai_comparison_df.to_string(index=False)
)

print(
    "\nComparison summary saved to:"
)

print(
    comparison_file
)

logger.info(
    "Experiment 4 Agentic XAI case comparison summary "
    "saved successfully."
)

print_section(
    "EXPERIMENT 4 COMPARISON SUMMARY COMPLETE"
)

