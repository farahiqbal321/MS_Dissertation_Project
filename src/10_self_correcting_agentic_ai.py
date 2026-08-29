# ============================================================
# EXPERIMENT 6
# SELF-CORRECTING AGENTIC AI
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

EXPERIMENT_06_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "experiment_06_self_correcting_agentic_ai"
)


# ============================================================
# EXPERIMENT 6 OUTPUT DIRECTORIES
# ============================================================

ORIGINAL_DIR = EXPERIMENT_06_DIR / "original"

CORRECTION_PROMPTS_DIR = (
    EXPERIMENT_06_DIR
    / "correction_prompts"
)

CORRECTED_DIR = (
    EXPERIMENT_06_DIR
    / "corrected"
)

VALIDATION_DIR = (
    EXPERIMENT_06_DIR
    / "validation"
)

AUDIT_TRAILS_DIR = (
    EXPERIMENT_06_DIR
    / "audit_trails"
)

TABLES_DIR = (
    EXPERIMENT_06_DIR
    / "tables"
)

FIGURES_DIR = (
    EXPERIMENT_06_DIR
    / "figures"
)

LOGS_DIR = (
    EXPERIMENT_06_DIR
    / "logs"
)


experiment_06_directories = [
    ORIGINAL_DIR,
    CORRECTION_PROMPTS_DIR,
    CORRECTED_DIR,
    VALIDATION_DIR,
    AUDIT_TRAILS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    LOGS_DIR,
]


# ============================================================
# CREATE / VERIFY OUTPUT DIRECTORIES
# ============================================================

for directory in experiment_06_directories:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

PATIENT_IDS = [7, 11, 37]

OLLAMA_MODEL = "llama3.1:8b"

GENERATION_TEMPERATURE = 0.2

GENERATION_SEED = 42

MAX_CORRECTION_ATTEMPTS = 2


# ============================================================
# INITIAL EXPERIMENT CHECK
# ============================================================

print("=" * 70)

print(
    "EXPERIMENT 6: "
    "SELF-CORRECTING AGENTIC AI"
)

print("=" * 70)


print("\nProject root:")
print(PROJECT_ROOT)


print("\nExperiment 4 source directory:")
print(EXPERIMENT_04_DIR)


print("\nExperiment 5 source directory:")
print(EXPERIMENT_05_DIR)


print("\nExperiment 6 output directory:")
print(EXPERIMENT_06_DIR)


print("\nRepresentative patient IDs:")
print(PATIENT_IDS)


print("\nSelf-correction settings:")

print(
    f"Model: {OLLAMA_MODEL}"
)

print(
    f"Temperature: "
    f"{GENERATION_TEMPERATURE}"
)

print(
    f"Seed: {GENERATION_SEED}"
)

print(
    f"Maximum correction attempts: "
    f"{MAX_CORRECTION_ATTEMPTS}"
)


print("\nExperiment 6 directories:")

for directory in experiment_06_directories:

    print(
        f"  [OK] {directory.name}"
    )


print("\n" + "=" * 70)

print(
    "EXPERIMENT 6 INITIALISATION COMPLETE"
)

print("=" * 70)

# ============================================================
# STEP 2
# LOAD ORIGINAL WITH-RAG BASELINE DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: LOADING ORIGINAL WITH-RAG BASELINE DATA")
print("=" * 70)


baseline_patient_data = {}


for patient_id in PATIENT_IDS:

    print(
        f"\nLoading baseline data for Patient {patient_id}..."
    )

    patient_dir = (
        EXPERIMENT_04_DIR
        / f"patient_{patient_id}"
    )

    # --------------------------------------------------------
    # Required Experiment 4 files
    # --------------------------------------------------------

    explanation_file = (
        patient_dir
        / f"patient_{patient_id}_agentic_explanation.txt"
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

    metadata_file = (
        patient_dir
        / f"patient_{patient_id}_generation_metadata.json"
    )


    required_files = {
        "Explanation": explanation_file,
        "SHAP": shap_file,
        "Evidence": evidence_file,
        "Validation": validation_file,
        "Metadata": metadata_file,
    }


    # --------------------------------------------------------
    # Verify files exist
    # --------------------------------------------------------

    missing_files = []

    for file_type, file_path in required_files.items():

        if file_path.exists():

            print(
                f"  [OK] {file_type}: "
                f"{file_path.name}"
            )

        else:

            print(
                f"  [MISSING] {file_type}: "
                f"{file_path.name}"
            )

            missing_files.append(
                file_type
            )


    if missing_files:

        raise FileNotFoundError(
            f"Patient {patient_id} is missing "
            f"required Experiment 4 files: "
            f"{missing_files}"
        )


    # --------------------------------------------------------
    # Load baseline data
    # --------------------------------------------------------

    original_explanation = (
        explanation_file.read_text(
            encoding="utf-8"
        )
    )

    shap_df = pd.read_csv(
        shap_file
    )

    evidence_df = pd.read_csv(
        evidence_file
    )

    with open(
        validation_file,
        "r",
        encoding="utf-8"
    ) as file:

        validation_data = json.load(
            file
        )


    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )


    # --------------------------------------------------------
    # Store baseline patient data
    # --------------------------------------------------------

    baseline_patient_data[
        patient_id
    ] = {

        "explanation":
            original_explanation,

        "shap":
            shap_df,

        "evidence":
            evidence_df,

        "validation":
            validation_data,

        "metadata":
            metadata,

    }


    # --------------------------------------------------------
    # Preserve copy of original explanation
    # --------------------------------------------------------

    original_copy_file = (
        ORIGINAL_DIR
        / f"patient_{patient_id}_original_explanation.txt"
    )

    original_copy_file.write_text(
        original_explanation,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Baseline summary
    # --------------------------------------------------------

    print(
        "  SHAP features loaded:",
        len(shap_df)
    )

    print(
        "  Evidence chunks loaded:",
        len(evidence_df)
    )

    print(
        "  Baseline validation status:",
        validation_data.get(
            "Validation_Status",
            "UNKNOWN"
        )
    )

    print(
        "  Baseline faithfulness flags:",
        validation_data.get(
            "Number_of_Flags",
            "UNKNOWN"
        )
    )

    print(
        "  Original explanation copied to:",
        original_copy_file.name
    )


# ============================================================
# STEP 2 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 6 BASELINE DATA SUMMARY")
print("=" * 70)

print(
    "Patients successfully loaded:",
    list(
        baseline_patient_data.keys()
    )
)

print(
    "Number of patient cases:",
    len(
        baseline_patient_data
    )
)

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 3
# BUILD STRUCTURED SELF-CORRECTION FEEDBACK
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: BUILDING SELF-CORRECTION FEEDBACK")
print("=" * 70)


correction_feedback = {}


for patient_id in PATIENT_IDS:

    print(
        f"\nProcessing validation feedback for Patient {patient_id}..."
    )

    patient_data = baseline_patient_data[
        patient_id
    ]

    validation = patient_data[
        "validation"
    ]

    validation_status = validation.get(
        "Validation_Status",
        "UNKNOWN"
    )

    flags = validation.get(
        "Flags",
        []
    )


    # --------------------------------------------------------
    # Build structured flag descriptions
    # --------------------------------------------------------

    flag_descriptions = []

    for flag_number, flag in enumerate(
        flags,
        start=1
    ):

        term = flag.get(
            "Term",
            "Unspecified term"
        )

        reason = flag.get(
            "Reason",
            "No reason supplied"
        )

        flag_text = (
            f"Flag {flag_number}: "
            f"Term = '{term}'. "
            f"Reason = {reason}"
        )

        flag_descriptions.append(
            flag_text
        )


    # --------------------------------------------------------
    # Create correction instruction
    # --------------------------------------------------------

    if flags:

        correction_instruction = (
            "The original explanation failed the automated "
            "faithfulness validation. Revise only the unsupported "
            "or over-interpreted statements identified below. "
            "Preserve statements that are supported by the SHAP "
            "results and retrieved clinical evidence. Do not invent "
            "new patient information, causal relationships, diagnoses, "
            "treatment recommendations, or unsupported clinical claims.\n\n"
            "VALIDATION FEEDBACK:\n"
            + "\n".join(flag_descriptions)
        )

    else:

        correction_instruction = (
            "No predefined faithfulness violations were detected. "
            "Preserve the original explanation without introducing "
            "new clinical claims."
        )


    # --------------------------------------------------------
    # Store feedback
    # --------------------------------------------------------

    correction_feedback[
        patient_id
    ] = {

        "validation_status":
            validation_status,

        "number_of_flags":
            len(flags),

        "flags":
            flags,

        "flag_descriptions":
            flag_descriptions,

        "correction_instruction":
            correction_instruction,

    }


    # --------------------------------------------------------
    # Display patient feedback
    # --------------------------------------------------------

    print(
        "  Validation status:",
        validation_status
    )

    print(
        "  Number of flags:",
        len(flags)
    )


    if flag_descriptions:

        print(
            "  Correction targets:"
        )

        for description in flag_descriptions:

            print(
                "   -",
                description
            )

    else:

        print(
            "  No correction targets identified."
        )


# ============================================================
# SAVE STRUCTURED FEEDBACK
# ============================================================

feedback_summary_file = (
    TABLES_DIR
    / "experiment_06_baseline_validation_feedback.json"
)


serialisable_feedback = {}

for patient_id, feedback in correction_feedback.items():

    serialisable_feedback[
        str(patient_id)
    ] = feedback


with open(
    feedback_summary_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        serialisable_feedback,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("EXPERIMENT 6 CORRECTION FEEDBACK SUMMARY")
print("=" * 70)

for patient_id in PATIENT_IDS:

    feedback = correction_feedback[
        patient_id
    ]

    print(
        f"Patient {patient_id}: "
        f"{feedback['validation_status']} | "
        f"Flags = {feedback['number_of_flags']}"
    )


print(
    "\nFeedback summary saved to:"
)

print(
    feedback_summary_file
)


print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 4
# BUILD SELF-CORRECTION PROMPTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: BUILDING SELF-CORRECTION PROMPTS")
print("=" * 70)


def build_self_correction_prompt(
    patient_id,
    patient_data,
    feedback
):
    """
    Build a validator-guided correction prompt for one patient.
    """

    original_explanation = patient_data[
        "explanation"
    ]

    shap_df = patient_data[
        "shap"
    ]

    evidence_df = patient_data[
        "evidence"
    ]

    metadata = patient_data[
        "metadata"
    ]


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
    # Build retrieved evidence context
    # --------------------------------------------------------

    evidence_lines = []

    for _, row in evidence_df.iterrows():

        chunk_id = str(
            row.get(
                "Chunk_ID",
                "UNKNOWN"
            )
        )

        evidence_text = str(
            row.get(
                "Evidence_Text",
                row.get(
                    "Text",
                    row.get(
                        "Content",
                        ""
                    )
                )
            )
        )

        evidence_lines.append(
            f"[{chunk_id}] {evidence_text}"
        )


    evidence_context = "\n".join(
        evidence_lines
    )


    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    predicted_probability = float(
        metadata[
            "Prediction_Probability"
        ]
    )

    predicted_class = int(
        metadata[
            "Predicted_Class"
        ]
    )

    actual_class = int(
        metadata[
            "Actual_Class"
        ]
    )


    # --------------------------------------------------------
    # Validator feedback
    # --------------------------------------------------------

    correction_instruction = feedback[
        "correction_instruction"
    ]


    # --------------------------------------------------------
    # Construct correction prompt
    # --------------------------------------------------------

    correction_prompt = f"""
You are a self-correcting AI explanation agent supporting
interpretation of a machine-learning prediction for
Multiple Sclerosis disability.

Your task is to revise the ORIGINAL EXPLANATION using the
VALIDATION FEEDBACK below.

IMPORTANT RULES:

1. Preserve the machine-learning prediction exactly.
2. Preserve all supplied SHAP values and their directions.
3. Use only the retrieved clinical evidence supplied below.
4. Do not invent patient facts, clinical associations,
   diagnoses, treatment recommendations or disease progression.
5. Do not infer whether an underlying feature value is
   younger, older, better, worse, increased, decreased,
   improved or poorer from SHAP direction alone.
6. A positive SHAP value must only be described as contributing
   towards moderate/high disability.
7. A negative SHAP value must only be described as contributing
   towards low disability.
8. A probability should only be described as borderline when
   it falls within +/- 0.05 of the 0.50 classification threshold.
9. Preserve valid retrieved evidence identifiers.
10. Correct only unsupported or inaccurate statements where
    possible. Do not unnecessarily rewrite valid content.
11. Keep the output concise, structured and suitable for
    decision support.
12. State that the explanation supports, but does not replace,
    professional clinical judgement.

PATIENT INFORMATION

Patient ID: {patient_id}
Actual class: {actual_class}
Predicted class: {predicted_class}
Predicted probability of moderate/high disability:
{predicted_probability:.4f}

PATIENT-SPECIFIC SHAP INFORMATION

{shap_context}

RETRIEVED CLINICAL EVIDENCE

{evidence_context}

VALIDATION FEEDBACK

{correction_instruction}

ORIGINAL EXPLANATION

{original_explanation}

TASK

Return the corrected explanation only.

Use the following structure:

1. Prediction Summary
2. Main Factors Supporting Moderate/High Disability
3. Main Factors Supporting Low Disability
4. Clinical Context from Retrieved Evidence
5. Uncertainty and Limitations
6. Decision-Support Summary
""".strip()

    return correction_prompt


# ============================================================
# BUILD AND SAVE CORRECTION PROMPTS
# ============================================================

self_correction_prompts = {}


for patient_id in PATIENT_IDS:

    patient_data = baseline_patient_data[
        patient_id
    ]

    feedback = correction_feedback[
        patient_id
    ]


    # --------------------------------------------------------
    # Patient 7 already passes validation
    # --------------------------------------------------------

    if feedback["number_of_flags"] == 0:

        print(
            f"\nPatient {patient_id}: "
            "baseline explanation already PASS."
        )

        print(
            "No correction prompt required."
        )

        self_correction_prompts[
            patient_id
        ] = None

        continue


    # --------------------------------------------------------
    # Build prompt for flagged explanation
    # --------------------------------------------------------

    correction_prompt = (
        build_self_correction_prompt(
            patient_id=patient_id,
            patient_data=patient_data,
            feedback=feedback
        )
    )


    self_correction_prompts[
        patient_id
    ] = correction_prompt


    prompt_file = (
        CORRECTION_PROMPTS_DIR
        / f"patient_{patient_id}_correction_attempt_1_prompt.txt"
    )


    prompt_file.write_text(
        correction_prompt,
        encoding="utf-8"
    )


    print(
        f"\nPatient {patient_id}: "
        "correction prompt created."
    )

    print(
        "Saved to:",
        prompt_file.name
    )


# ============================================================
# STEP 4 SUMMARY
# ============================================================

patients_requiring_correction = [

    patient_id

    for patient_id, prompt
    in self_correction_prompts.items()

    if prompt is not None
]


print("\n" + "=" * 70)
print("SELF-CORRECTION PROMPT SUMMARY")
print("=" * 70)

print(
    "Patients requiring correction:",
    patients_requiring_correction
)

print(
    "Number requiring correction:",
    len(
        patients_requiring_correction
    )
)

print("\n" + "=" * 70)
print("STEP 4 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 5
# GENERATE SELF-CORRECTED EXPLANATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: GENERATING SELF-CORRECTED EXPLANATIONS")
print("=" * 70)


def generate_corrected_explanation(prompt):
    """
    Generate one validator-guided corrected explanation
    using the local Llama 3.1 8B model through Ollama.
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

    corrected_text = (
        response_data["response"].strip()
    )

    return corrected_text


# ============================================================
# GENERATE CORRECTED EXPLANATIONS
# ============================================================

corrected_explanations = {}


for patient_id in PATIENT_IDS:

    feedback = correction_feedback[
        patient_id
    ]


    # --------------------------------------------------------
    # PASS CASES: PRESERVE ORIGINAL EXPLANATION
    # --------------------------------------------------------

    if feedback["number_of_flags"] == 0:

        original_explanation = (
            baseline_patient_data[
                patient_id
            ]["explanation"]
        )

        corrected_explanations[
            patient_id
        ] = original_explanation


        output_file = (
            CORRECTED_DIR
            / f"patient_{patient_id}_corrected_explanation.txt"
        )

        output_file.write_text(
            original_explanation,
            encoding="utf-8"
        )


        print(
            f"\nPatient {patient_id}: "
            "no correction required."
        )

        print(
            "Original PASS explanation preserved."
        )

        print(
            "Saved to:",
            output_file.name
        )

        continue


    # --------------------------------------------------------
    # FLAGGED CASES: GENERATE SELF-CORRECTION
    # --------------------------------------------------------

    print(
        f"\nGenerating corrected explanation "
        f"for Patient {patient_id}..."
    )


    correction_prompt = (
        self_correction_prompts[
            patient_id
        ]
    )


    corrected_text = (
        generate_corrected_explanation(
            correction_prompt
        )
    )


    corrected_explanations[
        patient_id
    ] = corrected_text


    # --------------------------------------------------------
    # SAVE CORRECTED EXPLANATION
    # --------------------------------------------------------

    output_file = (
        CORRECTED_DIR
        / f"patient_{patient_id}_corrected_explanation.txt"
    )

    output_file.write_text(
        corrected_text,
        encoding="utf-8"
    )


    print(
        f"Patient {patient_id}: "
        "corrected explanation generated."
    )

    print(
        "Saved to:",
        output_file.name
    )


# ============================================================
# STEP 5 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SELF-CORRECTION GENERATION SUMMARY")
print("=" * 70)


for patient_id in PATIENT_IDS:

    if (
        correction_feedback[
            patient_id
        ]["number_of_flags"] == 0
    ):

        status = "ORIGINAL PRESERVED"

    else:

        status = "CORRECTED"


    print(
        f"Patient {patient_id}: {status}"
    )


print(
    "\nNumber of explanations available:",
    len(corrected_explanations)
)


print("\n" + "=" * 70)
print("STEP 5 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 6
# RE-VALIDATE SELF-CORRECTED EXPLANATIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: RE-VALIDATING SELF-CORRECTED EXPLANATIONS")
print("=" * 70)


# ============================================================
# REUSE THE SAME FAITHFULNESS VALIDATOR
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


    if (
        explanation_calls_borderline
        and not is_borderline
    ):

        validation_flags.append(
            {
                "Flag_Type":
                    "Unsupported_Uncertainty_Interpretation",

                "Term":
                    "borderline/close to threshold",

                "Reason": (
                    f"The predicted probability is "
                    f"{predicted_probability:.4f}, which is "
                    f"{distance_from_threshold:.4f} away from the "
                    "0.50 classification threshold and therefore falls "
                    "outside the predefined borderline margin of +/- 0.05."
                ),
            }
        )


    if (
        is_borderline
        and not explanation_calls_borderline
    ):

        validation_flags.append(
            {
                "Flag_Type":
                    "Missing_Uncertainty_Acknowledgement",

                "Term":
                    "borderline uncertainty",

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
                "Expected_Direction":
                    expected_direction,
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
        "Validation_Status":
            validation_status,

        "Number_of_Flags":
            len(validation_flags),

        "Flags":
            validation_flags,

        "Predicted_Probability":
            predicted_probability,

        "Distance_From_Threshold":
            distance_from_threshold,

        "Borderline_Margin":
            BORDERLINE_MARGIN,

        "Expected_Borderline_Status":
            is_borderline,

        "SHAP_Direction_Audit":
            shap_direction_records,
    }


# ============================================================
# RUN RE-VALIDATION
# ============================================================

corrected_validation_results = {}


for patient_id in PATIENT_IDS:

    print(
        f"\nRe-validating Patient {patient_id}..."
    )


    patient_data = (
        baseline_patient_data[
            patient_id
        ]
    )


    corrected_explanation = (
        corrected_explanations[
            patient_id
        ]
    )


    shap_df = (
        patient_data[
            "shap"
        ]
    )


    predicted_probability = float(
        patient_data[
            "metadata"
        ][
            "Prediction_Probability"
        ]
    )


    corrected_validation = (
        validate_agentic_explanation(
            explanation=corrected_explanation,
            shap_df=shap_df,
            predicted_probability=predicted_probability
        )
    )


    corrected_validation_results[
        patient_id
    ] = corrected_validation


    # --------------------------------------------------------
    # SAVE CORRECTED VALIDATION RESULT
    # --------------------------------------------------------

    validation_output_file = (
        VALIDATION_DIR
        / f"patient_{patient_id}_corrected_validation.json"
    )


    with open(
        validation_output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            corrected_validation,
            file,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    original_validation = (
        patient_data[
            "validation"
        ]
    )


    original_flags = int(
        original_validation.get(
            "Number_of_Flags",
            0
        )
    )


    corrected_flags = int(
        corrected_validation[
            "Number_of_Flags"
        ]
    )


    flag_reduction = (
        original_flags
        - corrected_flags
    )


    print(
        "  Original status:",
        original_validation.get(
            "Validation_Status",
            "UNKNOWN"
        )
    )

    print(
        "  Original flags:",
        original_flags
    )

    print(
        "  Corrected status:",
        corrected_validation[
            "Validation_Status"
        ]
    )

    print(
        "  Corrected flags:",
        corrected_flags
    )

    print(
        "  Flag reduction:",
        flag_reduction
    )


    if corrected_validation["Flags"]:

        print(
            "  Remaining issues:"
        )

        for flag in corrected_validation[
            "Flags"
        ]:

            print(
                f"   - {flag['Term']}: "
                f"{flag['Reason']}"
            )

    else:

        print(
            "  No predefined faithfulness issues detected."
        )


    print(
        "  Saved to:",
        validation_output_file.name
    )


# ============================================================
# BUILD BEFORE-VS-AFTER COMPARISON TABLE
# ============================================================

comparison_rows = []


for patient_id in PATIENT_IDS:

    original_validation = (
        baseline_patient_data[
            patient_id
        ][
            "validation"
        ]
    )


    corrected_validation = (
        corrected_validation_results[
            patient_id
        ]
    )


    original_flags = int(
        original_validation.get(
            "Number_of_Flags",
            0
        )
    )


    corrected_flags = int(
        corrected_validation[
            "Number_of_Flags"
        ]
    )


    comparison_rows.append(
        {
            "Patient_ID":
                patient_id,

            "Original_Status":
                original_validation.get(
                    "Validation_Status",
                    "UNKNOWN"
                ),

            "Original_Flags":
                original_flags,

            "Corrected_Status":
                corrected_validation[
                    "Validation_Status"
                ],

            "Corrected_Flags":
                corrected_flags,

            "Flag_Reduction":
                original_flags
                - corrected_flags,

            "Improved":
                corrected_flags
                < original_flags,
        }
    )


self_correction_comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

comparison_file = (
    TABLES_DIR
    / "experiment_06_self_correction_comparison.csv"
)


self_correction_comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 6 BEFORE-VS-AFTER SUMMARY")
print("=" * 70)


print(
    self_correction_comparison_df.to_string(
        index=False
    )
)


total_original_flags = int(
    self_correction_comparison_df[
        "Original_Flags"
    ].sum()
)


total_corrected_flags = int(
    self_correction_comparison_df[
        "Corrected_Flags"
    ].sum()
)


total_flag_reduction = (
    total_original_flags
    - total_corrected_flags
)


print(
    "\nTotal original flags:",
    total_original_flags
)

print(
    "Total corrected flags:",
    total_corrected_flags
)

print(
    "Total flag reduction:",
    total_flag_reduction
)


print(
    "\nComparison table saved to:"
)

print(
    comparison_file
)


print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 7
# SECOND SELF-CORRECTION ATTEMPT FOR REMAINING FLAGGED CASES
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: SECOND SELF-CORRECTION ATTEMPT")
print("=" * 70)


# ------------------------------------------------------------
# Identify patients still requiring correction
# ------------------------------------------------------------

patients_requiring_second_correction = []

for patient_id in PATIENT_IDS:

    validation = corrected_validation_results[
        patient_id
    ]

    if (
        validation["Validation_Status"]
        == "FLAGGED"
    ):

        patients_requiring_second_correction.append(
            patient_id
        )


print(
    "\nPatients requiring second correction:",
    patients_requiring_second_correction
)

print(
    "Number requiring second correction:",
    len(
        patients_requiring_second_correction
    )
)


# ------------------------------------------------------------
# Store final explanations
#
# Begin with the explanations produced after Attempt 1.
# PASS cases are preserved without unnecessary regeneration.
# ------------------------------------------------------------

final_explanations = dict(
    corrected_explanations
)

second_attempt_results = {}


# ------------------------------------------------------------
# Process remaining FLAGGED cases
# ------------------------------------------------------------

for patient_id in patients_requiring_second_correction:

    print("\n" + "-" * 70)

    print(
        f"Running correction attempt 2 "
        f"for Patient {patient_id}..."
    )


    patient_data = (
        baseline_patient_data[
            patient_id
        ]
    )


    current_explanation = (
        corrected_explanations[
            patient_id
        ]
    )


    current_validation = (
        corrected_validation_results[
            patient_id
        ]
    )


    # --------------------------------------------------------
    # Convert validator feedback into correction instructions
    # --------------------------------------------------------

    feedback_lines = []

    for index, flag in enumerate(
        current_validation["Flags"],
        start=1
    ):

        feedback_lines.append(
            f"{index}. "
            f"Flag type: {flag['Flag_Type']}\n"
            f"   Problematic term: {flag['Term']}\n"
            f"   Reason: {flag['Reason']}"
        )


    validator_feedback = "\n".join(
        feedback_lines
    )


    # --------------------------------------------------------
    # Build second correction prompt
    # --------------------------------------------------------

    second_correction_prompt = f"""
You are performing a second faithfulness-correction pass on an
AI-generated clinical decision-support explanation.

Your task is to revise the explanation ONLY where necessary to
resolve the validator feedback below.

IMPORTANT REQUIREMENTS:

1. Preserve the original prediction and patient-specific meaning.
2. Preserve supported SHAP information.
3. Do not invent clinical facts.
4. Do not invent evidence.
5. Do not change numerical values unless they are incorrect.
6. Do not describe SHAP direction as proving that an underlying
   patient feature is younger, older, better, worse, increased,
   decreased, improved or poorer.
7. A prediction should only be described as borderline or close
   to the 0.50 classification threshold when it falls within
   +/- 0.05 of 0.50.
8. Remove or rewrite every statement identified by the validator.
9. Keep the explanation suitable for clinical decision support
   rather than diagnosis.
10. Return ONLY the revised explanation. Do not describe the
    editing process.

CURRENT EXPLANATION:

{current_explanation}


VALIDATOR FEEDBACK:

{validator_feedback}


REVISED EXPLANATION:
""".strip()


    # --------------------------------------------------------
    # Save second correction prompt
    # --------------------------------------------------------

    second_prompt_file = (
        CORRECTION_PROMPTS_DIR
        / (
            f"patient_{patient_id}_"
            "correction_attempt_2_prompt.txt"
        )
    )


    second_prompt_file.write_text(
        second_correction_prompt,
        encoding="utf-8"
    )


    print(
        "Second correction prompt saved to:",
        second_prompt_file.name
    )


    # --------------------------------------------------------
    # Generate second corrected explanation
    # --------------------------------------------------------

    print(
        f"Generating correction attempt 2 "
        f"for Patient {patient_id}..."
    )

    second_corrected_explanation = (
    generate_corrected_explanation(
        second_correction_prompt
    )
)

    # --------------------------------------------------------
    # Save second corrected explanation
    # --------------------------------------------------------

    second_corrected_file = (
        CORRECTED_DIR
        / (
            f"patient_{patient_id}_"
            "corrected_attempt_2_explanation.txt"
        )
    )


    second_corrected_file.write_text(
        second_corrected_explanation,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Re-run SAME validator
    # --------------------------------------------------------

    shap_df = (
        patient_data[
            "shap"
        ]
    )


    predicted_probability = float(
        patient_data[
            "metadata"
        ][
            "Prediction_Probability"
        ]
    )


    second_validation = (
        validate_agentic_explanation(
            explanation=
                second_corrected_explanation,

            shap_df=
                shap_df,

            predicted_probability=
                predicted_probability,
        )
    )


    second_attempt_results[
        patient_id
    ] = second_validation


    # --------------------------------------------------------
    # Save second-attempt validation
    # --------------------------------------------------------

    second_validation_file = (
        VALIDATION_DIR
        / (
            f"patient_{patient_id}_"
            "attempt_2_validation.json"
        )
    )


    with open(
        second_validation_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            second_validation,
            file,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # Update final explanation
    # --------------------------------------------------------

    final_explanations[
        patient_id
    ] = second_corrected_explanation


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    attempt_1_flags = int(
        current_validation[
            "Number_of_Flags"
        ]
    )


    attempt_2_flags = int(
        second_validation[
            "Number_of_Flags"
        ]
    )


    print(
        "\nAttempt 1 status:",
        current_validation[
            "Validation_Status"
        ]
    )

    print(
        "Attempt 1 flags:",
        attempt_1_flags
    )

    print(
        "Attempt 2 status:",
        second_validation[
            "Validation_Status"
        ]
    )

    print(
        "Attempt 2 flags:",
        attempt_2_flags
    )

    print(
        "Additional flag reduction:",
        attempt_1_flags
        - attempt_2_flags
    )


    if second_validation["Flags"]:

        print(
            "\nIssues remaining after "
            "maximum correction attempts:"
        )

        for flag in second_validation[
            "Flags"
        ]:

            print(
                f"  - {flag['Term']}: "
                f"{flag['Reason']}"
            )

    else:

        print(
            "\n[PASS] All predefined "
            "faithfulness issues resolved."
        )


    print(
        "\nSecond-attempt explanation saved to:",
        second_corrected_file.name
    )

    print(
        "Second-attempt validation saved to:",
        second_validation_file.name
    )


# ============================================================
# STEP 7 SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SECOND CORRECTION ATTEMPT SUMMARY")
print("=" * 70)


if not patients_requiring_second_correction:

    print(
        "No second correction attempts were required."
    )

else:

    for patient_id in (
        patients_requiring_second_correction
    ):

        result = (
            second_attempt_results[
                patient_id
            ]
        )

        print(
            f"Patient {patient_id}: "
            f"{result['Validation_Status']} "
            f"| Flags = "
            f"{result['Number_of_Flags']}"
        )


print("\n" + "=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)

# ============================================================
# STEP 8
# FINAL EXPERIMENT 6 EVALUATION AND SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: FINAL EXPERIMENT 6 EVALUATION")
print("=" * 70)


# ============================================================
# BUILD FINAL PATIENT-LEVEL SUMMARY
# ============================================================

final_summary_rows = []


for patient_id in PATIENT_IDS:

    original_validation = (
        baseline_patient_data[
            patient_id
        ]["validation"]
    )

    attempt_1_validation = (
        corrected_validation_results[
            patient_id
        ]
    )

    original_flags = int(
        original_validation.get(
            "Number_of_Flags",
            0
        )
    )

    attempt_1_flags = int(
        attempt_1_validation[
            "Number_of_Flags"
        ]
    )


    # --------------------------------------------------------
    # Check whether a completed Attempt 2 exists on disk
    # --------------------------------------------------------

    attempt_2_validation_file = (
        VALIDATION_DIR
        / f"patient_{patient_id}_attempt_2_validation.json"
    )


    if attempt_2_validation_file.exists():

        with open(
            attempt_2_validation_file,
            "r",
            encoding="utf-8"
        ) as file:

            attempt_2_validation = json.load(
                file
            )


        attempt_2_status = (
            attempt_2_validation[
                "Validation_Status"
            ]
        )

        attempt_2_flags = int(
            attempt_2_validation[
                "Number_of_Flags"
            ]
        )

        correction_attempts = 2

        final_status = attempt_2_status

        final_flags = attempt_2_flags


    else:

        attempt_2_validation = None

        attempt_2_status = "NOT REQUIRED"

        attempt_2_flags = None


        if original_flags == 0:

            correction_attempts = 0

        else:

            correction_attempts = 1


        final_status = (
            attempt_1_validation[
                "Validation_Status"
            ]
        )

        final_flags = attempt_1_flags


    # --------------------------------------------------------
    # Calculate total reduction
    # --------------------------------------------------------

    total_flag_reduction = (
        original_flags
        - final_flags
    )


    # --------------------------------------------------------
    # Add patient to final table
    # --------------------------------------------------------

    final_summary_rows.append(
        {
            "Patient_ID":
                patient_id,

            "Baseline_Status":
                original_validation.get(
                    "Validation_Status",
                    "UNKNOWN"
                ),

            "Baseline_Flags":
                original_flags,

            "Attempt_1_Status":
                attempt_1_validation[
                    "Validation_Status"
                ],

            "Attempt_1_Flags":
                attempt_1_flags,

            "Attempt_2_Status":
                attempt_2_status,

            "Attempt_2_Flags":
                attempt_2_flags,

            "Correction_Attempts":
                correction_attempts,

            "Final_Status":
                final_status,

            "Final_Flags":
                final_flags,

            "Total_Flag_Reduction":
                total_flag_reduction,
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

experiment_06_final_summary_df = pd.DataFrame(
    final_summary_rows
)


# ============================================================
# DISPLAY PATIENT-LEVEL RESULTS
# ============================================================

print(
    "\nFinal Experiment 6 patient-level summary:\n"
)

print(
    experiment_06_final_summary_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE FINAL SUMMARY TABLE
# ============================================================

final_summary_file = (
    TABLES_DIR
    / "experiment_06_final_summary.csv"
)


experiment_06_final_summary_df.to_csv(
    final_summary_file,
    index=False
)


# ============================================================
# AGGREGATE RESULTS
# ============================================================

total_baseline_flags = int(
    experiment_06_final_summary_df[
        "Baseline_Flags"
    ].sum()
)


total_attempt_1_flags = int(
    experiment_06_final_summary_df[
        "Attempt_1_Flags"
    ].sum()
)


total_final_flags = int(
    experiment_06_final_summary_df[
        "Final_Flags"
    ].sum()
)


total_flags_removed = (
    total_baseline_flags
    - total_final_flags
)


if total_baseline_flags > 0:

    flag_reduction_percent = (
        total_flags_removed
        / total_baseline_flags
        * 100
    )

else:

    flag_reduction_percent = 0.0


patients_final_pass = int(
    (
        experiment_06_final_summary_df[
            "Final_Status"
        ] == "PASS"
    ).sum()
)


print("\n" + "=" * 70)
print("AGGREGATE SELF-CORRECTION RESULTS")
print("=" * 70)


print(
    "Total baseline flags:",
    total_baseline_flags
)

print(
    "Total flags after Attempt 1:",
    total_attempt_1_flags
)

print(
    "Total final flags:",
    total_final_flags
)

print(
    "Total flags removed:",
    total_flags_removed
)

print(
    "Overall flag reduction:",
    f"{flag_reduction_percent:.1f}%"
)

print(
    "Patients with final PASS status:",
    f"{patients_final_pass}/{len(PATIENT_IDS)}"
)


print(
    "\nFinal summary saved to:"
)

print(
    final_summary_file
)


# ============================================================
# CREATE FINAL FIGURE
# ============================================================

import matplotlib.pyplot as plt
import numpy as np


patient_labels = [
    f"Patient {patient_id}"
    for patient_id in PATIENT_IDS
]


baseline_flags = (
    experiment_06_final_summary_df[
        "Baseline_Flags"
    ].tolist()
)


attempt_1_flags = (
    experiment_06_final_summary_df[
        "Attempt_1_Flags"
    ].tolist()
)


final_flags = (
    experiment_06_final_summary_df[
        "Final_Flags"
    ].tolist()
)


x = np.arange(
    len(patient_labels)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(9, 6)
)


bars_baseline = ax.bar(
    x - width,
    baseline_flags,
    width,
    label="Baseline"
)


bars_attempt_1 = ax.bar(
    x,
    attempt_1_flags,
    width,
    label="After Attempt 1"
)


bars_final = ax.bar(
    x + width,
    final_flags,
    width,
    label="Final"
)


ax.set_ylabel(
    "Number of Faithfulness Flags"
)


ax.set_title(
    "Experiment 6: Faithfulness Flags Before and After Self-Correction"
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    patient_labels
)


max_flags = max(
    baseline_flags
    + attempt_1_flags
    + final_flags
)


ax.set_ylim(
    0,
    max_flags + 1
)


ax.legend()


for bars in [
    bars_baseline,
    bars_attempt_1,
    bars_final
]:

    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height
            ),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom"
        )


fig.tight_layout()


final_figure_file = (
    FIGURES_DIR
    / "experiment_06_self_correction_flag_reduction.png"
)


plt.savefig(
    final_figure_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nFinal Experiment 6 figure saved to:"
)

print(
    final_figure_file
)


# ============================================================
# SAVE FINAL AUDIT TRAILS
# ============================================================

for patient_id in PATIENT_IDS:

    attempt_2_validation_file = (
        VALIDATION_DIR
        / f"patient_{patient_id}_attempt_2_validation.json"
    )


    if attempt_2_validation_file.exists():

        with open(
            attempt_2_validation_file,
            "r",
            encoding="utf-8"
        ) as file:

            attempt_2_validation = json.load(
                file
            )

    else:

        attempt_2_validation = None


    patient_summary = (
        experiment_06_final_summary_df[
            experiment_06_final_summary_df[
                "Patient_ID"
            ] == patient_id
        ].iloc[0]
    )


    audit_data = {

        "Patient_ID":
            patient_id,

        "Baseline_Validation":
            baseline_patient_data[
                patient_id
            ]["validation"],

        "Attempt_1_Validation":
            corrected_validation_results[
                patient_id
            ],

        "Attempt_2_Validation":
            attempt_2_validation,

        "Correction_Attempts":
            int(
                patient_summary[
                    "Correction_Attempts"
                ]
            ),

        "Final_Status":
            patient_summary[
                "Final_Status"
            ],

        "Final_Flags":
            int(
                patient_summary[
                    "Final_Flags"
                ]
            ),
    }


    audit_file = (
        AUDIT_TRAILS_DIR
        / f"patient_{patient_id}_self_correction_audit.json"
    )


    with open(
        audit_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            audit_data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# EXPERIMENT 6 COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 6 COMPLETE")
print("=" * 70)

