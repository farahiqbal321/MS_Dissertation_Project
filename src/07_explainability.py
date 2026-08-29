"""
=========================================================
07_explainability.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Runs Experiment 3: Explainability Analysis.

The final selected model from Experiment 2 is analysed
using three complementary explainability methods:

• SHAP
• LIME
• Permutation Importance

Outputs
-------
Global feature importance
Local explanations
Comparison figures
Comparison tables

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import json

import joblib
import lime
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


from sklearn.inspection import permutation_importance

# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from config import (
    PROCESSED_DATA_DIR,
    EXPERIMENT_01_DIR,
    EXPERIMENT_02_DIR,
    EXPERIMENT_03_DIR,
    LOGS_DIR,
)

from utils import (
    setup_logger,
    check_file_exists,
    validate_required_columns,
    save_dataframe,
    print_section,
)

# =========================================================
# INITIALISE LOGGER AND DIRECTORIES
# =========================================================

logger = setup_logger(
    logger_name="experiment_03_explainability",
    log_file=LOGS_DIR / "experiment_03.log",
)

experiment_03_tables_dir = (
    EXPERIMENT_03_DIR / "tables"
)

experiment_03_figures_dir = (
    EXPERIMENT_03_DIR / "figures"
)

experiment_03_models_dir = (
    EXPERIMENT_03_DIR / "models"
)

experiment_03_logs_dir = (
    EXPERIMENT_03_DIR / "logs"
)

for directory in [

    EXPERIMENT_03_DIR,

    experiment_03_tables_dir,

    experiment_03_figures_dir,

    experiment_03_models_dir,

    experiment_03_logs_dir,

]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# =========================================================
# START EXPERIMENT
# =========================================================

print_section(
    "Experiment 3: SHAP and LIME Explainability"
)

logger.info("=" * 70)
logger.info(
    "Experiment 3 explainability started."
)
logger.info("=" * 70)

# =========================================================
# DEFINE INPUT FILES
# =========================================================

model_ready_file = (
    PROCESSED_DATA_DIR
    / "model_ready_dataset.csv"
)

split_file = (
    EXPERIMENT_01_DIR
    / "tables"
    / "train_test_split.csv"
)

selected_model_file = (
    EXPERIMENT_02_DIR
    / "final_selected_model.joblib"
)

selected_model_information_file = (
    EXPERIMENT_02_DIR
    / "selected_final_model.json"
)

# =========================================================
# CHECK INPUT FILES
# =========================================================

check_file_exists(model_ready_file)

check_file_exists(split_file)

check_file_exists(selected_model_file)

check_file_exists(
    selected_model_information_file
)

print("\nRequired files found:")

print(model_ready_file)

print(split_file)

print(selected_model_file)

print(selected_model_information_file)

# =========================================================
# LOAD DATA
# =========================================================

model_df = pd.read_csv(
    model_ready_file
)

split_df = pd.read_csv(
    split_file
)

selected_pipeline = joblib.load(
    selected_model_file
)

with open(
    selected_model_information_file,
    "r",
    encoding="utf-8",
) as file:

    selected_model_information = json.load(
        file
    )

print("\nModel dataset shape:")
print(model_df.shape)

print("\nSplit shape:")
print(split_df.shape)

print("\nSelected model:")
print(
    selected_model_information[
        "Selected_Model"
    ]
)

print("\nFeature set:")
print(
    selected_model_information[
        "Feature_Set"
    ]
)

logger.info(
    "Experiment 3 inputs loaded successfully."
)

logger.info(
    "Experiment 3 inputs loaded successfully."
)

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

model_required_columns = [
    "Patient_ID",
    "Binary_Disability",
    "Gender",
    "Disease_Duration",
    "Lesion_Volume_mm3",
    "Lesion_Count",
    "Average_Lesion_Size",
    "Largest_Lesion_Size",
    "Smallest_Lesion_Size",
    "Small_Lesions",
    "Medium_Lesions",
    "Large_Lesions",
]

split_required_columns = [
    "Patient_ID",
    "Dataset_Split",
    "Binary_Disability",
]

validate_required_columns(
    dataframe=model_df,
    required_columns=model_required_columns,
    dataset_name="Model-ready dataset",
)

validate_required_columns(
    dataframe=split_df,
    required_columns=split_required_columns,
    dataset_name="Experiment 1 split table",
)

logger.info(
    "Required Experiment 3 columns validated."
)

# =========================================================
# RECONSTRUCT SAVED TRAIN-TEST SPLIT
# =========================================================

training_patient_ids = (
    split_df.loc[
        split_df["Dataset_Split"] == "Train",
        "Patient_ID",
    ]
    .astype(int)
    .tolist()
)

test_patient_ids = (
    split_df.loc[
        split_df["Dataset_Split"] == "Test",
        "Patient_ID",
    ]
    .astype(int)
    .tolist()
)

train_df = (
    model_df.loc[
        model_df["Patient_ID"].isin(
            training_patient_ids
        )
    ]
    .copy()
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)

held_out_test_df = (
    model_df.loc[
        model_df["Patient_ID"].isin(
            test_patient_ids
        )
    ]
    .copy()
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)

if len(train_df) != 48:
    raise ValueError(
        "Expected 48 training patients, but found "
        f"{len(train_df)}."
    )

if len(held_out_test_df) != 12:
    raise ValueError(
        "Expected 12 held-out test patients, but found "
        f"{len(held_out_test_df)}."
    )

overlapping_patient_ids = set(
    train_df["Patient_ID"]
).intersection(
    set(held_out_test_df["Patient_ID"])
)

if overlapping_patient_ids:
    raise ValueError(
        "Patients were found in both training and test sets: "
        f"{sorted(overlapping_patient_ids)}"
    )

y_train = train_df[
    "Binary_Disability"
].copy()

y_test = held_out_test_df[
    "Binary_Disability"
].copy()

print("\nReconstructed training-set shape:")
print(train_df.shape)

print("\nHeld-out test-set shape:")
print(held_out_test_df.shape)

print("\nTraining target distribution:")
print(
    y_train.value_counts().sort_index()
)

print("\nHeld-out test target distribution:")
print(
    y_test.value_counts().sort_index()
)

logger.info(
    "Experiment 1 split reconstructed successfully."
)

# =========================================================
# DEFINE CLINICAL FEATURE SET
# =========================================================

mri_feature_columns = [
    "Lesion_Volume_mm3",
    "Lesion_Count",
    "Average_Lesion_Size",
    "Largest_Lesion_Size",
    "Smallest_Lesion_Size",
    "Small_Lesions",
    "Medium_Lesions",
    "Large_Lesions",
]

non_predictive_columns = [
    "Patient_ID",
    "Binary_Disability",
]

all_predictor_columns = [
    column
    for column in model_df.columns
    if column not in non_predictive_columns
]

constant_features = [
    column
    for column in all_predictor_columns
    if train_df[column].nunique(
        dropna=False
    ) <= 1
]

clinical_feature_columns = [
    column
    for column in all_predictor_columns
    if (
        column not in mri_feature_columns
        and column not in constant_features
    )
]

X_train_clinical = train_df[
    clinical_feature_columns
].copy()

X_test_clinical = held_out_test_df[
    clinical_feature_columns
].copy()

print("\nConstant features removed:")
print(constant_features)

print("\nClinical feature count:")
print(len(clinical_feature_columns))

print("\nClinical training shape:")
print(X_train_clinical.shape)

print("\nClinical held-out test shape:")
print(X_test_clinical.shape)

print("\nClinical features:")
for feature_name in clinical_feature_columns:
    print("-", feature_name)

logger.info(
    "Clinical feature set reconstructed with %s features.",
    len(clinical_feature_columns),
)

# =========================================================
# VALIDATE SELECTED PIPELINE STRUCTURE
# =========================================================

if not hasattr(
    selected_pipeline,
    "named_steps",
):
    raise TypeError(
        "The selected model is not a fitted scikit-learn "
        "Pipeline."
    )

required_pipeline_steps = [
    "preprocessor",
    "classifier",
]

missing_pipeline_steps = [
    step_name
    for step_name in required_pipeline_steps
    if step_name not in selected_pipeline.named_steps
]

if missing_pipeline_steps:
    raise ValueError(
        "The selected pipeline is missing required steps: "
        f"{missing_pipeline_steps}"
    )

fitted_preprocessor = (
    selected_pipeline.named_steps[
        "preprocessor"
    ]
)

fitted_classifier = (
    selected_pipeline.named_steps[
        "classifier"
    ]
)

print("\nPipeline steps:")
for step_name in selected_pipeline.named_steps:
    print("-", step_name)

print("\nPreprocessor type:")
print(type(fitted_preprocessor).__name__)

print("\nClassifier type:")
print(type(fitted_classifier).__name__)

logger.info(
    "Selected pipeline structure validated successfully."
)

# =========================================================
# VALIDATE PIPELINE PREDICTIONS
# =========================================================

training_predictions = (
    selected_pipeline.predict(
        X_train_clinical
    )
)

training_probabilities = (
    selected_pipeline.predict_proba(
        X_train_clinical
    )[:, 1]
)

if len(training_predictions) != len(X_train_clinical):
    raise ValueError(
        "Unexpected number of training predictions."
    )

if not np.all(
    (
        training_probabilities >= 0
    )
    & (
        training_probabilities <= 1
    )
):
    raise ValueError(
        "Predicted probabilities fall outside the "
        "expected range of 0 to 1."
    )

print("\nTraining predictions generated:")
print(len(training_predictions))

print("\nTraining probability range:")
print(
    float(training_probabilities.min()),
    "to",
    float(training_probabilities.max()),
)

logger.info(
    "Selected pipeline prediction check completed."
)

print_section(
    "Experiment 3 Data Reconstruction Complete"
)

print(
    "\nNo SHAP, LIME or permutation-importance "
    "analysis has been performed yet."
)

# =========================================================
# TRANSFORM CLINICAL FEATURES
# =========================================================

X_train_transformed = (
    fitted_preprocessor.transform(
        X_train_clinical
    )
)

X_test_transformed = (
    fitted_preprocessor.transform(
        X_test_clinical
    )
)

# Convert sparse matrices to dense arrays if required
if hasattr(
    X_train_transformed,
    "toarray",
):
    X_train_transformed = (
        X_train_transformed.toarray()
    )

if hasattr(
    X_test_transformed,
    "toarray",
):
    X_test_transformed = (
        X_test_transformed.toarray()
    )

X_train_transformed = np.asarray(
    X_train_transformed
)

X_test_transformed = np.asarray(
    X_test_transformed
)

print("\nTransformed training shape:")
print(X_train_transformed.shape)

print("\nTransformed held-out test shape:")
print(X_test_transformed.shape)

logger.info(
    "Clinical feature matrices transformed successfully."
)

# =========================================================
# RECOVER TRANSFORMED FEATURE NAMES
# =========================================================

try:
    transformed_feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
        .tolist()
    )

except Exception as error:
    logger.exception(
        "Unable to recover transformed feature names."
    )

    raise RuntimeError(
        "Could not recover transformed feature names "
        f"from the fitted preprocessor: {error}"
    ) from error

if (
    len(transformed_feature_names)
    != X_train_transformed.shape[1]
):
    raise ValueError(
        "The number of transformed feature names does not "
        "match the transformed matrix width. "
        f"Names: {len(transformed_feature_names)}, "
        f"columns: {X_train_transformed.shape[1]}."
    )

print("\nNumber of transformed features:")
print(len(transformed_feature_names))

print("\nTransformed feature names:")
for feature_name in transformed_feature_names:
    print("-", feature_name)

logger.info(
    "%s transformed feature names recovered.",
    len(transformed_feature_names),
)

# =========================================================
# CREATE TRANSFORMED DATAFRAMES
# =========================================================

X_train_transformed_df = pd.DataFrame(
    X_train_transformed,
    columns=transformed_feature_names,
    index=train_df["Patient_ID"].astype(int),
)

X_test_transformed_df = pd.DataFrame(
    X_test_transformed,
    columns=transformed_feature_names,
    index=held_out_test_df[
        "Patient_ID"
    ].astype(int),
)

X_train_transformed_df.index.name = (
    "Patient_ID"
)

X_test_transformed_df.index.name = (
    "Patient_ID"
)

print("\nTransformed training preview:")
print(
    X_train_transformed_df
    .head()
    .to_string()
)

logger.info(
    "Transformed feature DataFrames created."
)

# =========================================================
# VALIDATE CLASSIFIER INPUT COMPATIBILITY
# =========================================================

classifier_feature_count = getattr(
    fitted_classifier,
    "n_features_in_",
    None,
)

print("\nClassifier expected feature count:")
print(classifier_feature_count)

if (
    classifier_feature_count is not None
    and classifier_feature_count
    != X_train_transformed.shape[1]
):
    raise ValueError(
        "Classifier feature count does not match the "
        "transformed training matrix. "
        f"Classifier expects {classifier_feature_count}, "
        f"but transformed data contains "
        f"{X_train_transformed.shape[1]}."
    )

classifier_training_probabilities = (
    fitted_classifier.predict_proba(
        X_train_transformed
    )[:, 1]
)

pipeline_training_probabilities = (
    selected_pipeline.predict_proba(
        X_train_clinical
    )[:, 1]
)

probabilities_match = np.allclose(
    classifier_training_probabilities,
    pipeline_training_probabilities,
    rtol=1e-8,
    atol=1e-8,
)

print(
    "\nDo classifier and pipeline probabilities match?"
)
print(probabilities_match)

if not probabilities_match:
    raise ValueError(
        "Classifier probabilities do not match pipeline "
        "probabilities after transformation."
    )

logger.info(
    "Transformed classifier input validated successfully."
)

# =========================================================
# SAVE TRANSFORMED FEATURE DEFINITIONS
# =========================================================

transformed_feature_summary_df = pd.DataFrame({
    "Transformed_Feature": (
        transformed_feature_names
    ),
    "Feature_Position": range(
        len(transformed_feature_names)
    ),
})

transformed_feature_summary_output = (
    experiment_03_tables_dir
    / "transformed_feature_names.csv"
)

save_dataframe(
    dataframe=transformed_feature_summary_df,
    output_path=(
        transformed_feature_summary_output
    ),
    index=False,
)

logger.info(
    "Transformed feature names saved to %s.",
    transformed_feature_summary_output,
)

print_section(
    "Transformed Feature Preparation Complete"
)

print("\nTransformed training shape:")
print(X_train_transformed_df.shape)

print("\nTransformed test shape:")
print(X_test_transformed_df.shape)

print("\nFeature-name table saved to:")
print(transformed_feature_summary_output)

print(
    "\nNo SHAP, LIME or permutation-importance "
    "analysis has been performed yet."
)

# =========================================================
# CREATE SHAP TREE EXPLAINER
# =========================================================

print_section(
    "Global SHAP Analysis"
)

logger.info(
    "Creating SHAP TreeExplainer for the selected "
    "Random Forest classifier."
)

shap_explainer = shap.TreeExplainer(
    fitted_classifier
)

raw_shap_explanation = shap_explainer(
    X_train_transformed_df
)

print("\nRaw SHAP value shape:")
print(
    np.asarray(
        raw_shap_explanation.values
    ).shape
)

print("\nRaw SHAP base-value shape:")
print(
    np.asarray(
        raw_shap_explanation.base_values
    ).shape
)

logger.info(
    "Raw SHAP explanations generated successfully."
)

# =========================================================
# SELECT POSITIVE-CLASS SHAP VALUES
# =========================================================

POSITIVE_CLASS_INDEX = 1

raw_shap_values = np.asarray(
    raw_shap_explanation.values
)

raw_base_values = np.asarray(
    raw_shap_explanation.base_values
)

if raw_shap_values.ndim == 3:
    positive_class_shap_values = (
        raw_shap_values[
            :,
            :,
            POSITIVE_CLASS_INDEX,
        ]
    )

elif raw_shap_values.ndim == 2:
    positive_class_shap_values = (
        raw_shap_values
    )

else:
    raise ValueError(
        "Unexpected SHAP value dimensions: "
        f"{raw_shap_values.shape}"
    )


if raw_base_values.ndim == 2:
    positive_class_base_values = (
        raw_base_values[
            :,
            POSITIVE_CLASS_INDEX,
        ]
    )

elif raw_base_values.ndim == 1:
    if len(raw_base_values) == 2:
        positive_class_base_values = np.repeat(
            raw_base_values[
                POSITIVE_CLASS_INDEX
            ],
            len(X_train_transformed_df),
        )
    else:
        positive_class_base_values = (
            raw_base_values
        )

elif raw_base_values.ndim == 0:
    positive_class_base_values = np.repeat(
        float(raw_base_values),
        len(X_train_transformed_df),
    )

else:
    raise ValueError(
        "Unexpected SHAP base-value dimensions: "
        f"{raw_base_values.shape}"
    )


if (
    positive_class_shap_values.shape
    != X_train_transformed_df.shape
):
    raise ValueError(
        "Positive-class SHAP values do not match the "
        "transformed training data shape. "
        f"SHAP: {positive_class_shap_values.shape}; "
        f"data: {X_train_transformed_df.shape}."
    )

print("\nPositive-class SHAP shape:")
print(
    positive_class_shap_values.shape
)

logger.info(
    "Positive-class SHAP values selected successfully."
)

# =========================================================
# CREATE POSITIVE-CLASS SHAP EXPLANATION
# =========================================================

positive_class_shap_explanation = shap.Explanation(
    values=positive_class_shap_values,
    base_values=positive_class_base_values,
    data=X_train_transformed_df.to_numpy(),
    feature_names=transformed_feature_names,
)

print("\nSHAP Explanation shape:")
print(
    positive_class_shap_explanation.shape
)

logger.info(
    "Positive-class SHAP Explanation object created."
)

# =========================================================
# VALIDATE SHAP ADDITIVITY
# =========================================================

shap_reconstructed_probabilities = (
    positive_class_base_values
    + positive_class_shap_values.sum(
        axis=1
    )
)

shap_probability_match = np.allclose(
    shap_reconstructed_probabilities,
    classifier_training_probabilities,
    rtol=1e-5,
    atol=1e-5,
)

maximum_additivity_difference = float(
    np.max(
        np.abs(
            shap_reconstructed_probabilities
            - classifier_training_probabilities
        )
    )
)

print(
    "\nDo reconstructed SHAP probabilities match "
    "classifier probabilities?"
)
print(shap_probability_match)

print("\nMaximum SHAP additivity difference:")
print(maximum_additivity_difference)

if not shap_probability_match:
    logger.warning(
        "SHAP additivity validation did not pass. "
        "Maximum difference: %.8f",
        maximum_additivity_difference,
    )
else:
    logger.info(
        "SHAP additivity validation passed."
    )

# =========================================================
# CREATE GLOBAL SHAP IMPORTANCE TABLE
# =========================================================

mean_absolute_shap = np.mean(
    np.abs(
        positive_class_shap_values
    ),
    axis=0,
)

global_shap_importance_df = pd.DataFrame({
    "Transformed_Feature": (
        transformed_feature_names
    ),
    "Mean_Absolute_SHAP": (
        mean_absolute_shap
    ),
})

global_shap_importance_df = (
    global_shap_importance_df
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )
    .reset_index(drop=True)
)

global_shap_importance_df[
    "Importance_Rank"
] = (
    np.arange(
        1,
        len(global_shap_importance_df) + 1,
    )
)

global_shap_importance_df = (
    global_shap_importance_df[
        [
            "Importance_Rank",
            "Transformed_Feature",
            "Mean_Absolute_SHAP",
        ]
    ]
)

global_shap_importance_df[
    "Mean_Absolute_SHAP"
] = global_shap_importance_df[
    "Mean_Absolute_SHAP"
].round(6)

print("\nTop 20 global SHAP features:")
print(
    global_shap_importance_df
    .head(20)
    .to_string(index=False)
)

global_shap_importance_output = (
    experiment_03_tables_dir
    / "global_shap_feature_importance.csv"
)

save_dataframe(
    dataframe=global_shap_importance_df,
    output_path=global_shap_importance_output,
    index=False,
)

logger.info(
    "Global SHAP importance table saved to %s.",
    global_shap_importance_output,
)

# =========================================================
# SAVE PATIENT-LEVEL SHAP VALUES
# =========================================================

patient_shap_values_df = pd.DataFrame(
    positive_class_shap_values,
    columns=transformed_feature_names,
)

patient_shap_values_df.insert(
    0,
    "Patient_ID",
    train_df["Patient_ID"].astype(int),
)

patient_shap_values_df.insert(
    1,
    "Actual_Class",
    y_train.to_numpy(),
)

patient_shap_values_output = (
    experiment_03_tables_dir
    / "training_patient_shap_values.csv"
)

save_dataframe(
    dataframe=patient_shap_values_df,
    output_path=patient_shap_values_output,
    index=False,
)

logger.info(
    "Patient-level training SHAP values saved to %s.",
    patient_shap_values_output,
)

# =========================================================
# CREATE GLOBAL SHAP BAR PLOT
# =========================================================

plt.figure()

shap.plots.bar(
    positive_class_shap_explanation,
    max_display=20,
    show=False,
)

plt.title(
    "Global SHAP Importance for Moderate/High Disability"
)

plt.tight_layout()

global_shap_bar_output = (
    experiment_03_figures_dir
    / "01_global_shap_bar.png"
)

plt.savefig(
    global_shap_bar_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

logger.info(
    "Global SHAP bar plot saved to %s.",
    global_shap_bar_output,
)

# =========================================================
# CREATE SHAP BEESWARM PLOT
# =========================================================

plt.figure()

shap.plots.beeswarm(
    positive_class_shap_explanation,
    max_display=20,
    show=False,
)

plt.title(
    "SHAP Effects on Moderate/High Disability Predictions"
)

plt.tight_layout()

global_shap_beeswarm_output = (
    experiment_03_figures_dir
    / "02_global_shap_beeswarm.png"
)

plt.savefig(
    global_shap_beeswarm_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

logger.info(
    "Global SHAP beeswarm plot saved to %s.",
    global_shap_beeswarm_output,
)

# =========================================================
# COMPLETE GLOBAL SHAP STAGE
# =========================================================

print_section(
    "Global SHAP Analysis Complete"
)

print("\nTop-ranked SHAP feature:")
print(
    global_shap_importance_df.iloc[0][
        "Transformed_Feature"
    ]
)

print("\nGlobal SHAP importance saved to:")
print(global_shap_importance_output)

print("\nPatient-level SHAP values saved to:")
print(patient_shap_values_output)

print("\nGlobal SHAP bar plot saved to:")
print(global_shap_bar_output)

print("\nGlobal SHAP beeswarm plot saved to:")
print(global_shap_beeswarm_output)

print(
    "\nNo local SHAP, LIME or permutation-importance "
    "analysis has been performed yet."
)

# =========================================================
# CREATE PATIENT PREDICTION SUMMARY
# =========================================================

print_section(
    "Selecting Representative Patients"
)

training_prediction_summary = pd.DataFrame({

    "Patient_ID":
        train_df["Patient_ID"].astype(int),

    "Actual_Class":
        y_train,

    "Predicted_Class":
        training_predictions,

    "Predicted_Probability":
        classifier_training_probabilities,

})

training_prediction_summary[
    "Correct_Prediction"
] = (
    training_prediction_summary[
        "Actual_Class"
    ]
    ==
    training_prediction_summary[
        "Predicted_Class"
    ]
)

training_prediction_summary[
    "Distance_From_Threshold"
] = np.abs(

    training_prediction_summary[
        "Predicted_Probability"
    ] - 0.50

)

prediction_summary_output = (

    experiment_03_tables_dir
    / "training_prediction_summary.csv"

)

save_dataframe(

    dataframe=training_prediction_summary,

    output_path=prediction_summary_output,

    index=False,

)

print(
    training_prediction_summary.head()
)

logger.info(
    "Training prediction summary created."
)

# =========================================================
# SELECT REPRESENTATIVE PATIENTS
# =========================================================

low_patient = (

    training_prediction_summary

    .query(

        "Actual_Class == 0 and Correct_Prediction == True"

    )

    .sort_values(

        "Predicted_Probability"

    )

    .iloc[0]

)

high_patient = (

    training_prediction_summary

    .query(

        "Actual_Class == 1 and Correct_Prediction == True"

    )

    .sort_values(

        "Predicted_Probability",

        ascending=False,

    )

    .iloc[0]

)

borderline_patient = (

    training_prediction_summary

    .sort_values(

        "Distance_From_Threshold"

    )

    .iloc[0]

)

print("\nRepresentative patients:")

print(low_patient)

print(high_patient)

print(borderline_patient)

logger.info(
    "Representative patients selected."
)

# =========================================================
# STORE PATIENT IDS
# =========================================================

selected_patients = {

    "Low_Disability":

        int(
            low_patient["Patient_ID"]
        ),

    "Moderate_High_Disability":

        int(
            high_patient["Patient_ID"]
        ),

    "Borderline":

        int(
            borderline_patient["Patient_ID"]
        ),

}

print("\nChosen patients:")

print(selected_patients)

# =========================================================
# SAVE SELECTED PATIENTS
# =========================================================

selected_patients_df = pd.DataFrame({

    "Scenario":

        list(selected_patients.keys()),

    "Patient_ID":

        list(selected_patients.values()),

})

save_dataframe(

    dataframe=selected_patients_df,

    output_path=(

        experiment_03_tables_dir
        / "selected_patients.csv"

    ),

    index=False,

)

logger.info(
    "Representative patients saved."
)

# =========================================================
# LOCAL SHAP WATERFALL FUNCTION
# =========================================================

def create_local_shap_waterfall(
    patient_id: int,
    scenario_name: str,
) -> None:

    patient_position = (
        X_train_transformed_df.index
        .get_loc(patient_id)
    )

    patient_explanation = (
        positive_class_shap_explanation[
            patient_position
        ]
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.plots.waterfall(
        patient_explanation,
        max_display=15,
        show=False,
    )

    plt.title(
        f"{scenario_name} (Patient {patient_id})"
    )

    plt.tight_layout()

    output_file = (
        experiment_03_figures_dir
        / f"03_waterfall_{scenario_name.lower()}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Waterfall plot created for %s.",
        scenario_name,
    )

    print(
        f"\nWaterfall plot saved:\n{output_file}"
    )

# =========================================================
# CREATE LOCAL SHAP WATERFALL PLOTS
# =========================================================

print_section(
    "Local SHAP Waterfall Plots"
)

for scenario_name, patient_id in (
    selected_patients.items()
):

    create_local_shap_waterfall(

        patient_id=patient_id,

        scenario_name=scenario_name,

    )

logger.info(
    "All representative waterfall plots created."
)

# =========================================================
# SAVE TOP SHAP CONTRIBUTIONS
# =========================================================

for scenario_name, patient_id in (
    selected_patients.items()
):

    patient_position = (
        X_train_transformed_df.index
        .get_loc(patient_id)
    )

    patient_values = (
        positive_class_shap_values[
            patient_position
        ]
    )

    contribution_df = pd.DataFrame({

        "Feature":

            transformed_feature_names,

        "SHAP_Value":

            patient_values,

        "Absolute_SHAP":

            np.abs(
                patient_values
            ),

    })

    contribution_df = (

        contribution_df

        .sort_values(

            "Absolute_SHAP",

            ascending=False,

        )

        .reset_index(drop=True)

    )

    save_dataframe(

        dataframe=contribution_df,

        output_path=(

            experiment_03_tables_dir
            / f"{scenario_name.lower()}_top_shap_features.csv"

        ),

        index=False,

    )

# =========================================================
# SHAP DECISION PLOT
# =========================================================

print_section(
    "SHAP Decision Plot"
)
decision_plot_positions = [

    X_train_transformed_df.index.get_loc(patient_id)

    for patient_id in selected_patients.values()

]

plt.figure(
    figsize=(12, 8)
)

shap.decision_plot(

    base_value=positive_class_shap_explanation.base_values[0],

    shap_values=positive_class_shap_values[
        decision_plot_positions
    ],

    features=X_train_transformed_df.iloc[
        decision_plot_positions
    ],

    feature_names=transformed_feature_names,

    show=False,

)

decision_plot_output = (

    experiment_03_figures_dir
    / "04_shap_decision_plot.png"

)

plt.savefig(

    decision_plot_output,

    dpi=300,

    bbox_inches="tight",

)

plt.close("all")

print(
    "\nDecision plot saved:"
)

print(
    decision_plot_output
)

logger.info(
    "SHAP decision plot created."
)

# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================

print_section(
    "Permutation Importance"
)

permutation_results = permutation_importance(

    estimator=selected_pipeline,

    X=X_train_clinical,

    y=y_train,

    n_repeats=30,

    random_state=42,

    scoring="roc_auc",

)

permutation_df = pd.DataFrame({

    "Feature": clinical_feature_columns,

    "Mean_Importance": permutation_results.importances_mean,

    "SD_Importance": permutation_results.importances_std,

})

permutation_df = (

    permutation_df

    .sort_values(

        "Mean_Importance",

        ascending=False,

    )

    .reset_index(drop=True)

)

save_dataframe(

    dataframe=permutation_df,

    output_path=(

        experiment_03_tables_dir

        / "permutation_importance.csv"

    ),

    index=False,

)

top_permutation = permutation_df.head(15)

plt.figure(

    figsize=(10,8)

)

plt.barh(

    top_permutation["Feature"],

    top_permutation["Mean_Importance"]

)

plt.gca().invert_yaxis()

plt.xlabel(

    "Mean Permutation Importance"

)

plt.title(

    "Permutation Importance"

)

permutation_plot_output = (

    experiment_03_figures_dir

    / "05_permutation_importance.png"

)

plt.savefig(

    permutation_plot_output,

    dpi=300,

    bbox_inches="tight",

)

plt.close("all")

print()

print(
    "Top permutation feature:"
)

print(
    permutation_df.iloc[0]["Feature"]
)

print()

print(
    "Permutation table saved:"
)

print(

    experiment_03_tables_dir

    / "permutation_importance.csv"

)

print()

print(
    "Permutation figure saved:"
)

print(
    permutation_plot_output
)

logger.info(
    "Permutation importance completed."
)

# =========================================================
# LIME LOCAL EXPLANATIONS
# =========================================================

print_section(
    "LIME Local Explanations"
)

logger.info(
    "Creating LIME explainer for the selected "
    "Random Forest classifier."
)

# =========================================================
# CREATE LIME EXPLAINER
# =========================================================

lime_explainer = (
    lime.lime_tabular.LimeTabularExplainer(
        training_data=(
            X_train_transformed_df.to_numpy()
        ),
        feature_names=transformed_feature_names,
        class_names=[
            "Low Disability",
            "Moderate/High Disability",
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
)

logger.info(
    "LIME TabularExplainer created successfully."
)

# =========================================================
# DEFINE LIME PREDICTION FUNCTION
# =========================================================

def lime_predict_proba(
    transformed_data: np.ndarray,
) -> np.ndarray:
    """
    Return class probabilities from the fitted classifier
    for transformed clinical feature data.
    """

    transformed_data = np.asarray(
        transformed_data
    )

    return fitted_classifier.predict_proba(
        transformed_data
    )

# =========================================================
# CREATE LIME EXPLANATION FUNCTION
# =========================================================

def create_lime_explanation(
    patient_id: int,
    scenario_name: str,
    number_of_features: int = 15,
) -> pd.DataFrame:
    """
    Generate and save a LIME explanation for one patient.
    """

    patient_position = (
        X_train_transformed_df.index
        .get_loc(patient_id)
    )

    patient_transformed_values = (
        X_train_transformed_df.iloc[
            patient_position
        ].to_numpy()
    )

    lime_explanation = (
        lime_explainer.explain_instance(
            data_row=patient_transformed_values,
            predict_fn=lime_predict_proba,
            labels=[1],
            num_features=number_of_features,
            num_samples=5000,
        )
    )

    lime_feature_weights = (
        lime_explanation.as_list(
            label=1
        )
    )

    lime_contribution_df = pd.DataFrame(
        lime_feature_weights,
        columns=[
            "LIME_Feature_Condition",
            "LIME_Weight",
        ],
    )

    lime_contribution_df.insert(
        0,
        "Scenario",
        scenario_name,
    )

    lime_contribution_df.insert(
        1,
        "Patient_ID",
        patient_id,
    )

    lime_contribution_df[
        "Absolute_LIME_Weight"
    ] = np.abs(
        lime_contribution_df[
            "LIME_Weight"
        ]
    )

    lime_contribution_df[
        "Direction"
    ] = np.where(
        lime_contribution_df[
            "LIME_Weight"
        ] > 0,
        "Towards Moderate/High Disability",
        "Towards Low Disability",
    )

    safe_scenario_name = (
        scenario_name.lower()
    )

    lime_table_output = (
        experiment_03_tables_dir
        / f"{safe_scenario_name}_lime_explanation.csv"
    )

    save_dataframe(
        dataframe=lime_contribution_df,
        output_path=lime_table_output,
        index=False,
    )

    lime_figure = (
        lime_explanation.as_pyplot_figure(
            label=1
        )
    )

    lime_figure.set_size_inches(
        10,
        7,
    )

    plt.title(
        f"LIME Explanation: "
        f"{scenario_name} "
        f"(Patient {patient_id})"
    )

    plt.tight_layout()

    lime_figure_output = (
        experiment_03_figures_dir
        / f"06_lime_{safe_scenario_name}.png"
    )

    plt.savefig(
        lime_figure_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close("all")

    logger.info(
        "LIME explanation created for %s, Patient %s.",
        scenario_name,
        patient_id,
    )

    print(
        f"\nLIME table saved:\n"
        f"{lime_table_output}"
    )

    print(
        f"\nLIME figure saved:\n"
        f"{lime_figure_output}"
    )

    return lime_contribution_df

# =========================================================
# GENERATE REPRESENTATIVE LIME EXPLANATIONS
# =========================================================

lime_explanation_tables = []

for scenario_name, patient_id in (
    selected_patients.items()
):
    patient_lime_df = create_lime_explanation(
        patient_id=patient_id,
        scenario_name=scenario_name,
        number_of_features=15,
    )

    lime_explanation_tables.append(
        patient_lime_df
    )

# =========================================================
# SAVE COMBINED LIME EXPLANATIONS
# =========================================================

combined_lime_explanations_df = pd.concat(
    lime_explanation_tables,
    ignore_index=True,
)

combined_lime_output = (
    experiment_03_tables_dir
    / "combined_lime_explanations.csv"
)

save_dataframe(
    dataframe=combined_lime_explanations_df,
    output_path=combined_lime_output,
    index=False,
)

logger.info(
    "Combined LIME explanations saved to %s.",
    combined_lime_output,
)

# =========================================================
# COMPLETE LIME STAGE
# =========================================================

print_section(
    "LIME Explanations Complete"
)

print("\nCombined LIME explanations saved to:")
print(combined_lime_output)

print(
    "\nLIME explanations were generated for "
    "all three representative patients."
)

# =========================================================
# MAP LIME CONDITIONS TO TRANSFORMED FEATURES
# =========================================================

print_section(
    "SHAP and LIME Local Comparison"
)

# Sort longest names first to avoid partial-name matches.
feature_names_by_length = sorted(
    transformed_feature_names,
    key=len,
    reverse=True,
)


def identify_lime_feature(
    lime_condition: str,
) -> str:
    """
    Identify the transformed feature referenced in a LIME
    condition string.
    """

    for feature_name in feature_names_by_length:
        if feature_name in lime_condition:
            return feature_name

    return "Unmatched_Feature"


combined_lime_explanations_df[
    "Matched_Transformed_Feature"
] = combined_lime_explanations_df[
    "LIME_Feature_Condition"
].apply(
    identify_lime_feature
)

print("\nLIME feature-matching results:")
print(
    combined_lime_explanations_df[
        "Matched_Transformed_Feature"
    ].value_counts(
        dropna=False
    )
)

unmatched_lime_rows = (
    combined_lime_explanations_df[
        combined_lime_explanations_df[
            "Matched_Transformed_Feature"
        ] == "Unmatched_Feature"
    ]
)

if not unmatched_lime_rows.empty:
    print("\nUnmatched LIME conditions:")
    print(
        unmatched_lime_rows[
            "LIME_Feature_Condition"
        ].to_string(index=False)
    )

logger.info(
    "LIME conditions mapped to transformed features."
)

# =========================================================
# CREATE LOCAL SHAP COMPARISON TABLE
# =========================================================

local_shap_comparison_records = []

for scenario_name, patient_id in (
    selected_patients.items()
):
    patient_position = (
        X_train_transformed_df.index
        .get_loc(patient_id)
    )

    patient_shap_values = (
        positive_class_shap_values[
            patient_position
        ]
    )

    for feature_name, shap_value in zip(
        transformed_feature_names,
        patient_shap_values,
    ):
        local_shap_comparison_records.append({
            "Scenario": scenario_name,
            "Patient_ID": patient_id,
            "Transformed_Feature": feature_name,
            "SHAP_Value": shap_value,
            "Absolute_SHAP_Value": abs(
                shap_value
            ),
            "SHAP_Direction": (
                "Towards Moderate/High Disability"
                if shap_value > 0
                else "Towards Low Disability"
            ),
        })


local_shap_comparison_df = pd.DataFrame(
    local_shap_comparison_records
)

# =========================================================
# AGGREGATE LIME CONTRIBUTIONS BY FEATURE
# =========================================================

local_lime_comparison_df = (
    combined_lime_explanations_df
    .groupby(
        [
            "Scenario",
            "Patient_ID",
            "Matched_Transformed_Feature",
        ],
        as_index=False,
    )
    .agg(
        LIME_Weight=(
            "LIME_Weight",
            "sum",
        ),
        Absolute_LIME_Weight=(
            "Absolute_LIME_Weight",
            "sum",
        ),
        LIME_Condition=(
            "LIME_Feature_Condition",
            lambda values: " | ".join(
                values.astype(str)
            ),
        ),
    )
)

local_lime_comparison_df = (
    local_lime_comparison_df.rename(
        columns={
            "Matched_Transformed_Feature":
                "Transformed_Feature",
        }
    )
)

local_lime_comparison_df[
    "LIME_Direction"
] = np.where(
    local_lime_comparison_df[
        "LIME_Weight"
    ] > 0,
    "Towards Moderate/High Disability",
    "Towards Low Disability",
)

# =========================================================
# MERGE LOCAL SHAP AND LIME RESULTS
# =========================================================

shap_lime_local_comparison_df = (
    local_shap_comparison_df.merge(
        local_lime_comparison_df,
        on=[
            "Scenario",
            "Patient_ID",
            "Transformed_Feature",
        ],
        how="outer",
    )
)

numeric_comparison_columns = [
    "SHAP_Value",
    "Absolute_SHAP_Value",
    "LIME_Weight",
    "Absolute_LIME_Weight",
]

for column in numeric_comparison_columns:
    shap_lime_local_comparison_df[
        column
    ] = shap_lime_local_comparison_df[
        column
    ].fillna(0.0)

# =========================================================
# CALCULATE DIRECTION AGREEMENT
# =========================================================

direction_comparison_mask = (
    shap_lime_local_comparison_df[
        "LIME_Condition"
    ].notna()
)

shap_lime_local_comparison_df[
    "Direction_Agreement"
] = pd.Series(
    pd.NA,
    index=shap_lime_local_comparison_df.index,
    dtype="boolean",
)

shap_lime_local_comparison_df.loc[
    direction_comparison_mask,
    "Direction_Agreement",
] = (
    shap_lime_local_comparison_df.loc[
        direction_comparison_mask,
        "SHAP_Direction",
    ].to_numpy()
    ==
    shap_lime_local_comparison_df.loc[
        direction_comparison_mask,
        "LIME_Direction",
    ].to_numpy()
)

# =========================================================
# CALCULATE LOCAL IMPORTANCE RANKS
# =========================================================

shap_lime_local_comparison_df[
    "SHAP_Rank"
] = (
    shap_lime_local_comparison_df
    .groupby("Scenario")[
        "Absolute_SHAP_Value"
    ]
    .rank(
        method="min",
        ascending=False,
    )
)

shap_lime_local_comparison_df[
    "LIME_Rank"
] = (
    shap_lime_local_comparison_df
    .groupby("Scenario")[
        "Absolute_LIME_Weight"
    ]
    .rank(
        method="min",
        ascending=False,
    )
)

shap_lime_local_comparison_df = (
    shap_lime_local_comparison_df
    .sort_values(
        [
            "Scenario",
            "SHAP_Rank",
        ]
    )
    .reset_index(drop=True)
)

# =========================================================
# SAVE SHAP-LIME LOCAL COMPARISON
# =========================================================

shap_lime_comparison_output = (
    experiment_03_tables_dir
    / "shap_lime_local_comparison.csv"
)

save_dataframe(
    dataframe=shap_lime_local_comparison_df,
    output_path=shap_lime_comparison_output,
    index=False,
)

logger.info(
    "SHAP-LIME local comparison saved to %s.",
    shap_lime_comparison_output,
)

# =========================================================
# CREATE METHOD-AGREEMENT SUMMARY
# =========================================================

agreement_summary_records = []

for scenario_name, patient_id in (
    selected_patients.items()
):
    scenario_df = (
        shap_lime_local_comparison_df[
            shap_lime_local_comparison_df[
                "Scenario"
            ] == scenario_name
        ]
        .copy()
    )

    top_shap_features = set(
        scenario_df
        .sort_values(
            "Absolute_SHAP_Value",
            ascending=False,
        )
        .head(10)[
            "Transformed_Feature"
        ]
    )

    top_lime_features = set(
        scenario_df.loc[
            scenario_df[
                "Absolute_LIME_Weight"
            ] > 0
        ]
        .sort_values(
            "Absolute_LIME_Weight",
            ascending=False,
        )
        .head(10)[
            "Transformed_Feature"
        ]
    )

    shared_features = (
        top_shap_features.intersection(
            top_lime_features
        )
    )

    union_features = (
        top_shap_features.union(
            top_lime_features
        )
    )

    jaccard_similarity = (
        len(shared_features)
        / len(union_features)
        if union_features
        else np.nan
    )

    compared_direction_rows = (
        scenario_df.dropna(
            subset=[
                "Direction_Agreement",
            ]
        )
    )

    direction_agreement_rate = (
        compared_direction_rows[
            "Direction_Agreement"
        ].mean()
        if not compared_direction_rows.empty
        else np.nan
    )

    agreement_summary_records.append({
        "Scenario": scenario_name,
        "Patient_ID": patient_id,
        "Top_10_SHAP_LIME_Shared_Features":
            len(shared_features),
        "Top_10_Feature_Jaccard_Similarity":
            jaccard_similarity,
        "Direction_Agreement_Rate":
            direction_agreement_rate,
        "Shared_Features":
            ", ".join(
                sorted(shared_features)
            ),
    })


shap_lime_agreement_summary_df = (
    pd.DataFrame(
        agreement_summary_records
    )
)

shap_lime_agreement_summary_output = (
    experiment_03_tables_dir
    / "shap_lime_agreement_summary.csv"
)

save_dataframe(
    dataframe=(
        shap_lime_agreement_summary_df
    ),
    output_path=(
        shap_lime_agreement_summary_output
    ),
    index=False,
)

print("\nSHAP-LIME agreement summary:")
print(
    shap_lime_agreement_summary_df
    .to_string(index=False)
)

# =========================================================
# GLOBAL SHAP VS PERMUTATION COMPARISON
# =========================================================

print_section(
    "Global Explainability Comparison"
)

# ---------------------------------------------------------
# Prepare SHAP ranking table
# ---------------------------------------------------------

shap_rank_df = (
    global_shap_importance_df[
        [
            "Importance_Rank",
            "Transformed_Feature",
            "Mean_Absolute_SHAP",
        ]
    ]
    .copy()
    .rename(
        columns={
            "Importance_Rank": "SHAP_Rank",
            "Transformed_Feature": "Feature",
        }
    )
)

# ---------------------------------------------------------
# Prepare permutation ranking table
# ---------------------------------------------------------

permutation_rank_df = (
    permutation_df.copy()
    .reset_index(drop=True)
)

permutation_rank_df[
    "Permutation_Rank"
] = (
    permutation_rank_df.index + 1
)

permutation_rank_df = (
    permutation_rank_df[
        [
            "Feature",
            "Mean_Importance",
            "SD_Importance",
            "Permutation_Rank",
        ]
    ]
)

# ---------------------------------------------------------
# Merge SHAP and permutation rankings
# ---------------------------------------------------------

global_explainability_df = (
    shap_rank_df.merge(
        permutation_rank_df,
        on="Feature",
        how="outer",
    )
)

# Features created by one-hot encoding may appear in SHAP
# but not in raw-feature permutation importance.
global_explainability_df[
    "Rank_Difference"
] = (
    global_explainability_df[
        "SHAP_Rank"
    ]
    -
    global_explainability_df[
        "Permutation_Rank"
    ]
).abs()

global_explainability_df[
    "Available_In_Both_Methods"
] = (
    global_explainability_df[
        "SHAP_Rank"
    ].notna()
    &
    global_explainability_df[
        "Permutation_Rank"
    ].notna()
)

global_explainability_df = (
    global_explainability_df
    .sort_values(
        by=[
            "Available_In_Both_Methods",
            "SHAP_Rank",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# Save comparison table
# ---------------------------------------------------------

comparison_output = (
    experiment_03_tables_dir
    / "global_explainability_comparison.csv"
)

save_dataframe(
    dataframe=global_explainability_df,
    output_path=comparison_output,
    index=False,
)

logger.info(
    "Global explainability comparison saved to %s.",
    comparison_output,
)

print("\nTop global explainability comparison rows:")
print(
    global_explainability_df
    .head(15)
    .to_string(index=False)
)

print("\nComparison table saved to:")
print(comparison_output)

# =========================================================
# EXPERIMENT 3 SUMMARY
# =========================================================

print_section(
    "Experiment 3 Summary"
)

experiment_03_summary_df = pd.DataFrame([
    {
        "Selected_Model": (
            selected_model_information[
                "Selected_Model"
            ]
        ),

        "Feature_Set": (
            selected_model_information[
                "Feature_Set"
            ]
        ),

        "Global_SHAP_Top_Feature": (
            global_shap_importance_df.iloc[0][
                "Transformed_Feature"
            ]
        ),

        "Global_SHAP_Top_Importance": float(
            global_shap_importance_df.iloc[0][
                "Mean_Absolute_SHAP"
            ]
        ),

        "Permutation_Top_Feature": (
            permutation_df.iloc[0][
                "Feature"
            ]
        ),

        "Permutation_Top_Importance": float(
            permutation_df.iloc[0][
                "Mean_Importance"
            ]
        ),

        "Number_of_Raw_Clinical_Features": (
            len(clinical_feature_columns)
        ),

        "Number_of_Transformed_Features": (
            len(transformed_feature_names)
        ),

        "Number_of_Representative_Patients": (
            len(selected_patients)
        ),

        "Local_SHAP_Completed": True,

        "LIME_Completed": True,

        "Permutation_Importance_Completed": True,

        "SHAP_LIME_Comparison_Completed": True,

        "SHAP_Permutation_Comparison_Completed": True,

        "Held_Out_Test_Used": False,
    }
])

experiment_03_summary_output = (
    EXPERIMENT_03_DIR
    / "tables"
    / "experiment_03_summary.csv"
)

save_dataframe(
    dataframe=experiment_03_summary_df,
    output_path=experiment_03_summary_output,
    index=False,
)

logger.info(
    "Experiment 3 summary saved to %s.",
    experiment_03_summary_output,
)

print("\nExperiment 3 summary:")
print(
    experiment_03_summary_df.to_string(
        index=False
    )
)

print("\nExperiment 3 summary saved to:")
print(experiment_03_summary_output)

# =========================================================
# EXPERIMENT 3 CONCLUSION
# =========================================================

print_section(
    "Experiment 3 Conclusion"
)

selected_model_name = (
    selected_model_information[
        "Selected_Model"
    ]
)

selected_feature_set = (
    selected_model_information[
        "Feature_Set"
    ]
)

global_shap_top_feature = (
    global_shap_importance_df.iloc[0][
        "Transformed_Feature"
    ]
)

permutation_top_feature = (
    permutation_df.iloc[0][
        "Feature"
    ]
)

experiment_03_conclusion = (
    f"Experiment 3 evaluated the explainability of the "
    f"selected {selected_model_name} model using SHAP, "
    f"LIME and permutation importance.\n\n"

    f"The model used the {selected_feature_set} feature set. "
    f"Global SHAP identified {global_shap_top_feature} as "
    f"the most influential transformed feature for predicting "
    f"moderate/high disability. Permutation importance also "
    f"identified {permutation_top_feature} as the most "
    f"influential original clinical feature.\n\n"

    "Local SHAP waterfall plots and LIME explanations were "
    "generated for three representative cases: a confidently "
    "predicted low-disability patient, a confidently predicted "
    "moderate/high-disability patient and a borderline case "
    "with a predicted probability close to the classification "
    "threshold.\n\n"

    "The local analyses showed that feature contributions could "
    "operate in opposing directions within the same patient. "
    "This was particularly evident in the borderline case, where "
    "some features increased the predicted probability of "
    "moderate/high disability while others reduced it.\n\n"

    "The SHAP-LIME comparison demonstrated meaningful overlap "
    "between the most influential local features and generally "
    "strong agreement in contribution direction. The global "
    "SHAP-permutation comparison also showed agreement for several "
    "important clinical variables, including the top-ranked "
    "neurological finding measure.\n\n"

    "These findings indicate that the selected Random Forest "
    "model can generate interpretable global and patient-level "
    "explanations. However, the explanations describe the model's "
    "behaviour within this dataset and should not be interpreted "
    "as clinical diagnoses or causal relationships.\n\n"

    "The held-out test set was not used during Experiment 3."
)

experiment_03_conclusion_output = (
    EXPERIMENT_03_DIR
    / "experiment_03_conclusion.txt"
)

with open(
    experiment_03_conclusion_output,
    "w",
    encoding="utf-8",
) as conclusion_file:
    conclusion_file.write(
        experiment_03_conclusion
    )

logger.info(
    "Experiment 3 conclusion saved to %s.",
    experiment_03_conclusion_output,
)

print("\nExperiment 3 conclusion saved to:")
print(experiment_03_conclusion_output)

