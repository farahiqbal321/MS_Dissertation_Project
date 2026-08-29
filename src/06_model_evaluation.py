"""
=========================================================
06_model_evaluation.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Runs Experiment 2: model refinement and robustness
evaluation.

Four candidate model and modality combinations selected
from Experiment 1 are evaluated using repeated stratified
cross-validation before hyperparameter optimisation.

The previously defined held-out test set remains excluded
from model development.

Outputs
-------
Repeated cross-validation and tuning outputs will be saved
under experiment_02_model_refinement.

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)

import joblib

from time import perf_counter

from scipy.stats import (
    randint,
    loguniform,
    uniform,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from config import (
    PROCESSED_DATA_DIR,
    EXPERIMENT_01_DIR,
    EXPERIMENT_02_DIR,
    LOGS_DIR,
    RANDOM_STATE,
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
    logger_name="model_refinement",
    log_file=LOGS_DIR / "model_refinement.log",
)

experiment_02_tables_dir = (
    EXPERIMENT_02_DIR / "tables"
)

experiment_02_models_dir = (
    EXPERIMENT_02_DIR / "models"
)

experiment_02_figures_dir = (
    EXPERIMENT_02_DIR / "figures"
)

experiment_02_logs_dir = (
    EXPERIMENT_02_DIR / "logs"
)

for directory in [
    EXPERIMENT_02_DIR,
    experiment_02_tables_dir,
    experiment_02_models_dir,
    experiment_02_figures_dir,
    experiment_02_logs_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# =========================================================
# START EXPERIMENT 2
# =========================================================

print_section(
    "Experiment 2: Model Refinement"
)

logger.info("=" * 70)
logger.info(
    "Experiment 2 model refinement started."
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

baseline_summary_file = (
    EXPERIMENT_01_DIR
    / "tables"
    / "baseline_cv_summary.csv"
)

check_file_exists(model_ready_file)
check_file_exists(split_file)
check_file_exists(baseline_summary_file)

print("\nModel-ready dataset:")
print(model_ready_file)

print("\nSaved train-test split:")
print(split_file)

print("\nExperiment 1 cross-validation summary:")
print(baseline_summary_file)

# =========================================================
# LOAD DATASETS
# =========================================================

model_df = pd.read_csv(
    model_ready_file
)

split_df = pd.read_csv(
    split_file
)

baseline_summary_df = pd.read_csv(
    baseline_summary_file
)

logger.info(
    "Model-ready dataset loaded with %s rows and %s columns.",
    model_df.shape[0],
    model_df.shape[1],
)

logger.info(
    "Saved split table loaded with %s records.",
    len(split_df),
)

print("\nModel-ready dataset shape:")
print(model_df.shape)

print("\nSplit distribution:")
print(
    split_df["Dataset_Split"]
    .value_counts()
)

print("\nTarget distribution:")
print(
    model_df["Binary_Disability"]
    .value_counts()
    .sort_index()
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
    "Required Experiment 2 columns validated."
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
        "The reconstructed training dataset should contain "
        f"48 patients, but found {len(train_df)}."
    )

if len(held_out_test_df) != 12:
    raise ValueError(
        "The reconstructed test dataset should contain "
        f"12 patients, but found {len(held_out_test_df)}."
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

print("\nReconstructed training-set shape:")
print(train_df.shape)

print("\nHeld-out test-set shape:")
print(held_out_test_df.shape)

print("\nTraining target distribution:")
print(
    y_train.value_counts().sort_index()
)

print(
    "\nThe held-out test records have been loaded only to "
    "verify separation and will not be used during tuning."
)

logger.info(
    "Original Experiment 1 split reconstructed successfully."
)

# =========================================================
# DEFINE EXPERIMENT 2 FEATURE SETS
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

multimodal_feature_columns = (
    clinical_feature_columns
    + mri_feature_columns
)

X_train_clinical = train_df[
    clinical_feature_columns
].copy()

X_train_multimodal = train_df[
    multimodal_feature_columns
].copy()

print("\nConstant features removed:")
print(constant_features)

print("\nClinical training feature shape:")
print(X_train_clinical.shape)

print("\nMultimodal training feature shape:")
print(X_train_multimodal.shape)

logger.info(
    "Experiment 2 feature sets created. Clinical: %s. "
    "Multimodal: %s.",
    X_train_clinical.shape,
    X_train_multimodal.shape,
)

# =========================================================
# DEFINE LEAKAGE-SAFE PREPROCESSING
# =========================================================

def create_preprocessor(
    dataframe: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create numerical and categorical preprocessing pipelines.

    Preprocessing is fitted within each training fold.
    """

    categorical_columns = (
        dataframe
        .select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    numerical_columns = (
        dataframe
        .select_dtypes(
            include=[
                "number",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# =========================================================
# DEFINE EXPERIMENT 2 CANDIDATES
# =========================================================

candidate_models = {
    "Clinical Logistic Regression": {
        "Feature_Set": "Clinical",
        "X_Train": X_train_clinical,
        "Estimator": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    },

    "Clinical Random Forest": {
        "Feature_Set": "Clinical",
        "X_Train": X_train_clinical,
        "Estimator": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    },

    "Clinical Support Vector Machine": {
        "Feature_Set": "Clinical",
        "X_Train": X_train_clinical,
        "Estimator": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    },

    "Multimodal XGBoost": {
        "Feature_Set": "Multimodal",
        "X_Train": X_train_multimodal,
        "Estimator": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    },
}

print("\nExperiment 2 candidates:")

for candidate_name, candidate_details in (
    candidate_models.items()
):
    print(
        f"- {candidate_name}: "
        f"{candidate_details['X_Train'].shape[1]} features"
    )

logger.info(
    "%s Experiment 2 candidates defined.",
    len(candidate_models),
)

# =========================================================
# CONFIGURE REPEATED CROSS-VALIDATION
# =========================================================

REPEATED_CV_FOLDS = 5
REPEATED_CV_REPEATS = 10

repeated_cross_validation = (
    RepeatedStratifiedKFold(
        n_splits=REPEATED_CV_FOLDS,
        n_repeats=REPEATED_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
)

total_validation_folds = (
    REPEATED_CV_FOLDS
    * REPEATED_CV_REPEATS
)

print("\nRepeated cross-validation configuration:")
print(f"- Folds: {REPEATED_CV_FOLDS}")
print(f"- Repeats: {REPEATED_CV_REPEATS}")
print(
    f"- Total validation folds per candidate: "
    f"{total_validation_folds}"
)
print(f"- Random state: {RANDOM_STATE}")

logger.info(
    "Repeated stratified cross-validation configured: "
    "%s folds × %s repeats.",
    REPEATED_CV_FOLDS,
    REPEATED_CV_REPEATS,
)

# =========================================================
# SAVE EXPERIMENT 2 SETUP
# =========================================================

experiment_02_setup = {
    "experiment": (
        "Experiment 2: Model Refinement"
    ),
    "training_patients": int(
        len(train_df)
    ),
    "held_out_test_patients": int(
        len(held_out_test_df)
    ),
    "held_out_test_used_during_tuning": False,
    "repeated_cv_folds": (
        REPEATED_CV_FOLDS
    ),
    "repeated_cv_repeats": (
        REPEATED_CV_REPEATS
    ),
    "total_validation_folds_per_candidate": (
        total_validation_folds
    ),
    "random_state": RANDOM_STATE,
    "constant_features_removed": (
        constant_features
    ),
    "clinical_feature_count": len(
        clinical_feature_columns
    ),
    "multimodal_feature_count": len(
        multimodal_feature_columns
    ),
    "candidate_models": list(
        candidate_models.keys()
    ),
}

experiment_02_setup_output = (
    EXPERIMENT_02_DIR
    / "experiment_02_setup.json"
)

with open(
    experiment_02_setup_output,
    "w",
    encoding="utf-8",
) as setup_file:
    json.dump(
        experiment_02_setup,
        setup_file,
        indent=4,
    )

logger.info(
    "Experiment 2 setup saved to %s.",
    experiment_02_setup_output,
)

print_section(
    "Experiment 2 Setup Complete"
)

print("\nExperiment setup saved to:")
print(experiment_02_setup_output)

print(
    "\nNo hyperparameter tuning or held-out test "
    "evaluation has been performed yet."
)

# =========================================================
# DEFINE REPEATED-CV SCORING METRICS
# =========================================================

repeated_cv_scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        zero_division=0,
    ),
    "roc_auc": "roc_auc",
}

logger.info(
    "Repeated cross-validation scoring metrics configured."
)


# =========================================================
# RUN UNTUNED REPEATED CROSS-VALIDATION
# =========================================================

print_section(
    "Untuned Repeated Cross-Validation"
)

repeated_cv_fold_records = []

for candidate_number, (
    candidate_name,
    candidate_details,
) in enumerate(
    candidate_models.items(),
    start=1,
):

    print(
        f"\n[{candidate_number}/{len(candidate_models)}] "
        f"Evaluating {candidate_name}"
    )

    feature_set_name = (
        candidate_details["Feature_Set"]
    )

    X_candidate = (
        candidate_details["X_Train"]
    )

    estimator = (
        candidate_details["Estimator"]
    )

    preprocessor = create_preprocessor(
        X_candidate
    )

    candidate_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                clone(estimator),
            ),
        ]
    )

    try:
        cv_results = cross_validate(
            estimator=candidate_pipeline,
            X=X_candidate,
            y=y_train,
            cv=repeated_cross_validation,
            scoring=repeated_cv_scoring,
            return_train_score=True,
            return_estimator=False,
            error_score="raise",

            # Keep this at 1 because some classifiers
            # already use internal parallel processing.
            n_jobs=1,
        )

    except Exception as error:
        logger.exception(
            "Repeated cross-validation failed for %s.",
            candidate_name,
        )

        raise RuntimeError(
            "Repeated cross-validation failed for "
            f"{candidate_name}: {error}"
        ) from error

    number_of_results = len(
        cv_results["test_accuracy"]
    )

    if number_of_results != total_validation_folds:
        raise ValueError(
            f"{candidate_name} produced "
            f"{number_of_results} validation results; "
            f"expected {total_validation_folds}."
        )

    for result_index in range(
        number_of_results
    ):
        repeat_number = (
            result_index // REPEATED_CV_FOLDS
        ) + 1

        fold_number = (
            result_index % REPEATED_CV_FOLDS
        ) + 1

        repeated_cv_fold_records.append({
            "Candidate": candidate_name,
            "Feature_Set": feature_set_name,
            "Repeat": repeat_number,
            "Fold": fold_number,

            "Train_Accuracy": cv_results[
                "train_accuracy"
            ][result_index],

            "Validation_Accuracy": cv_results[
                "test_accuracy"
            ][result_index],

            "Validation_Precision": cv_results[
                "test_precision"
            ][result_index],

            "Validation_Recall": cv_results[
                "test_recall"
            ][result_index],

            "Validation_F1": cv_results[
                "test_f1"
            ][result_index],

            "Validation_ROC_AUC": cv_results[
                "test_roc_auc"
            ][result_index],

            "Fit_Time_Seconds": cv_results[
                "fit_time"
            ][result_index],

            "Score_Time_Seconds": cv_results[
                "score_time"
            ][result_index],
        })

    logger.info(
        "Untuned repeated cross-validation completed "
        "for %s.",
        candidate_name,
    )

    print(
        f"Completed {number_of_results} "
        "validation folds."
    )


# =========================================================
# CREATE FOLD-LEVEL RESULTS TABLE
# =========================================================

untuned_repeated_cv_fold_df = pd.DataFrame(
    repeated_cv_fold_records
)

expected_total_rows = (
    len(candidate_models)
    * total_validation_folds
)

if len(untuned_repeated_cv_fold_df) != expected_total_rows:
    raise ValueError(
        "Unexpected number of repeated-CV results. "
        f"Expected {expected_total_rows}, found "
        f"{len(untuned_repeated_cv_fold_df)}."
    )

print("\nRepeated-CV fold-level result shape:")
print(untuned_repeated_cv_fold_df.shape)

print("\nFirst ten results:")
print(
    untuned_repeated_cv_fold_df
    .head(10)
    .to_string(index=False)
)

# =========================================================
# SUMMARISE UNTUNED REPEATED-CV PERFORMANCE
# =========================================================

untuned_repeated_cv_summary_df = (
    untuned_repeated_cv_fold_df
    .groupby(
        [
            "Candidate",
            "Feature_Set",
        ],
        as_index=False,
    )
    .agg(
        Mean_Train_Accuracy=(
            "Train_Accuracy",
            "mean",
        ),
        SD_Train_Accuracy=(
            "Train_Accuracy",
            "std",
        ),
        Mean_CV_Accuracy=(
            "Validation_Accuracy",
            "mean",
        ),
        SD_CV_Accuracy=(
            "Validation_Accuracy",
            "std",
        ),
        Mean_CV_Precision=(
            "Validation_Precision",
            "mean",
        ),
        SD_CV_Precision=(
            "Validation_Precision",
            "std",
        ),
        Mean_CV_Recall=(
            "Validation_Recall",
            "mean",
        ),
        SD_CV_Recall=(
            "Validation_Recall",
            "std",
        ),
        Mean_CV_F1=(
            "Validation_F1",
            "mean",
        ),
        SD_CV_F1=(
            "Validation_F1",
            "std",
        ),
        Mean_CV_ROC_AUC=(
            "Validation_ROC_AUC",
            "mean",
        ),
        SD_CV_ROC_AUC=(
            "Validation_ROC_AUC",
            "std",
        ),
        Mean_Fit_Time_Seconds=(
            "Fit_Time_Seconds",
            "mean",
        ),
    )
)

untuned_repeated_cv_summary_df[
    "Train_Validation_Accuracy_Gap"
] = (
    untuned_repeated_cv_summary_df[
        "Mean_Train_Accuracy"
    ]
    - untuned_repeated_cv_summary_df[
        "Mean_CV_Accuracy"
    ]
)

untuned_repeated_cv_summary_df = (
    untuned_repeated_cv_summary_df
    .sort_values(
        by=[
            "Mean_CV_ROC_AUC",
            "Mean_CV_F1",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

numeric_summary_columns = (
    untuned_repeated_cv_summary_df
    .select_dtypes(include="number")
    .columns
)

untuned_repeated_cv_summary_df[
    numeric_summary_columns
] = untuned_repeated_cv_summary_df[
    numeric_summary_columns
].round(4)

print("\nUntuned repeated-CV summary:")
print(
    untuned_repeated_cv_summary_df
    .to_string(index=False)
)

# =========================================================
# SAVE UNTUNED REPEATED-CV RESULTS
# =========================================================

untuned_fold_output = (
    experiment_02_tables_dir
    / "untuned_repeated_cv_fold_results.csv"
)

untuned_summary_output = (
    experiment_02_tables_dir
    / "untuned_repeated_cv_summary.csv"
)

save_dataframe(
    dataframe=untuned_repeated_cv_fold_df,
    output_path=untuned_fold_output,
    index=False,
)

save_dataframe(
    dataframe=untuned_repeated_cv_summary_df,
    output_path=untuned_summary_output,
    index=False,
)

logger.info(
    "Untuned repeated-CV fold results saved to %s.",
    untuned_fold_output,
)

logger.info(
    "Untuned repeated-CV summary saved to %s.",
    untuned_summary_output,
)

print_section(
    "Untuned Repeated Cross-Validation Complete"
)

print("\nFold-level results saved to:")
print(untuned_fold_output)

print("\nSummary results saved to:")
print(untuned_summary_output)

print(
    "\nThe held-out test set has not been used."
)

# =========================================================
# EXPERIMENT 2 VISUALISATIONS
# =========================================================

print_section(
    "Experiment 2 Visualisations"
)

experiment_02_figures_dir = (
    EXPERIMENT_02_DIR /
    "figures"
)

experiment_02_figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------
# Figure 1
# Mean ROC-AUC
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

ordered = untuned_repeated_cv_summary_df.sort_values(
    "Mean_CV_ROC_AUC",
    ascending=False
)

plt.bar(
    ordered["Candidate"],
    ordered["Mean_CV_ROC_AUC"]
)

plt.xticks(rotation=15, ha="right")

plt.ylabel("Mean ROC-AUC")

plt.title(
    "Experiment 2 - Mean ROC-AUC"
)

plt.tight_layout()

roc_output = (
    experiment_02_figures_dir /
    "01_mean_roc_auc.png"
)

plt.savefig(
    roc_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Figure 2
# Mean Accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

ordered = untuned_repeated_cv_summary_df.sort_values(
    "Mean_CV_Accuracy",
    ascending=False
)

plt.bar(
    ordered["Candidate"],
    ordered["Mean_CV_Accuracy"]
)

plt.xticks(rotation=15, ha="right")

plt.ylabel("Mean Accuracy")

plt.title(
    "Experiment 2 - Mean Accuracy"
)

plt.tight_layout()

accuracy_output = (
    experiment_02_figures_dir /
    "02_mean_accuracy.png"
)

plt.savefig(
    accuracy_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Figure 3
# Mean F1
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

ordered = untuned_repeated_cv_summary_df.sort_values(
    "Mean_CV_F1",
    ascending=False
)

plt.bar(
    ordered["Candidate"],
    ordered["Mean_CV_F1"]
)

plt.xticks(rotation=15, ha="right")

plt.ylabel("Mean F1")

plt.title(
    "Experiment 2 - Mean F1 Score"
)

plt.tight_layout()

f1_output = (
    experiment_02_figures_dir /
    "03_mean_f1.png"
)

plt.savefig(
    f1_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Figure 4
# Train/Test Gap
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

ordered = untuned_repeated_cv_summary_df.sort_values(
    "Train_Validation_Accuracy_Gap",
    ascending=False
)

plt.bar(
    ordered["Candidate"],
    ordered["Train_Validation_Accuracy_Gap"]
)

plt.xticks(rotation=15, ha="right")

plt.ylabel("Accuracy Gap")

plt.title(
    "Experiment 2 - Train vs Validation Gap"
)

plt.tight_layout()

gap_output = (
    experiment_02_figures_dir /
    "04_train_validation_gap.png"
)

plt.savefig(
    gap_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Figure 5
# ROC-AUC with Error Bars
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

ordered = untuned_repeated_cv_summary_df.sort_values(
    "Mean_CV_ROC_AUC",
    ascending=False
)

plt.bar(
    ordered["Candidate"],
    ordered["Mean_CV_ROC_AUC"],
    yerr=ordered["SD_CV_ROC_AUC"],
    capsize=5
)

plt.xticks(rotation=15, ha="right")

plt.ylabel("ROC-AUC")

plt.title(
    "Experiment 2 - ROC-AUC Stability"
)

plt.tight_layout()

error_output = (
    experiment_02_figures_dir /
    "05_roc_auc_errorbars.png"
)

plt.savefig(
    error_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# Figure 6
# Boxplot of Validation Accuracy
# ---------------------------------------------------------

plt.figure(figsize=(9,6))

boxplot_data = []

labels = []

for candidate in untuned_repeated_cv_fold_df[
    "Candidate"
].unique():

    labels.append(candidate)

    boxplot_data.append(

        untuned_repeated_cv_fold_df.loc[
            untuned_repeated_cv_fold_df[
                "Candidate"
            ] == candidate,
            "Validation_Accuracy"
        ]

    )

plt.boxplot(
    boxplot_data,
    tick_labels=labels
)

plt.xticks(rotation=15, ha="right")

plt.ylabel(
    "Validation Accuracy"
)

plt.title(
    "Experiment 2 - Accuracy Distribution Across 50 Validation Folds"
)

plt.tight_layout()

boxplot_output = (
    experiment_02_figures_dir /
    "06_accuracy_boxplot.png"
)

plt.savefig(
    boxplot_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()

print("Experiment 2 figures saved to:")

print(experiment_02_figures_dir)

logger.info(
    "Experiment 2 visualisations completed successfully."
)

# =========================================================
# EXPERIMENT 2B — HYPERPARAMETER OPTIMISATION
# =========================================================

print_section(
    "Experiment 2B: Hyperparameter Optimisation"
)

logger.info("=" * 70)
logger.info(
    "Experiment 2B hyperparameter optimisation started."
)
logger.info("=" * 70)


# =========================================================
# CONFIGURE INNER CROSS-VALIDATION
# =========================================================

tuning_cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

TUNING_ITERATIONS = 20

tuning_scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        zero_division=0,
    ),
    "roc_auc": "roc_auc",
}

print("\nHyperparameter search configuration:")
print("- Search method: RandomizedSearchCV")
print("- Inner validation: Stratified 5-fold")
print(f"- Parameter samples per candidate: {TUNING_ITERATIONS}")
print("- Optimisation metric: ROC-AUC")
print("- Held-out test set used: No")

logger.info(
    "RandomizedSearchCV configured with %s iterations "
    "and stratified five-fold validation.",
    TUNING_ITERATIONS,
)

# =========================================================
# DEFINE HYPERPARAMETER SEARCH SPACES
# =========================================================

candidate_search_spaces = {
    "Clinical Logistic Regression": {
        "classifier__C": loguniform(
            1e-3,
            1e2,
        ),
        "classifier__penalty": [
            "l1",
            "l2",
        ],
        "classifier__solver": [
            "liblinear",
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    },

    "Clinical Random Forest": {
        "classifier__n_estimators": randint(
            100,
            601,
        ),
        "classifier__max_depth": [
            None,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
        ],
        "classifier__min_samples_split": randint(
            2,
            11,
        ),
        "classifier__min_samples_leaf": randint(
            1,
            7,
        ),
        "classifier__max_features": [
            "sqrt",
            "log2",
            0.5,
            0.75,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],
    },

    "Clinical Support Vector Machine": {
        "classifier__C": loguniform(
            1e-3,
            1e2,
        ),
        "classifier__gamma": loguniform(
            1e-4,
            1.0,
        ),
        "classifier__kernel": [
            "rbf",
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    },

    "Multimodal XGBoost": {
        "classifier__n_estimators": randint(
            50,
            401,
        ),
        "classifier__max_depth": randint(
            1,
            6,
        ),
        "classifier__learning_rate": loguniform(
            0.01,
            0.30,
        ),
        "classifier__subsample": uniform(
            0.60,
            0.40,
        ),
        "classifier__colsample_bytree": uniform(
            0.60,
            0.40,
        ),
        "classifier__min_child_weight": randint(
            1,
            11,
        ),
        "classifier__gamma": uniform(
            0.0,
            1.0,
        ),
        "classifier__reg_alpha": loguniform(
            1e-4,
            10.0,
        ),
        "classifier__reg_lambda": loguniform(
            1e-3,
            100.0,
        ),
        "classifier__scale_pos_weight": [
            1.0,
            1.4,
        ],
    },
}

for candidate_name in candidate_models:
    if candidate_name not in candidate_search_spaces:
        raise KeyError(
            "No hyperparameter search space was defined for "
            f"{candidate_name}."
        )

logger.info(
    "Hyperparameter search spaces defined for all candidates."
)

# =========================================================
# RUN HYPERPARAMETER SEARCH
# =========================================================

tuning_summary_records = []
all_tuning_results = []
best_tuned_pipelines = {}

for candidate_number, (
    candidate_name,
    candidate_details,
) in enumerate(
    candidate_models.items(),
    start=1,
):

    print_section(
        f"Tuning: {candidate_name}"
    )

    print(
        f"\nCandidate {candidate_number} "
        f"of {len(candidate_models)}"
    )

    feature_set_name = (
        candidate_details["Feature_Set"]
    )

    X_candidate = (
        candidate_details["X_Train"]
    )

    estimator = clone(
        candidate_details["Estimator"]
    )

    preprocessor = create_preprocessor(
        X_candidate
    )

    candidate_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "classifier",
                estimator,
            ),
        ]
    )

    random_search = RandomizedSearchCV(
        estimator=candidate_pipeline,
        param_distributions=(
            candidate_search_spaces[
                candidate_name
            ]
        ),
        n_iter=TUNING_ITERATIONS,
        scoring=tuning_scoring,
        refit="roc_auc",
        cv=tuning_cross_validation,
        random_state=RANDOM_STATE,
        return_train_score=True,
        error_score="raise",

        # Use one outer job to avoid nested parallelism
        # with Random Forest and XGBoost.
        n_jobs=1,

        verbose=1,
    )

    tuning_start_time = perf_counter()

    try:
        random_search.fit(
            X_candidate,
            y_train,
        )

    except Exception as error:
        logger.exception(
            "Hyperparameter tuning failed for %s.",
            candidate_name,
        )

        raise RuntimeError(
            "Hyperparameter tuning failed for "
            f"{candidate_name}: {error}"
        ) from error

    tuning_time_seconds = (
        perf_counter() - tuning_start_time
    )

    best_tuned_pipelines[
        candidate_name
    ] = random_search.best_estimator_

    candidate_cv_results_df = pd.DataFrame(
        random_search.cv_results_
    )

    candidate_cv_results_df.insert(
        0,
        "Candidate",
        candidate_name,
    )

    candidate_cv_results_df.insert(
        1,
        "Feature_Set",
        feature_set_name,
    )

    all_tuning_results.append(
        candidate_cv_results_df
    )

    best_result_index = (
        random_search.best_index_
    )

    tuning_summary_records.append({
        "Candidate": candidate_name,
        "Feature_Set": feature_set_name,
        "Best_Mean_CV_ROC_AUC": (
            random_search.best_score_
        ),
        "Best_SD_CV_ROC_AUC": (
            candidate_cv_results_df.loc[
                best_result_index,
                "std_test_roc_auc",
            ]
        ),
        "Best_Mean_CV_Accuracy": (
            candidate_cv_results_df.loc[
                best_result_index,
                "mean_test_accuracy",
            ]
        ),
        "Best_Mean_CV_Precision": (
            candidate_cv_results_df.loc[
                best_result_index,
                "mean_test_precision",
            ]
        ),
        "Best_Mean_CV_Recall": (
            candidate_cv_results_df.loc[
                best_result_index,
                "mean_test_recall",
            ]
        ),
        "Best_Mean_CV_F1": (
            candidate_cv_results_df.loc[
                best_result_index,
                "mean_test_f1",
            ]
        ),
        "Best_Mean_Train_Accuracy": (
            candidate_cv_results_df.loc[
                best_result_index,
                "mean_train_accuracy",
            ]
        ),
        "Tuning_Time_Seconds": (
            tuning_time_seconds
        ),
        "Best_Parameters_JSON": json.dumps(
            random_search.best_params_,
            sort_keys=True,
        ),
    })

    safe_candidate_name = (
        candidate_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    tuned_model_output = (
        experiment_02_models_dir
        / f"tuned_{safe_candidate_name}.joblib"
    )

    joblib.dump(
        random_search.best_estimator_,
        tuned_model_output,
    )

    best_parameters_output = (
        experiment_02_models_dir
        / f"tuned_{safe_candidate_name}_parameters.json"
    )

    with open(
        best_parameters_output,
        "w",
        encoding="utf-8",
    ) as parameters_file:
        json.dump(
            random_search.best_params_,
            parameters_file,
            indent=4,
        )

    logger.info(
        "Tuning completed for %s. Best ROC-AUC: %.4f.",
        candidate_name,
        random_search.best_score_,
    )

    print("\nBest mean CV ROC-AUC:")
    print(f"{random_search.best_score_:.4f}")

    print("\nBest parameters:")
    for parameter_name, parameter_value in (
        random_search.best_params_.items()
    ):
        print(
            f"- {parameter_name}: {parameter_value}"
        )

    print("\nTuned pipeline saved to:")
    print(tuned_model_output)

# =========================================================
# CREATE TUNING RESULTS TABLES
# =========================================================

tuning_summary_df = pd.DataFrame(
    tuning_summary_records
)

tuning_summary_df[
    "Train_Validation_Accuracy_Gap"
] = (
    tuning_summary_df[
        "Best_Mean_Train_Accuracy"
    ]
    - tuning_summary_df[
        "Best_Mean_CV_Accuracy"
    ]
)

tuning_summary_df = (
    tuning_summary_df
    .sort_values(
        by=[
            "Best_Mean_CV_ROC_AUC",
            "Best_Mean_CV_F1",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

numeric_tuning_columns = (
    tuning_summary_df
    .select_dtypes(include="number")
    .columns
)

tuning_summary_df[
    numeric_tuning_columns
] = tuning_summary_df[
    numeric_tuning_columns
].round(4)

all_tuning_results_df = pd.concat(
    all_tuning_results,
    ignore_index=True,
)

print("\nHyperparameter tuning summary:")
print(
    tuning_summary_df.to_string(
        index=False
    )
)

tuning_summary_output = (
    experiment_02_tables_dir
    / "hyperparameter_tuning_summary.csv"
)

all_tuning_results_output = (
    experiment_02_tables_dir
    / "hyperparameter_search_all_results.csv"
)

save_dataframe(
    dataframe=tuning_summary_df,
    output_path=tuning_summary_output,
    index=False,
)

save_dataframe(
    dataframe=all_tuning_results_df,
    output_path=all_tuning_results_output,
    index=False,
)

logger.info(
    "Hyperparameter tuning summary saved to %s.",
    tuning_summary_output,
)

logger.info(
    "All hyperparameter search results saved to %s.",
    all_tuning_results_output,
)

# =========================================================
# SAVE TUNING CONFIGURATION
# =========================================================

tuning_configuration = {
    "search_method": "RandomizedSearchCV",
    "number_of_candidates": len(
        candidate_models
    ),
    "parameter_samples_per_candidate": (
        TUNING_ITERATIONS
    ),
    "inner_cross_validation": (
        "StratifiedKFold"
    ),
    "inner_cv_folds": 5,
    "refit_metric": "roc_auc",
    "random_state": RANDOM_STATE,
    "training_patients": int(
        len(train_df)
    ),
    "held_out_test_used": False,
    "estimated_candidate_fold_fits": (
        len(candidate_models)
        * TUNING_ITERATIONS
        * 5
    ),
}

tuning_configuration_output = (
    EXPERIMENT_02_DIR
    / "hyperparameter_tuning_configuration.json"
)

with open(
    tuning_configuration_output,
    "w",
    encoding="utf-8",
) as configuration_file:
    json.dump(
        tuning_configuration,
        configuration_file,
        indent=4,
    )

logger.info(
    "Tuning configuration saved to %s.",
    tuning_configuration_output,
)

print_section(
    "Hyperparameter Optimisation Complete"
)

print("\nTuning summary saved to:")
print(tuning_summary_output)

print("\nComplete search results saved to:")
print(all_tuning_results_output)

print("\nTuning configuration saved to:")
print(tuning_configuration_output)

print(
    "\nThe held-out test set was not used during tuning."
)

# =========================================================
# RE-EVALUATE TUNED MODELS USING REPEATED CROSS-VALIDATION
# =========================================================

print_section(
    "Tuned Repeated Cross-Validation"
)

tuned_repeated_cv_fold_records = []

for candidate_number, (
    candidate_name,
    tuned_pipeline,
) in enumerate(
    best_tuned_pipelines.items(),
    start=1,
):
    print(
        f"\n[{candidate_number}/{len(best_tuned_pipelines)}] "
        f"Evaluating tuned {candidate_name}"
    )

    candidate_details = candidate_models[
        candidate_name
    ]

    feature_set_name = candidate_details[
        "Feature_Set"
    ]

    X_candidate = candidate_details[
        "X_Train"
    ]

    try:
        tuned_cv_results = cross_validate(
            estimator=clone(tuned_pipeline),
            X=X_candidate,
            y=y_train,
            cv=repeated_cross_validation,
            scoring=repeated_cv_scoring,
            return_train_score=True,
            error_score="raise",
            n_jobs=1,
        )

    except Exception as error:
        logger.exception(
            "Tuned repeated cross-validation failed "
            "for %s.",
            candidate_name,
        )

        raise RuntimeError(
            "Tuned repeated cross-validation failed for "
            f"{candidate_name}: {error}"
        ) from error

    number_of_results = len(
        tuned_cv_results["test_accuracy"]
    )

    if number_of_results != total_validation_folds:
        raise ValueError(
            f"{candidate_name} produced "
            f"{number_of_results} validation results; "
            f"expected {total_validation_folds}."
        )

    for result_index in range(
        number_of_results
    ):
        repeat_number = (
            result_index // REPEATED_CV_FOLDS
        ) + 1

        fold_number = (
            result_index % REPEATED_CV_FOLDS
        ) + 1

        tuned_repeated_cv_fold_records.append({
            "Candidate": candidate_name,
            "Feature_Set": feature_set_name,
            "Model_Stage": "Tuned",
            "Repeat": repeat_number,
            "Fold": fold_number,

            "Train_Accuracy": tuned_cv_results[
                "train_accuracy"
            ][result_index],

            "Validation_Accuracy": tuned_cv_results[
                "test_accuracy"
            ][result_index],

            "Validation_Precision": tuned_cv_results[
                "test_precision"
            ][result_index],

            "Validation_Recall": tuned_cv_results[
                "test_recall"
            ][result_index],

            "Validation_F1": tuned_cv_results[
                "test_f1"
            ][result_index],

            "Validation_ROC_AUC": tuned_cv_results[
                "test_roc_auc"
            ][result_index],

            "Fit_Time_Seconds": tuned_cv_results[
                "fit_time"
            ][result_index],

            "Score_Time_Seconds": tuned_cv_results[
                "score_time"
            ][result_index],
        })

    logger.info(
        "Tuned repeated cross-validation completed "
        "for %s.",
        candidate_name,
    )

    print(
        f"Completed {number_of_results} "
        "validation folds."
    )


# =========================================================
# CREATE TUNED FOLD-LEVEL RESULTS
# =========================================================

tuned_repeated_cv_fold_df = pd.DataFrame(
    tuned_repeated_cv_fold_records
)

expected_tuned_rows = (
    len(best_tuned_pipelines)
    * total_validation_folds
)

if len(tuned_repeated_cv_fold_df) != expected_tuned_rows:
    raise ValueError(
        "Unexpected number of tuned repeated-CV results. "
        f"Expected {expected_tuned_rows}, found "
        f"{len(tuned_repeated_cv_fold_df)}."
    )

print("\nTuned repeated-CV fold-level shape:")
print(tuned_repeated_cv_fold_df.shape)


# =========================================================
# SUMMARISE TUNED REPEATED-CV PERFORMANCE
# =========================================================

tuned_repeated_cv_summary_df = (
    tuned_repeated_cv_fold_df
    .groupby(
        [
            "Candidate",
            "Feature_Set",
        ],
        as_index=False,
    )
    .agg(
        Mean_Train_Accuracy=(
            "Train_Accuracy",
            "mean",
        ),
        SD_Train_Accuracy=(
            "Train_Accuracy",
            "std",
        ),
        Mean_CV_Accuracy=(
            "Validation_Accuracy",
            "mean",
        ),
        SD_CV_Accuracy=(
            "Validation_Accuracy",
            "std",
        ),
        Mean_CV_Precision=(
            "Validation_Precision",
            "mean",
        ),
        SD_CV_Precision=(
            "Validation_Precision",
            "std",
        ),
        Mean_CV_Recall=(
            "Validation_Recall",
            "mean",
        ),
        SD_CV_Recall=(
            "Validation_Recall",
            "std",
        ),
        Mean_CV_F1=(
            "Validation_F1",
            "mean",
        ),
        SD_CV_F1=(
            "Validation_F1",
            "std",
        ),
        Mean_CV_ROC_AUC=(
            "Validation_ROC_AUC",
            "mean",
        ),
        SD_CV_ROC_AUC=(
            "Validation_ROC_AUC",
            "std",
        ),
        Mean_Fit_Time_Seconds=(
            "Fit_Time_Seconds",
            "mean",
        ),
    )
)

tuned_repeated_cv_summary_df[
    "Train_Validation_Accuracy_Gap"
] = (
    tuned_repeated_cv_summary_df[
        "Mean_Train_Accuracy"
    ]
    - tuned_repeated_cv_summary_df[
        "Mean_CV_Accuracy"
    ]
)

tuned_repeated_cv_summary_df = (
    tuned_repeated_cv_summary_df
    .sort_values(
        by=[
            "Mean_CV_ROC_AUC",
            "Mean_CV_F1",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

tuned_numeric_columns = (
    tuned_repeated_cv_summary_df
    .select_dtypes(include="number")
    .columns
)

tuned_repeated_cv_summary_df[
    tuned_numeric_columns
] = tuned_repeated_cv_summary_df[
    tuned_numeric_columns
].round(4)

print("\nTuned repeated-CV summary:")
print(
    tuned_repeated_cv_summary_df.to_string(
        index=False
    )
)


# =========================================================
# CREATE UNTUNED VERSUS TUNED COMPARISON
# =========================================================

untuned_comparison_df = (
    untuned_repeated_cv_summary_df[
        [
            "Candidate",
            "Feature_Set",
            "Mean_CV_Accuracy",
            "Mean_CV_F1",
            "Mean_CV_ROC_AUC",
            "SD_CV_ROC_AUC",
            "Train_Validation_Accuracy_Gap",
        ]
    ]
    .copy()
)

untuned_comparison_df[
    "Model_Stage"
] = "Untuned"

tuned_comparison_df = (
    tuned_repeated_cv_summary_df[
        [
            "Candidate",
            "Feature_Set",
            "Mean_CV_Accuracy",
            "Mean_CV_F1",
            "Mean_CV_ROC_AUC",
            "SD_CV_ROC_AUC",
            "Train_Validation_Accuracy_Gap",
        ]
    ]
    .copy()
)

tuned_comparison_df[
    "Model_Stage"
] = "Tuned"

before_after_comparison_df = pd.concat(
    [
        untuned_comparison_df,
        tuned_comparison_df,
    ],
    ignore_index=True,
)

before_after_comparison_df = (
    before_after_comparison_df
    .sort_values(
        [
            "Candidate",
            "Model_Stage",
        ]
    )
    .reset_index(drop=True)
)

print("\nUntuned versus tuned comparison:")
print(
    before_after_comparison_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE TUNED EVALUATION RESULTS
# =========================================================

tuned_fold_output = (
    experiment_02_tables_dir
    / "tuned_repeated_cv_fold_results.csv"
)

tuned_summary_output = (
    experiment_02_tables_dir
    / "tuned_repeated_cv_summary.csv"
)

before_after_output = (
    experiment_02_tables_dir
    / "untuned_vs_tuned_comparison.csv"
)

save_dataframe(
    dataframe=tuned_repeated_cv_fold_df,
    output_path=tuned_fold_output,
    index=False,
)

save_dataframe(
    dataframe=tuned_repeated_cv_summary_df,
    output_path=tuned_summary_output,
    index=False,
)

save_dataframe(
    dataframe=before_after_comparison_df,
    output_path=before_after_output,
    index=False,
)

logger.info(
    "Tuned repeated-CV summary saved to %s.",
    tuned_summary_output,
)

logger.info(
    "Untuned-versus-tuned comparison saved to %s.",
    before_after_output,
)

print_section(
    "Tuned Repeated Cross-Validation Complete"
)

print("\nTuned summary saved to:")
print(tuned_summary_output)

print("\nBefore-and-after comparison saved to:")
print(before_after_output)

print(
    "\nThe held-out test set has not been used."
)

print_section(
    "Final Candidate Comparison"
)

comparison = before_after_comparison_df.copy()

comparison = comparison[
    [
        "Candidate",
        "Model_Stage",
        "Mean_CV_Accuracy",
        "Mean_CV_F1",
        "Mean_CV_ROC_AUC",
        "Train_Validation_Accuracy_Gap",
    ]
]

print(comparison.to_string(index=False))

best_model = tuned_repeated_cv_summary_df.iloc[0]

print_section(
    "Selected Final Model"
)

print(best_model)

logger.info(
    "Final model selected: %s",
    best_model["Candidate"],
)

final_model_information = {

    "Selected_Model":

        best_model["Candidate"],

    "Feature_Set":

        best_model["Feature_Set"],

    "ROC_AUC":

        float(best_model["Mean_CV_ROC_AUC"]),

    "Accuracy":

        float(best_model["Mean_CV_Accuracy"]),

    "F1":

        float(best_model["Mean_CV_F1"]),

    "Reason":

        (
            "Highest repeated cross-validation ROC-AUC "
            "with strong balanced performance."
        ),

}

with open(

    EXPERIMENT_02_DIR
    / "selected_final_model.json",

    "w",

    encoding="utf-8",

) as file:

    json.dump(
        final_model_information,
        file,
        indent=4,
    )

best_pipeline = best_tuned_pipelines[
    best_model["Candidate"]
]

joblib.dump(

    best_pipeline,

    EXPERIMENT_02_DIR
    / "final_selected_model.joblib"

)

# =========================================================
# CREATE UNTUNED VS TUNED ROC-AUC COMPARISON FIGURE
# =========================================================

print_section(
    "Untuned Versus Tuned ROC-AUC Comparison"
)

roc_auc_comparison_df = (
    before_after_comparison_df[
        [
            "Candidate",
            "Model_Stage",
            "Mean_CV_ROC_AUC",
        ]
    ]
    .copy()
)

roc_auc_pivot_df = (
    roc_auc_comparison_df
    .pivot(
        index="Candidate",
        columns="Model_Stage",
        values="Mean_CV_ROC_AUC",
    )
    .reset_index()
)

# Ensure the expected columns exist
required_stage_columns = [
    "Untuned",
    "Tuned",
]

missing_stage_columns = [
    column
    for column in required_stage_columns
    if column not in roc_auc_pivot_df.columns
]

if missing_stage_columns:
    raise ValueError(
        "Missing model-stage columns for comparison: "
        f"{missing_stage_columns}"
    )

roc_auc_pivot_df = (
    roc_auc_pivot_df
    .sort_values(
        by="Tuned",
        ascending=True,
    )
    .reset_index(drop=True)
)

chart_positions = np.arange(
    len(roc_auc_pivot_df)
)

bar_height = 0.36

plt.figure(
    figsize=(11, 7)
)

plt.barh(
    chart_positions - bar_height / 2,
    roc_auc_pivot_df["Untuned"],
    height=bar_height,
    label="Untuned",
)

plt.barh(
    chart_positions + bar_height / 2,
    roc_auc_pivot_df["Tuned"],
    height=bar_height,
    label="Tuned",
)

plt.yticks(
    chart_positions,
    roc_auc_pivot_df["Candidate"],
)

plt.xlabel(
    "Mean repeated cross-validation ROC-AUC"
)

plt.ylabel(
    "Candidate model"
)

plt.title(
    "Untuned and Tuned Model ROC-AUC Comparison"
)

plt.xlim(
    0,
    1.0,
)

plt.legend()

plt.tight_layout()

untuned_tuned_figure_output = (
    experiment_02_figures_dir
    / "07_untuned_vs_tuned_roc_auc.png"
)

plt.savefig(
    untuned_tuned_figure_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

logger.info(
    "Untuned-versus-tuned ROC-AUC figure saved to %s.",
    untuned_tuned_figure_output,
)

print("\nUntuned-versus-tuned figure saved to:")
print(untuned_tuned_figure_output)

# =========================================================
# SAVE EXPERIMENT 2 CONCLUSION
# =========================================================

experiment_02_conclusion = (
    "Experiment 2 evaluated four candidate models using "
    "repeated stratified five-fold cross-validation with "
    "ten repeats before and after hyperparameter optimisation.\n\n"
    
    "The tuned Clinical Random Forest achieved the strongest "
    "overall repeated cross-validation performance, with a "
    "mean ROC-AUC of 0.8303, mean accuracy of 0.7464, mean "
    "recall of 0.7850 and mean F1-score of 0.7193. Its "
    "train-validation accuracy gap was reduced to 0.0947, "
    "indicating substantially lower overfitting than the "
    "untuned baseline.\n\n"
    
    "The tuned Clinical Logistic Regression model remained "
    "highly competitive, achieving a mean ROC-AUC of 0.8246 "
    "and mean F1-score of 0.6937, while demonstrating the "
    "smallest positive train-validation accuracy gap among "
    "the leading candidates. It was therefore retained as "
    "the interpretable comparator model.\n\n"
    
    "The tuned Support Vector Machine and Multimodal XGBoost "
    "models achieved reasonable ROC-AUC values but weak "
    "threshold-based precision, recall and F1-scores. They "
    "were therefore not selected for the main explainability "
    "analysis.\n\n"
    
    "The tuned Clinical Random Forest was selected as the "
    "primary predictive model for the subsequent SHAP and "
    "LIME explainability stage."
)

experiment_02_conclusion_output = (
    EXPERIMENT_02_DIR
    / "experiment_02_conclusion.txt"
)

with open(
    experiment_02_conclusion_output,
    "w",
    encoding="utf-8",
) as conclusion_file:
    conclusion_file.write(
        experiment_02_conclusion
    )

logger.info(
    "Experiment 2 conclusion saved to %s.",
    experiment_02_conclusion_output,
)

print("\nExperiment 2 conclusion saved to:")
print(experiment_02_conclusion_output)

# =========================================================
# COMPLETE EXPERIMENT 2
# =========================================================

logger.info(
    "Experiment 2 completed successfully."
)

print_section(
    "Experiment 2 Complete"
)

print("\nSelected primary model:")
print(
    final_model_information[
        "Selected_Model"
    ]
)

print("\nSelected feature set:")
print(
    final_model_information[
        "Feature_Set"
    ]
)

print("\nSelected model ROC-AUC:")
print(
    final_model_information[
        "ROC_AUC"
    ]
)

print("\nFinal selected model saved to:")
print(
    EXPERIMENT_02_DIR
    / "final_selected_model.joblib"
)

print("\nSelected-model metadata saved to:")
print(
    EXPERIMENT_02_DIR
    / "selected_final_model.json"
)

print("\nExperiment conclusion saved to:")
print(
    experiment_02_conclusion_output
)

print(
    "\nExperiment 2 is complete. "
    "The held-out test set has not been used "
    "during model refinement."
)

