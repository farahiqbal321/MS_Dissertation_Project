"""
=========================================================
05_model_training.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Runs Experiment 1: modality and baseline model comparison.

Clinical-only, MRI-only and combined multimodal feature
sets are evaluated using identical stratified cross-
validation procedures.

A final test set is held aside and is not used for model
selection or cross-validation.

Outputs
-------
baseline_cv_fold_results.csv
baseline_cv_summary.csv
train_test_split.csv

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import json
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from time import perf_counter

# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from config import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    LOGS_DIR,
    MODELS_DIR,
    EXPERIMENT_01_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CROSS_VALIDATION_FOLDS,
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
    logger_name="model_training",
    log_file=LOGS_DIR / "model_training.log",
)

experiment_tables_dir = (
    EXPERIMENT_01_DIR / "tables"
)

experiment_models_dir = (
    EXPERIMENT_01_DIR / "models"
)

experiment_logs_dir = (
    EXPERIMENT_01_DIR / "logs"
)

for directory in [
    EXPERIMENT_01_DIR,
    experiment_tables_dir,
    experiment_models_dir,
    experiment_logs_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

# =========================================================
# START EXPERIMENT
# =========================================================

print_section(
    "Experiment 1: Modality and Model Comparison"
)

logger.info("=" * 70)
logger.info(
    "Experiment 1 modality and model comparison started."
)
logger.info("=" * 70)

# =========================================================
# LOAD MODEL-READY DATASET
# =========================================================

model_ready_file = (
    PROCESSED_DATA_DIR
    / "model_ready_dataset.csv"
)

check_file_exists(model_ready_file)

model_df = pd.read_csv(model_ready_file)

logger.info(
    "Model-ready dataset loaded with %s rows and %s columns.",
    model_df.shape[0],
    model_df.shape[1],
)

print("\nModel-ready dataset shape:")
print(model_df.shape)

print("\nTarget distribution:")
print(
    model_df["Binary_Disability"]
    .value_counts()
    .sort_index()
)

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
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

validate_required_columns(
    dataframe=model_df,
    required_columns=required_columns,
    dataset_name="Model-ready dataset",
)

logger.info(
    "Required model-training columns validated."
)

# =========================================================
# DEFINE FEATURE SETS
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

clinical_feature_columns = [
    column
    for column in all_predictor_columns
    if column not in mri_feature_columns
]

# Remove constant features from all relevant feature sets
constant_features = [
    column
    for column in all_predictor_columns
    if model_df[column].nunique(
        dropna=False
    ) <= 1
]

print("\nConstant features identified:")
print(constant_features)

clinical_feature_columns = [
    column
    for column in clinical_feature_columns
    if column not in constant_features
]

mri_feature_columns = [
    column
    for column in mri_feature_columns
    if column not in constant_features
]

multimodal_feature_columns = (
    clinical_feature_columns
    + mri_feature_columns
)

feature_sets = {
    "Clinical": clinical_feature_columns,
    "MRI": mri_feature_columns,
    "Multimodal": multimodal_feature_columns,
}

print("\nFeature-set sizes:")
for feature_set_name, columns in feature_sets.items():
    print(
        f"- {feature_set_name}: {len(columns)} features"
    )

logger.info(
    "Constant features removed: %s",
    constant_features,
)

# =========================================================
# CREATE FIXED TRAIN-TEST SPLIT
# =========================================================

patient_ids = model_df["Patient_ID"].copy()
target = model_df["Binary_Disability"].copy()

train_indices, test_indices = train_test_split(
    model_df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=target,
)

train_df = (
    model_df.loc[train_indices]
    .copy()
    .sort_values("Patient_ID")
)

test_df = (
    model_df.loc[test_indices]
    .copy()
    .sort_values("Patient_ID")
)

y_train = train_df[
    "Binary_Disability"
].copy()

y_test = test_df[
    "Binary_Disability"
].copy()

print("\nTraining-set shape:")
print(train_df.shape)

print("\nTest-set shape:")
print(test_df.shape)

print("\nTraining target distribution:")
print(
    y_train.value_counts().sort_index()
)

print("\nTest target distribution:")
print(
    y_test.value_counts().sort_index()
)

# =========================================================
# SAVE TRAIN-TEST SPLIT RECORD
# =========================================================

split_records = []

for _, row in train_df.iterrows():
    split_records.append({
        "Patient_ID": int(
            row["Patient_ID"]
        ),
        "Dataset_Split": "Train",
        "Binary_Disability": int(
            row["Binary_Disability"]
        ),
    })

for _, row in test_df.iterrows():
    split_records.append({
        "Patient_ID": int(
            row["Patient_ID"]
        ),
        "Dataset_Split": "Test",
        "Binary_Disability": int(
            row["Binary_Disability"]
        ),
    })

split_df = (
    pd.DataFrame(split_records)
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)

split_output = (
    experiment_tables_dir
    / "train_test_split.csv"
)

save_dataframe(
    dataframe=split_df,
    output_path=split_output,
    index=False,
)

# =========================================================
# DEFINE CROSS-VALIDATION STRATEGY
# =========================================================

cross_validation = StratifiedKFold(
    n_splits=CROSS_VALIDATION_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring_metrics = {
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

print("\nCross-validation configuration:")
print(
    f"- Strategy: Stratified "
    f"{CROSS_VALIDATION_FOLDS}-fold"
)
print("- Shuffle: True")
print(f"- Random state: {RANDOM_STATE}")

logger.info(
    "Stratified %s-fold cross-validation configured.",
    CROSS_VALIDATION_FOLDS,
)

# =========================================================
# DEFINE PREPROCESSING PIPELINE
# =========================================================

def create_preprocessor(
    dataframe: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create a leakage-safe preprocessing transformer.

    Numerical variables are median-imputed and standardised.
    Categorical variables are mode-imputed and one-hot encoded.
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

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor

# =========================================================
# DEFINE BASELINE CLASSIFICATION MODELS
# =========================================================

baseline_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

print("\nBaseline models:")
for model_name in baseline_models:
    print("-", model_name)

logger.info(
    "%s baseline models defined.",
    len(baseline_models),
)

# =========================================================
# PREPARE TRAINING FEATURE SETS
# =========================================================

training_feature_sets = {
    "Clinical": train_df[
        clinical_feature_columns
    ].copy(),

    "MRI": train_df[
        mri_feature_columns
    ].copy(),

    "Multimodal": train_df[
        multimodal_feature_columns
    ].copy(),
}

print("\nTraining feature-set shapes:")

for feature_set_name, feature_dataframe in (
    training_feature_sets.items()
):
    print(
        f"- {feature_set_name}: "
        f"{feature_dataframe.shape}"
    )

# =========================================================
# RUN BASELINE CROSS-VALIDATION
# =========================================================

fold_result_records = []

total_runs = (
    len(training_feature_sets)
    * len(baseline_models)
)

completed_runs = 0

for feature_set_name, X_train_set in (
    training_feature_sets.items()
):
    print_section(
        f"Cross-Validation: {feature_set_name}"
    )

    preprocessor = create_preprocessor(
        X_train_set
    )

    for model_name, model in baseline_models.items():
        completed_runs += 1

        print(
            f"\n[{completed_runs}/{total_runs}] "
            f"{feature_set_name} — {model_name}"
        )

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    clone(model),
                ),
            ]
        )

        try:
            cv_results = cross_validate(
                estimator=model_pipeline,
                X=X_train_set,
                y=y_train,
                cv=cross_validation,
                scoring=scoring_metrics,
                return_train_score=True,
                error_score="raise",
                n_jobs=1,
            )

            for fold_number in range(
                CROSS_VALIDATION_FOLDS
            ):
                fold_result_records.append({
                    "Feature_Set": feature_set_name,
                    "Model": model_name,
                    "Fold": fold_number + 1,

                    "Train_Accuracy": cv_results[
                        "train_accuracy"
                    ][fold_number],

                    "Validation_Accuracy": cv_results[
                        "test_accuracy"
                    ][fold_number],

                    "Validation_Precision": cv_results[
                        "test_precision"
                    ][fold_number],

                    "Validation_Recall": cv_results[
                        "test_recall"
                    ][fold_number],

                    "Validation_F1": cv_results[
                        "test_f1"
                    ][fold_number],

                    "Validation_ROC_AUC": cv_results[
                        "test_roc_auc"
                    ][fold_number],

                    "Fit_Time_Seconds": cv_results[
                        "fit_time"
                    ][fold_number],

                    "Score_Time_Seconds": cv_results[
                        "score_time"
                    ][fold_number],
                })

            logger.info(
                "Cross-validation completed for %s — %s.",
                feature_set_name,
                model_name,
            )

        except Exception as error:
            logger.exception(
                "Cross-validation failed for %s — %s.",
                feature_set_name,
                model_name,
            )

            raise RuntimeError(
                "Cross-validation failed for "
                f"{feature_set_name} — {model_name}: "
                f"{error}"
            ) from error

# =========================================================
# CREATE FOLD-LEVEL RESULTS TABLE
# =========================================================

baseline_cv_fold_df = pd.DataFrame(
    fold_result_records
)

print("\nFold-level result shape:")
print(baseline_cv_fold_df.shape)

print("\nFirst ten fold results:")
print(
    baseline_cv_fold_df
    .head(10)
    .to_string(index=False)
)

# =========================================================
# CREATE CROSS-VALIDATION SUMMARY
# =========================================================

baseline_cv_summary_df = (
    baseline_cv_fold_df
    .groupby(
        [
            "Feature_Set",
            "Model",
        ],
        as_index=False,
    )
    .agg(
        Mean_Train_Accuracy=(
            "Train_Accuracy",
            "mean",
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

baseline_cv_summary_df[
    "Train_Validation_Accuracy_Gap"
] = (
    baseline_cv_summary_df[
        "Mean_Train_Accuracy"
    ]
    - baseline_cv_summary_df[
        "Mean_CV_Accuracy"
    ]
)
baseline_cv_summary_df = (
    baseline_cv_summary_df
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
    baseline_cv_summary_df
    .select_dtypes(include="number")
    .columns
)

baseline_cv_summary_df[
    numeric_summary_columns
] = baseline_cv_summary_df[
    numeric_summary_columns
].round(4)

print("\nCross-validation summary:")
print(
    baseline_cv_summary_df.to_string(
        index=False
    )
)

# =========================================================
# SAVE BASELINE CROSS-VALIDATION RESULTS
# =========================================================

fold_results_output = (
    experiment_tables_dir
    / "baseline_cv_fold_results.csv"
)

summary_results_output = (
    experiment_tables_dir
    / "baseline_cv_summary.csv"
)

save_dataframe(
    dataframe=baseline_cv_fold_df,
    output_path=fold_results_output,
    index=False,
)

save_dataframe(
    dataframe=baseline_cv_summary_df,
    output_path=summary_results_output,
    index=False,
)

logger.info(
    "Baseline fold-level results saved to %s.",
    fold_results_output,
)

logger.info(
    "Baseline cross-validation summary saved to %s.",
    summary_results_output,
)

# =========================================================
# SAVE EXPERIMENT SETTINGS
# =========================================================

experiment_settings = {
    "experiment": (
        "Experiment 1: Modality and Model Comparison"
    ),
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cross_validation_folds": (
        CROSS_VALIDATION_FOLDS
    ),
    "cross_validation_strategy": (
        "StratifiedKFold with shuffling"
    ),
    "training_patients": int(
        len(train_df)
    ),
    "test_patients": int(
        len(test_df)
    ),
    "clinical_feature_count": len(
        clinical_feature_columns
    ),
    "mri_feature_count": len(
        mri_feature_columns
    ),
    "multimodal_feature_count": len(
        multimodal_feature_columns
    ),
    "constant_features_removed": (
        constant_features
    ),
    "models": list(
        baseline_models.keys()
    ),
    "test_set_used_for_model_selection": False,
}

settings_output = (
    EXPERIMENT_01_DIR
    / "experiment_settings.json"
)

with open(
    settings_output,
    "w",
    encoding="utf-8",
) as settings_file:
    json.dump(
        experiment_settings,
        settings_file,
        indent=4,
    )

logger.info(
    "Experiment settings saved to %s.",
    settings_output,
)

# =========================================================
# COMPLETE EXPERIMENT 1 BASELINE STAGE
# =========================================================

logger.info(
    "Experiment 1 baseline cross-validation "
    "completed successfully."
)

print_section(
    "Baseline Cross-Validation Complete"
)

print("\nFold-level results saved to:")
print(fold_results_output)

print("\nSummary results saved to:")
print(summary_results_output)

print("\nExperiment settings saved to:")
print(settings_output)

print(
    "\nThe held-out test set has not been used."
)

# =========================================================
# TRAIN AND EVALUATE FINAL BASELINE MODELS
# =========================================================

test_feature_sets = {
    "Clinical": test_df[
        clinical_feature_columns
    ].copy(),

    "MRI": test_df[
        mri_feature_columns
    ].copy(),

    "Multimodal": test_df[
        multimodal_feature_columns
    ].copy(),
}

baseline_test_results = []

for feature_set_name in training_feature_sets:

    print_section(
        f"Final Baseline Models: {feature_set_name}"
    )

    X_train = training_feature_sets[
        feature_set_name
    ]

    X_test = test_feature_sets[
        feature_set_name
    ]

    for model_name, model in baseline_models.items():

        print(
            f"\nTraining {feature_set_name} — {model_name}"
        )

        preprocessor = create_preprocessor(
            X_train
        )

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    clone(model),
                ),
            ]
        )

        # Measure training time
        training_start = perf_counter()

        model_pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            perf_counter() - training_start
        )

        # Measure prediction time
        prediction_start = perf_counter()

        predictions = model_pipeline.predict(
            X_test
        )

        probabilities = (
            model_pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        prediction_time = (
            perf_counter() - prediction_start
        )

        # Calculate held-out test metrics
        test_accuracy = accuracy_score(
            y_test,
            predictions,
        )

        test_precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        test_recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        test_f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        test_roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        # Calculate training accuracy for overfitting comparison
        training_predictions = (
            model_pipeline.predict(
                X_train
            )
        )

        training_accuracy = accuracy_score(
            y_train,
            training_predictions,
        )

        test_accuracy_gap = (
            training_accuracy - test_accuracy
        )

        # Create safe filename
        safe_feature_set_name = (
            feature_set_name
            .lower()
            .replace(" ", "_")
        )

        safe_model_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        model_filename = (
            f"{safe_feature_set_name}_"
            f"{safe_model_name}.joblib"
        )

        model_path = (
            experiment_models_dir
            / model_filename
        )

        joblib.dump(
            model_pipeline,
            model_path,
        )

        baseline_test_results.append({
            "Feature_Set": feature_set_name,
            "Model": model_name,
            "Train_Accuracy": training_accuracy,
            "Test_Accuracy": test_accuracy,
            "Test_Precision": test_precision,
            "Test_Recall": test_recall,
            "Test_F1": test_f1,
            "Test_ROC_AUC": test_roc_auc,
            "Train_Test_Accuracy_Gap": test_accuracy_gap,
            "Training_Time_Seconds": training_time,
            "Prediction_Time_Seconds": prediction_time,
            "Model_Path": str(model_path),
        })

        logger.info(
            "Final baseline model completed: %s — %s. "
            "Test ROC-AUC: %.4f. Test F1: %.4f.",
            feature_set_name,
            model_name,
            test_roc_auc,
            test_f1,
        )

        print(
            f"Test Accuracy: {test_accuracy:.4f}"
        )
        print(
            f"Test F1: {test_f1:.4f}"
        )
        print(
            f"Test ROC-AUC: {test_roc_auc:.4f}"
        )
        print(
            f"Saved model: {model_path.name}"
        )


# =========================================================
# CREATE AND SAVE HELD-OUT TEST RESULTS
# =========================================================

baseline_test_results_df = pd.DataFrame(
    baseline_test_results
)

baseline_test_results_df = (
    baseline_test_results_df
    .sort_values(
        by=[
            "Test_ROC_AUC",
            "Test_F1",
            "Test_Accuracy",
        ],
        ascending=False,
    )
    .reset_index(drop=True)
)

numeric_test_columns = (
    baseline_test_results_df
    .select_dtypes(include="number")
    .columns
)

baseline_test_results_df[
    numeric_test_columns
] = baseline_test_results_df[
    numeric_test_columns
].round(4)

print("\nHeld-out test results:")
print(
    baseline_test_results_df.to_string(
        index=False
    )
)

baseline_test_results_output = (
    experiment_tables_dir
    / "baseline_test_results.csv"
)

save_dataframe(
    dataframe=baseline_test_results_df,
    output_path=baseline_test_results_output,
    index=False,
)

logger.info(
    "Held-out baseline test results saved to %s.",
    baseline_test_results_output,
)

print("\nHeld-out test results saved to:")
print(baseline_test_results_output)

# =========================================================
# CREATE EXPERIMENT 1 VISUALISATIONS
# =========================================================

experiment_figures_dir = (
    EXPERIMENT_01_DIR / "figures"
)

experiment_figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)


def create_result_labels(
    results_dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Create readable labels containing the feature set
    and model name.
    """

    return (
        results_dataframe["Feature_Set"]
        + " | "
        + results_dataframe["Model"]
    )


def save_horizontal_metric_chart(
    results_dataframe: pd.DataFrame,
    metric_column: str,
    chart_title: str,
    x_axis_label: str,
    output_filename: str,
) -> None:
    """
    Create and save a horizontal comparison chart for
    one held-out test metric.
    """

    chart_df = (
        results_dataframe[
            [
                "Feature_Set",
                "Model",
                metric_column,
            ]
        ]
        .copy()
        .sort_values(
            metric_column,
            ascending=True,
        )
        .reset_index(drop=True)
    )

    chart_df["Result_Label"] = (
        create_result_labels(chart_df)
    )

    figure_height = max(
        8,
        len(chart_df) * 0.42,
    )

    plt.figure(
        figsize=(11, figure_height)
    )

    plt.barh(
        chart_df["Result_Label"],
        chart_df[metric_column],
    )

    plt.xlabel(x_axis_label)
    plt.ylabel("Feature set and model")
    plt.title(chart_title)

    # Add the numerical result beside each bar
    for row_number, metric_value in enumerate(
        chart_df[metric_column]
    ):
        plt.text(
            metric_value + 0.01,
            row_number,
            f"{metric_value:.3f}",
            va="center",
            fontsize=8,
        )

    # Classification metrics range from 0 to 1
    if metric_column in [
        "Test_Accuracy",
        "Test_F1",
        "Test_ROC_AUC",
    ]:
        plt.xlim(
            0,
            max(
                1.05,
                chart_df[metric_column].max() + 0.10,
            ),
        )

    plt.tight_layout()

    output_path = (
        experiment_figures_dir
        / output_filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Experiment figure saved to %s.",
        output_path,
    )


# =========================================================
# FIGURE 1 — HELD-OUT TEST ACCURACY
# =========================================================

save_horizontal_metric_chart(
    results_dataframe=baseline_test_results_df,
    metric_column="Test_Accuracy",
    chart_title=(
        "Held-Out Test Accuracy by Feature Set and Model"
    ),
    x_axis_label="Test accuracy",
    output_filename=(
        "01_test_accuracy_comparison.png"
    ),
)


# =========================================================
# FIGURE 2 — HELD-OUT TEST ROC-AUC
# =========================================================

save_horizontal_metric_chart(
    results_dataframe=baseline_test_results_df,
    metric_column="Test_ROC_AUC",
    chart_title=(
        "Held-Out Test ROC-AUC by Feature Set and Model"
    ),
    x_axis_label="Test ROC-AUC",
    output_filename=(
        "02_test_roc_auc_comparison.png"
    ),
)


# =========================================================
# FIGURE 3 — HELD-OUT TEST F1-SCORE
# =========================================================

save_horizontal_metric_chart(
    results_dataframe=baseline_test_results_df,
    metric_column="Test_F1",
    chart_title=(
        "Held-Out Test F1-Score by Feature Set and Model"
    ),
    x_axis_label="Test F1-score",
    output_filename=(
        "03_test_f1_comparison.png"
    ),
)


# =========================================================
# FIGURE 4 — MODEL TRAINING TIME
# =========================================================

save_horizontal_metric_chart(
    results_dataframe=baseline_test_results_df,
    metric_column="Training_Time_Seconds",
    chart_title=(
        "Baseline Model Training Time"
    ),
    x_axis_label="Training time in seconds",
    output_filename=(
        "04_training_time_comparison.png"
    ),
)


# =========================================================
# FIGURE 5 — TRAINING VS TEST ACCURACY
# =========================================================

accuracy_comparison_df = (
    baseline_test_results_df[
        [
            "Feature_Set",
            "Model",
            "Train_Accuracy",
            "Test_Accuracy",
        ]
    ]
    .copy()
    .sort_values(
        "Test_Accuracy",
        ascending=True,
    )
    .reset_index(drop=True)
)

accuracy_comparison_df[
    "Result_Label"
] = create_result_labels(
    accuracy_comparison_df
)

chart_positions = np.arange(
    len(accuracy_comparison_df)
)

bar_height = 0.38

figure_height = max(
    8,
    len(accuracy_comparison_df) * 0.46,
)

plt.figure(
    figsize=(12, figure_height)
)

plt.barh(
    chart_positions - bar_height / 2,
    accuracy_comparison_df[
        "Train_Accuracy"
    ],
    height=bar_height,
    label="Training accuracy",
)

plt.barh(
    chart_positions + bar_height / 2,
    accuracy_comparison_df[
        "Test_Accuracy"
    ],
    height=bar_height,
    label="Test accuracy",
)

plt.yticks(
    chart_positions,
    accuracy_comparison_df[
        "Result_Label"
    ],
)

plt.xlabel("Accuracy")
plt.ylabel("Feature set and model")
plt.title(
    "Training and Held-Out Test Accuracy Comparison"
)

plt.xlim(0, 1.05)
plt.legend()
plt.tight_layout()

train_test_figure_output = (
    experiment_figures_dir
    / "05_train_vs_test_accuracy.png"
)

plt.savefig(
    train_test_figure_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

logger.info(
    "Train-versus-test accuracy figure saved to %s.",
    train_test_figure_output,
)


# =========================================================
# FIGURE 6 — CROSS-VALIDATION ROC-AUC
# =========================================================

cv_roc_auc_df = (
    baseline_cv_summary_df[
        [
            "Feature_Set",
            "Model",
            "Mean_CV_ROC_AUC",
            "SD_CV_ROC_AUC",
        ]
    ]
    .copy()
    .sort_values(
        "Mean_CV_ROC_AUC",
        ascending=True,
    )
    .reset_index(drop=True)
)

cv_roc_auc_df[
    "Result_Label"
] = create_result_labels(
    cv_roc_auc_df
)

plt.figure(
    figsize=(
        12,
        max(8, len(cv_roc_auc_df) * 0.45),
    )
)

plt.barh(
    cv_roc_auc_df["Result_Label"],
    cv_roc_auc_df["Mean_CV_ROC_AUC"],
    xerr=cv_roc_auc_df["SD_CV_ROC_AUC"],
    capsize=3,
)

plt.xlabel(
    "Mean five-fold cross-validation ROC-AUC"
)
plt.ylabel("Feature set and model")
plt.title(
    "Cross-Validation ROC-AUC with Standard Deviation"
)

plt.xlim(0, 1.05)
plt.tight_layout()

cv_roc_auc_figure_output = (
    experiment_figures_dir
    / "06_cv_roc_auc_with_variability.png"
)

plt.savefig(
    cv_roc_auc_figure_output,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

logger.info(
    "Cross-validation ROC-AUC figure saved to %s.",
    cv_roc_auc_figure_output,
)


# =========================================================
# PRINT VISUALISATION OUTPUTS
# =========================================================

print("\nExperiment 1 figures saved to:")
print(experiment_figures_dir)

