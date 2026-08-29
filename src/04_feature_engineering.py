"""
=========================================================
04_feature_engineering.py

Feature Engineering

This script prepares the final modelling dataset.

Author: Farah Iqbal
MSc Artificial Intelligence
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    LOGS_DIR,
)

from utils import (
    setup_logger,
    check_file_exists,
    validate_required_columns,
    report_dataset_quality,
    save_dataframe,
    print_section,
)

logger = setup_logger(
    logger_name="feature_engineering",
    log_file=LOGS_DIR / "feature_engineering.log"
)

print_section("Feature Engineering")

logger.info("=" * 70)
logger.info("Feature engineering started.")
logger.info("=" * 70)

# =========================================================
# DEFINE AND VALIDATE INPUT FILE
# =========================================================

multimodal_file = (
    PROCESSED_DATA_DIR
    / "multimodal_dataset.csv"
)

check_file_exists(multimodal_file)

print("\nMultimodal dataset found:")
print(multimodal_file)

# =========================================================
# LOAD MULTIMODAL DATASET
# =========================================================

multimodal_df = pd.read_csv(multimodal_file)

logger.info(
    "Multimodal dataset loaded with %s rows and %s columns.",
    multimodal_df.shape[0],
    multimodal_df.shape[1],
)

print("\nDataset shape:")
print(multimodal_df.shape)

print("\nFirst five rows:")
print(multimodal_df.head().to_string())

multimodal_file = PROCESSED_DATA_DIR / "multimodal_dataset.csv"

multimodal_df = pd.read_csv(multimodal_file)

logger.info("Multimodal dataset loaded successfully.")

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Patient_ID",
    "EDSS",
    "EDSS_Category",
    "Gender",
    "Age",
    "Age_of_onset",
    "Disease_Duration",
    "Total_Abnormal_Neuro_Findings",
    "Lesion_Voxels",
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
    dataframe=multimodal_df,
    required_columns=required_columns,
    dataset_name="Multimodal dataset",
)

logger.info("Required modelling columns validated.")

# =========================================================
# INSPECT EDSS CATEGORIES
# =========================================================

print("\nEDSS category distribution:")
print(
    multimodal_df["EDSS_Category"]
    .value_counts(dropna=False)
    .reindex(["Low", "Moderate", "High"])
)

# =========================================================
# CREATE BINARY DISABILITY TARGET
# =========================================================

binary_target_mapping = {
    "Low": 0,
    "Moderate": 1,
    "High": 1,
}

multimodal_df["Binary_Disability"] = (
    multimodal_df["EDSS_Category"]
    .map(binary_target_mapping)
)

if multimodal_df["Binary_Disability"].isna().any():
    invalid_categories = (
        multimodal_df.loc[
            multimodal_df["Binary_Disability"].isna(),
            "EDSS_Category",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Unexpected EDSS categories were found: "
        f"{invalid_categories}"
    )

multimodal_df["Binary_Disability"] = (
    multimodal_df["Binary_Disability"]
    .astype(int)
)

logger.info("Binary disability target created.")

print("\nBinary target distribution:")
print(
    multimodal_df["Binary_Disability"]
    .value_counts()
    .sort_index()
)

# =========================================================
# DEFINE NON-PREDICTIVE AND LEAKAGE VARIABLES
# =========================================================

columns_excluded_from_modelling = [
    "Patient_ID",
    "EDSS",
    "EDSS_Category",
    "Binary_Disability",
    "MRI_EDSS_Time_Difference_Under_2_Months",
    "Lesion_Voxels",
]

print("\nColumns excluded from predictive modelling:")

for column in columns_excluded_from_modelling:
    print("-", column)

# =========================================================
# CREATE FULL PREDICTOR DATASET
# =========================================================

X_full = multimodal_df.drop(
    columns=columns_excluded_from_modelling
).copy()

y = multimodal_df["Binary_Disability"].copy()

print("\nFull predictor dataset shape:")
print(X_full.shape)

print("\nTarget vector shape:")
print(y.shape)

# =========================================================
# DEFINE MRI FEATURE SET
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

validate_required_columns(
    dataframe=X_full,
    required_columns=mri_feature_columns,
    dataset_name="Predictor dataset",
)

X_mri = X_full[mri_feature_columns].copy()

print("\nMRI feature-set shape:")
print(X_mri.shape)

# =========================================================
# DEFINE CLINICAL FEATURE SET
# =========================================================

clinical_feature_columns = [
    column
    for column in X_full.columns
    if column not in mri_feature_columns
]

X_clinical = X_full[
    clinical_feature_columns
].copy()

print("\nClinical feature-set shape:")
print(X_clinical.shape)

print("\nClinical features:")
for column in clinical_feature_columns:
    print("-", column)

# =========================================================
# DEFINE MULTIMODAL FEATURE SET
# =========================================================

X_multimodal = X_full.copy()

print("\nMultimodal feature-set shape:")
print(X_multimodal.shape)

# =========================================================
# IDENTIFY FEATURE DATA TYPES
# =========================================================

categorical_features = (
    X_multimodal
    .select_dtypes(
        include=["object", "string", "category"]
    )
    .columns
    .tolist()
)

numerical_features = (
    X_multimodal
    .select_dtypes(
        include=["number", "bool"]
    )
    .columns
    .tolist()
)

print("\nCategorical features:")
for column in categorical_features:
    print("-", column)

print("\nNumber of categorical features:")
print(len(categorical_features))

print("\nNumerical features:")
for column in numerical_features:
    print("-", column)

print("\nNumber of numerical features:")
print(len(numerical_features))

# =========================================================
# CREATE FEATURE SUMMARY TABLE
# =========================================================

feature_summary_records = []

for column in X_multimodal.columns:
    if column in mri_feature_columns:
        modality = "MRI"
    else:
        modality = "Clinical"

    if column in categorical_features:
        feature_type = "Categorical"
    else:
        feature_type = "Numerical"

    feature_summary_records.append({
        "Feature": column,
        "Modality": modality,
        "Feature_Type": feature_type,
        "Data_Type": str(
            X_multimodal[column].dtype
        ),
        "Missing_Values": int(
            X_multimodal[column].isna().sum()
        ),
        "Unique_Values": int(
            X_multimodal[column].nunique(
                dropna=False
            )
        ),
    })

feature_summary_df = pd.DataFrame(
    feature_summary_records
)

print("\nFeature summary:")
print(feature_summary_df.to_string(index=False))

# =========================================================
# SAVE FEATURE DEFINITIONS
# =========================================================

feature_summary_output = (
    TABLES_DIR
    / "feature_summary.csv"
)

clinical_feature_output = (
    TABLES_DIR
    / "clinical_feature_list.csv"
)

mri_feature_output = (
    TABLES_DIR
    / "mri_feature_list.csv"
)

multimodal_feature_output = (
    TABLES_DIR
    / "multimodal_feature_list.csv"
)

save_dataframe(
    dataframe=feature_summary_df,
    output_path=feature_summary_output,
    index=False,
)

save_dataframe(
    dataframe=pd.DataFrame({
        "Feature": clinical_feature_columns
    }),
    output_path=clinical_feature_output,
    index=False,
)

save_dataframe(
    dataframe=pd.DataFrame({
        "Feature": mri_feature_columns
    }),
    output_path=mri_feature_output,
    index=False,
)

save_dataframe(
    dataframe=pd.DataFrame({
        "Feature": X_multimodal.columns
    }),
    output_path=multimodal_feature_output,
    index=False,
)

# =========================================================
# CREATE MODEL-READY DATASET
# =========================================================

model_ready_df = X_multimodal.copy()

model_ready_df.insert(
    0,
    "Patient_ID",
    multimodal_df["Patient_ID"],
)

model_ready_df["Binary_Disability"] = y

print("\nModel-ready dataset shape:")
print(model_ready_df.shape)

print("\nColumns containing missing values:")
print(
    model_ready_df
    .isna()
    .sum()
    .loc[
        lambda values: values > 0
    ]
    .sort_values(ascending=False)
)

# =========================================================
# CREATE MODEL-READY DATA QUALITY REPORT
# =========================================================

model_quality_report = report_dataset_quality(
    dataframe=model_ready_df,
    dataset_name="Model-Ready Dataset",
)

model_quality_output = (
    TABLES_DIR
    / "model_ready_data_quality.csv"
)

save_dataframe(
    dataframe=model_quality_report,
    output_path=model_quality_output,
    index=False,
)

# =========================================================
# SAVE MODEL-READY DATASET
# =========================================================

model_ready_output = (
    PROCESSED_DATA_DIR
    / "model_ready_dataset.csv"
)

save_dataframe(
    dataframe=model_ready_df,
    output_path=model_ready_output,
    index=False,
)

logger.info(
    "Model-ready dataset saved to %s.",
    model_ready_output,
)

logger.info(
    "Feature engineering completed successfully."
)

print_section("Feature Engineering Complete")

print("\nClinical feature-set shape:")
print(X_clinical.shape)

print("\nMRI feature-set shape:")
print(X_mri.shape)

print("\nMultimodal feature-set shape:")
print(X_multimodal.shape)

print("\nFinal model-ready dataset shape:")
print(model_ready_df.shape)

print("\nModel-ready dataset saved to:")
print(model_ready_output)

