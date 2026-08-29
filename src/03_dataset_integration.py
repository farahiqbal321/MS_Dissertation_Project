"""
=========================================================
03_dataset_integration.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Integrates the cleaned clinical dataset with patient-level
FLAIR MRI-derived lesion biomarkers to create a validated
multimodal dataset for disability-severity modelling.

Outputs
-------
multimodal_dataset.csv
multimodal_merge_validation.csv

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

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


# =========================================================
# INITIALISE LOGGER
# =========================================================

logger = setup_logger(
    logger_name="dataset_integration",
    log_file=LOGS_DIR / "dataset_integration.log",
)


# =========================================================
# START PIPELINE
# =========================================================

print_section("Multimodal Dataset Integration")

logger.info("=" * 70)
logger.info("Multimodal dataset integration started.")
logger.info("=" * 70)


# =========================================================
# DEFINE INPUT AND OUTPUT PATHS
# =========================================================

clinical_file = (
    PROCESSED_DATA_DIR
    / "cleaned_clinical_data.csv"
)

flair_mri_file = (
    PROCESSED_DATA_DIR
    / "mri_features_flair.csv"
)

multimodal_output_file = (
    PROCESSED_DATA_DIR
    / "multimodal_dataset.csv"
)

merge_validation_output = (
    TABLES_DIR
    / "multimodal_merge_validation.csv"
)

multimodal_quality_output = (
    TABLES_DIR
    / "multimodal_data_quality.csv"
)


# =========================================================
# VALIDATE INPUT FILES
# =========================================================

check_file_exists(clinical_file)
check_file_exists(flair_mri_file)

print("\nClinical dataset found:")
print(clinical_file)

print("\nFLAIR MRI feature dataset found:")
print(flair_mri_file)

logger.info(
    "Clinical input file: %s",
    clinical_file,
)

logger.info(
    "FLAIR MRI input file: %s",
    flair_mri_file,
)


# =========================================================
# LOAD DATASETS
# =========================================================

clinical_df = pd.read_csv(clinical_file)

flair_mri_df = pd.read_csv(flair_mri_file)

print("\nClinical dataset shape:")
print(clinical_df.shape)

print("\nFLAIR MRI dataset shape:")
print(flair_mri_df.shape)

logger.info(
    "Clinical dataset loaded with %s rows and %s columns.",
    clinical_df.shape[0],
    clinical_df.shape[1],
)

logger.info(
    "FLAIR MRI dataset loaded with %s rows and %s columns.",
    flair_mri_df.shape[0],
    flair_mri_df.shape[1],
)


# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

clinical_required_columns = [
    "Patient_ID",
    "Gender",
    "Age",
    "Age_of_onset",
    "EDSS",
    "Disease_Duration",
    "EDSS_Category",
    "Total_Abnormal_Neuro_Findings",
]

mri_required_columns = [
    "Patient_ID",
    "Modality",
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
    dataframe=clinical_df,
    required_columns=clinical_required_columns,
    dataset_name="Cleaned clinical dataset",
)

validate_required_columns(
    dataframe=flair_mri_df,
    required_columns=mri_required_columns,
    dataset_name="FLAIR MRI feature dataset",
)

logger.info(
    "Required clinical and MRI columns were validated."
)


# =========================================================
# VALIDATE FLAIR MODALITY
# =========================================================

modality_values = (
    flair_mri_df["Modality"]
    .dropna()
    .unique()
    .tolist()
)

print("\nMRI modalities present:")
print(modality_values)

if modality_values != ["FLAIR"]:
    raise ValueError(
        "The primary MRI dataset should contain only FLAIR "
        f"records, but found: {modality_values}"
    )

logger.info(
    "FLAIR-only modality validation completed."
)


# =========================================================
# CHECK PATIENT IDENTIFIERS
# =========================================================

clinical_duplicate_ids = (
    clinical_df["Patient_ID"]
    .duplicated()
    .sum()
)

mri_duplicate_ids = (
    flair_mri_df["Patient_ID"]
    .duplicated()
    .sum()
)

print("\nDuplicate clinical Patient IDs:")
print(clinical_duplicate_ids)

print("\nDuplicate MRI Patient IDs:")
print(mri_duplicate_ids)

if clinical_duplicate_ids > 0:
    raise ValueError(
        "Duplicate Patient IDs were found in the clinical "
        f"dataset: {clinical_duplicate_ids}"
    )

if mri_duplicate_ids > 0:
    raise ValueError(
        "Duplicate Patient IDs were found in the FLAIR MRI "
        f"dataset: {mri_duplicate_ids}"
    )

clinical_ids = set(
    clinical_df["Patient_ID"]
    .astype(int)
)

mri_ids = set(
    flair_mri_df["Patient_ID"]
    .astype(int)
)

clinical_without_mri = sorted(
    clinical_ids - mri_ids
)

mri_without_clinical = sorted(
    mri_ids - clinical_ids
)

print("\nClinical patients without MRI features:")
print(clinical_without_mri)

print("\nMRI patients without clinical information:")
print(mri_without_clinical)

if clinical_without_mri:
    logger.warning(
        "Clinical patients without MRI features: %s",
        clinical_without_mri,
    )

if mri_without_clinical:
    logger.warning(
        "MRI patients without clinical information: %s",
        mri_without_clinical,
    )


# =========================================================
# SELECT MRI BIOMARKERS FOR INTEGRATION
# =========================================================

mri_biomarker_columns = [
    "Patient_ID",
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

flair_biomarkers_df = (
    flair_mri_df[mri_biomarker_columns]
    .copy()
)

print("\nMRI biomarkers selected for integration:")

for column in flair_biomarkers_df.columns:
    print("-", column)

logger.info(
    "%s MRI biomarker columns selected for integration.",
    len(mri_biomarker_columns) - 1,
)


# =========================================================
# CREATE MERGE VALIDATION TABLE
# =========================================================

merge_validation_df = pd.merge(
    clinical_df[["Patient_ID"]],
    flair_biomarkers_df[["Patient_ID"]],
    on="Patient_ID",
    how="outer",
    indicator=True,
)

merge_status_counts = (
    merge_validation_df["_merge"]
    .value_counts()
)

print("\nMerge-validation status:")
print(merge_status_counts)

save_dataframe(
    dataframe=merge_validation_df,
    output_path=merge_validation_output,
    index=False,
)

logger.info(
    "Merge-validation table saved to %s.",
    merge_validation_output,
)


# =========================================================
# MERGE CLINICAL AND MRI DATA
# =========================================================

multimodal_df = pd.merge(
    clinical_df,
    flair_biomarkers_df,
    on="Patient_ID",
    how="inner",
    validate="one_to_one",
)

multimodal_df = (
    multimodal_df
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)

logger.info(
    "Clinical and MRI datasets merged successfully."
)

print("\nMultimodal dataset shape:")
print(multimodal_df.shape)

print("\nFirst five multimodal records:")
print(multimodal_df.head().to_string())


# =========================================================
# VALIDATE MERGED DATASET
# =========================================================

expected_rows = len(
    clinical_ids.intersection(mri_ids)
)

if len(multimodal_df) != expected_rows:
    raise ValueError(
        "Unexpected number of rows after integration. "
        f"Expected {expected_rows}, found {len(multimodal_df)}."
    )

merged_duplicate_ids = (
    multimodal_df["Patient_ID"]
    .duplicated()
    .sum()
)

if merged_duplicate_ids > 0:
    raise ValueError(
        "Duplicate Patient IDs were found after integration: "
        f"{merged_duplicate_ids}"
    )

missing_values = (
    multimodal_df
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nDuplicate Patient IDs after integration:")
print(merged_duplicate_ids)

print("\nColumns containing missing values:")
print(missing_values[missing_values > 0])

logger.info(
    "Merged dataset validation completed. "
    "Rows: %s. Columns: %s. Missing values: %s.",
    multimodal_df.shape[0],
    multimodal_df.shape[1],
    multimodal_df.isna().sum().sum(),
)


# =========================================================
# CHECK REDUNDANT LESION FEATURES
# =========================================================

lesion_values_identical = (
    multimodal_df["Lesion_Voxels"]
    == multimodal_df["Lesion_Volume_mm3"]
).all()

print(
    "\nAre Lesion_Voxels and Lesion_Volume_mm3 identical?"
)
print(lesion_values_identical)

if lesion_values_identical:
    logger.info(
        "Lesion_Voxels and Lesion_Volume_mm3 are identical "
        "because the FLAIR voxel volume is 1 mm³. Both are "
        "retained in the integrated dataset for provenance; "
        "Lesion_Voxels will be removed before modelling."
    )


# =========================================================
# CREATE MULTIMODAL DATA QUALITY REPORT
# =========================================================

multimodal_quality_report = report_dataset_quality(
    dataframe=multimodal_df,
    dataset_name="Multimodal Dataset",
)

save_dataframe(
    dataframe=multimodal_quality_report,
    output_path=multimodal_quality_output,
    index=False,
)

logger.info(
    "Multimodal data-quality report saved to %s.",
    multimodal_quality_output,
)


# =========================================================
# EXPORT MULTIMODAL DATASET
# =========================================================

save_dataframe(
    dataframe=multimodal_df,
    output_path=multimodal_output_file,
    index=False,
)

logger.info(
    "Multimodal dataset saved to %s.",
    multimodal_output_file,
)

logger.info(
    "Multimodal dataset integration completed successfully."
)

print_section("Dataset Integration Complete")

print("\nFinal multimodal dataset shape:")
print(multimodal_df.shape)

print("\nMultimodal dataset saved to:")
print(multimodal_output_file)
