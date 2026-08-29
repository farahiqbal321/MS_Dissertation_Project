"""
=========================================================
01_clinical_preprocessing.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Loads the raw clinical dataset, performs preprocessing,
creates engineered clinical variables, evaluates data
quality and saves a cleaned clinical dataset for the
subsequent multimodal AI pipeline.

Outputs
-------
cleaned_clinical_data.csv
clinical_data_quality.csv

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from pathlib import Path

# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from config import (
    RAW_CLINICAL_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    LOGS_DIR,
)

from utils import (
    setup_logger,
    report_dataset_quality,
    validate_required_columns,
    save_dataframe,
    print_section,
)

# =========================================================
# INITIALISE LOGGER
# =========================================================

logger = setup_logger(
    logger_name="clinical_preprocessing",
    log_file=LOGS_DIR / "clinical_preprocessing.log",
)

# =========================================================
# START PIPELINE
# =========================================================

print_section("Clinical Data Preprocessing")

logger.info("=" * 70)
logger.info("Clinical preprocessing started.")
logger.info("=" * 70)


# =========================================================
# LOCATE CLINICAL DATASET
# =========================================================

clinical_file = RAW_CLINICAL_DIR / "patient_information.xlsx"

logger.info("Clinical dataset path: %s", clinical_file)

if not clinical_file.exists():
    raise FileNotFoundError(
        f"Clinical dataset was not found:\n{clinical_file}"
    )

print(f"\nClinical dataset found: {clinical_file.exists()}")


# =========================================================
# LOAD CLINICAL DATASET
# =========================================================

# The first Excel row contains a merged title.
# The second Excel row contains the actual column names.
clinical_df = pd.read_excel(
    clinical_file,
    header=1,
)

logger.info(
    "Clinical dataset loaded successfully with %s rows and %s columns.",
    clinical_df.shape[0],
    clinical_df.shape[1],
)


# =========================================================
# DISPLAY DATASET OVERVIEW
# =========================================================

print("\nDataset shape:")
print(clinical_df.shape)

print("\nColumn names:")
for column in clinical_df.columns:
    print("-", column)

print("\nFirst five rows:")
print(clinical_df.head().to_string())

print("\nDataset information:\n")
clinical_df.info()

# =========================================================
# STANDARDISE COLUMN NAMES
# =========================================================

# Remove leading/trailing spaces and repeated internal spaces
# from the original Excel column headings.
clinical_df.columns = (
    clinical_df.columns
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

column_name_mapping = {
    "ID": "Patient_ID",
    "Gender": "Gender",
    "Age": "Age",
    "Age of onset": "Age_of_onset",
    "EDSS": "EDSS",
    (
        "Does the time difference between MRI acquisition "
        "and EDSS < two months"
    ): "MRI_EDSS_Time_Difference_Under_2_Months",
    "Types of Medicines": "Types_of_Medicines",
    "Presenting Symptom": "Presenting_Symptom",

    # Original spreadsheet spelling
    "Dose the patient has Co-moroidity": "Co_morbidity",

    # Additional possible spelling variants
    "Does the patient has Co-moroidity": "Co_morbidity",
    "Dose the patient has Co-morbidity": "Co_morbidity",
    "Does the patient has Co-morbidity": "Co_morbidity",

    "Pyramidal": "Pyramidal",
    "Cerebella": "Cerebella",
    "Brain stem": "Brain_stem",
    "Sensory": "Sensory",
    "Sphincters": "Sphincters",
    "Visual": "Visual",
    "Mental": "Mental",
    "Speech": "Speech",
    "Motor System": "Motor_System",
    "Sensory System": "Sensory_System",
    "Coordination": "Coordination",
    "Gait": "Gait",
    "Bowel and bladder function": "Bowel_and_bladder_function",
    "Mobility": "Mobility",
    "Mental State": "Mental_State",
    "Optic discs": "Optic_discs",
    "Fields": "Fields",
    "Nystagmus": "Nystagmus",
    "Ocular Movement": "Ocular_Movement",
    "Swallowing": "Swallowing",
}

clinical_df = clinical_df.rename(columns=column_name_mapping)

logger.info("Clinical column names standardised.")

print("\nStandardised column names:")
for column in clinical_df.columns:
    print("-", column)

# =========================================================
# VALIDATE REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Patient_ID",
    "Gender",
    "Age",
    "Age_of_onset",
    "EDSS",
    "Types_of_Medicines",
    "Presenting_Symptom",
    "Co_morbidity",
    "Pyramidal",
    "Cerebella",
    "Brain_stem",
    "Sensory",
    "Sphincters",
    "Visual",
    "Mental",
    "Speech",
    "Motor_System",
    "Sensory_System",
    "Coordination",
    "Gait",
    "Bowel_and_bladder_function",
    "Mobility",
    "Mental_State",
    "Optic_discs",
    "Fields",
    "Nystagmus",
    "Ocular_Movement",
    "Swallowing",
]

validate_required_columns(
    dataframe=clinical_df,
    required_columns=required_columns,
    dataset_name="Clinical dataset",
)

logger.info("All required clinical columns were found.")

# =========================================================
# CLEAN CATEGORICAL VARIABLES
# =========================================================

categorical_columns = [
    "Gender",
    "MRI_EDSS_Time_Difference_Under_2_Months",
    "Types_of_Medicines",
    "Presenting_Symptom",
    "Co_morbidity",
]

for column in categorical_columns:
    clinical_df[column] = (
        clinical_df[column]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

# Treat the single invalid Gender code "N" as missing.
# Imputation will later be performed inside the modelling pipeline
# to avoid fitting preprocessing decisions on the complete dataset.
clinical_df["Gender"] = clinical_df["Gender"].replace(
    {"N": pd.NA}
)

# Correct obvious spelling and spacing inconsistencies.
clinical_df["Presenting_Symptom"] = (
    clinical_df["Presenting_Symptom"]
    .str.replace("Motore", "Motor", regex=False)
    .str.replace(r"\s*&\s*", " & ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

clinical_df["Co_morbidity"] = (
    clinical_df["Co_morbidity"]
    .str.title()
)

logger.info("Categorical clinical variables cleaned.")

print("\nGender values after cleaning:")
print(clinical_df["Gender"].value_counts(dropna=False))

print("\nPresenting symptom categories after cleaning:")
for category in sorted(
    clinical_df["Presenting_Symptom"].dropna().unique()
):
    print("-", category)

# =========================================================
# CREATE AND VALIDATE DISEASE DURATION
# =========================================================

clinical_df["Disease_Duration"] = (
    clinical_df["Age"] - clinical_df["Age_of_onset"]
)

# Identify negative derived durations
invalid_duration_mask = (
    clinical_df["Disease_Duration"] < 0
)

invalid_duration_rows = clinical_df.loc[
    invalid_duration_mask,
    [
        "Patient_ID",
        "Age",
        "Age_of_onset",
        "Disease_Duration",
    ],
]

invalid_duration_count = len(invalid_duration_rows)

print("\nRecords with negative disease duration:")

if invalid_duration_count == 0:
    print("None")
else:
    print(
        invalid_duration_rows.to_string(index=False)
    )

    logger.warning(
        "%s negative disease-duration value(s) identified.",
        invalid_duration_count,
    )

    # Retain the original Age and Age_of_onset values.
    # Mark only the derived duration as missing.
    clinical_df.loc[
        invalid_duration_mask,
        "Disease_Duration",
    ] = np.nan

logger.info(
    "Disease duration created. Invalid values set to missing: %s.",
    invalid_duration_count,
)

print("\nDisease duration summary:")
print(clinical_df["Disease_Duration"].describe())

print("\nMissing disease-duration values:")
print(
    clinical_df["Disease_Duration"].isna().sum()
)

# =========================================================
# CREATE EDSS DISABILITY CATEGORIES
# =========================================================

clinical_df["EDSS_Category"] = pd.cut(
    clinical_df["EDSS"],
    bins=[
        -np.inf,
        2.5,
        5.5,
        np.inf,
    ],
    labels=[
        "Low",
        "Moderate",
        "High",
    ],
    include_lowest=True,
)

logger.info("EDSS disability categories created.")

print("\nEDSS category distribution:")
print(
    clinical_df["EDSS_Category"]
    .value_counts()
    .reindex(["Low", "Moderate", "High"])
)

# =========================================================
# CREATE TOTAL ABNORMAL NEUROLOGICAL FINDINGS
# =========================================================

neurological_columns = [
    "Pyramidal",
    "Cerebella",
    "Brain_stem",
    "Sensory",
    "Sphincters",
    "Visual",
    "Mental",
    "Speech",
    "Motor_System",
    "Sensory_System",
    "Coordination",
    "Gait",
    "Bowel_and_bladder_function",
    "Mobility",
    "Mental_State",
    "Optic_discs",
    "Fields",
    "Nystagmus",
    "Ocular_Movement",
    "Swallowing",
]

clinical_df["Total_Abnormal_Neuro_Findings"] = (
    clinical_df[neurological_columns].sum(axis=1)
)

logger.info(
    "Total abnormal neurological findings feature created."
)

print("\nTotal abnormal neurological findings summary:")
print(
    clinical_df[
        "Total_Abnormal_Neuro_Findings"
    ].describe()
)

# =========================================================
# DATA QUALITY CHECKS
# =========================================================

duplicate_patient_ids = (
    clinical_df["Patient_ID"].duplicated().sum()
)

missing_values = (
    clinical_df.isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nDuplicate Patient IDs:")
print(duplicate_patient_ids)

print("\nColumns containing missing values:")
print(missing_values[missing_values > 0])

if duplicate_patient_ids > 0:
    raise ValueError(
        f"Duplicate Patient IDs found: {duplicate_patient_ids}"
    )

logger.info(
    "Clinical data-quality checks completed. "
    "Duplicate Patient IDs: %s. Total missing values: %s.",
    duplicate_patient_ids,
    clinical_df.isna().sum().sum(),
)

# =========================================================
# CREATE DATA QUALITY REPORT
# =========================================================

quality_report = report_dataset_quality(
    dataframe=clinical_df,
    dataset_name="Cleaned Clinical Dataset",
)

clinical_quality_output = (
    TABLES_DIR / "clinical_data_quality.csv"
)

save_dataframe(
    dataframe=quality_report,
    output_path=clinical_quality_output,
    index=False,
)

logger.info(
    "Clinical data-quality report saved to %s.",
    clinical_quality_output,
)

# =========================================================
# EXPORT CLEANED CLINICAL DATASET
# =========================================================

clinical_output_file = (
    PROCESSED_DATA_DIR / "cleaned_clinical_data.csv"
)

save_dataframe(
    dataframe=clinical_df,
    output_path=clinical_output_file,
    index=False,
)

logger.info(
    "Cleaned clinical dataset saved to %s.",
    clinical_output_file,
)

logger.info("Clinical preprocessing completed successfully.")

print_section("Clinical Preprocessing Complete")

print("\nFinal cleaned dataset shape:")
print(clinical_df.shape)

print("\nCleaned dataset saved to:")
print(clinical_output_file)

