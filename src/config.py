"""
Central configuration file for the MS dissertation project.

This module stores reusable project paths and modelling settings.
Using pathlib keeps the paths compatible across different computers
and avoids hard-coded absolute Windows paths.
"""

from pathlib import Path


# =========================================================
# PROJECT DIRECTORIES
# =========================================================

# Locate the main project folder automatically.
# config.py is inside src, so .parent.parent moves to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_CLINICAL_DIR = RAW_DATA_DIR / "clinical"
RAW_MRI_DIR = RAW_DATA_DIR / "mri"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"

# =========================================================
# EXPERIMENT DIRECTORIES
# =========================================================

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

EXPERIMENT_01_DIR = (
    EXPERIMENTS_DIR / "experiment_01_modality_model_comparison"
)

EXPERIMENT_02_DIR = (
    EXPERIMENTS_DIR / "experiment_02_model_refinement"
)

EXPERIMENT_03_DIR = (
    EXPERIMENTS_DIR / "experiment_03_shap_lime_analysis"
)

EXPERIMENT_04_DIR = (
    EXPERIMENTS_DIR / "experiment_04_agentic_explanation_comparison"
)
EXPERIMENT_05_DIR = (
    EXPERIMENTS_DIR / "experiment_05_agentic_ablation_comparison"
)

EXPERIMENT_06_DIR = (
    EXPERIMENTS_DIR / "experiment_06_self_correcting_agentic_ai"
)


# =========================================================
# RAW DATA FILES
# =========================================================

PATIENT_INFORMATION_FILE = (
    RAW_CLINICAL_DIR
    / "patient_information.xlsx"
)

SEQUENCE_PARAMETERS_FILE = (
    RAW_CLINICAL_DIR
    / "sequence_parameters.xlsx"
)


# =========================================================
# PROCESSED DATA FILES
# =========================================================

CLEANED_CLINICAL_FILE = (
    PROCESSED_DATA_DIR
    / "cleaned_clinical_data.csv"
)

MRI_FEATURES_FLAIR_FILE = (
    PROCESSED_DATA_DIR
    / "mri_features_flair.csv"
)

MRI_FEATURES_ALL_MODALITIES_FILE = (
    PROCESSED_DATA_DIR
    / "mri_features_all_modalities.csv"
)

MULTIMODAL_DATASET_FILE = (
    PROCESSED_DATA_DIR
    / "multimodal_dataset.csv"
)

EDSS_CORRELATIONS_FILE = (
    PROCESSED_DATA_DIR
    / "edss_feature_correlations.csv"
)


# =========================================================
# MODELLING SETTINGS
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5

TARGET_COLUMN = "Binary_Disability"

LOW_DISABILITY_LABEL = 0
MODERATE_HIGH_DISABILITY_LABEL = 1


# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

def create_output_directories() -> None:
    """Create output folders if they do not already exist."""

    directories = [
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        LOGS_DIR,
        EXPERIMENTS_DIR,
        EXPERIMENT_01_DIR,
        EXPERIMENT_02_DIR,
        EXPERIMENT_03_DIR,
        EXPERIMENT_04_DIR,
        EXPERIMENT_05_DIR,
        EXPERIMENT_06_DIR,
        
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_output_directories()

    print("Project root:", PROJECT_ROOT)
    print("Raw clinical directory:", RAW_CLINICAL_DIR)
    print("Raw MRI directory:", RAW_MRI_DIR)
    print("Processed data directory:", PROCESSED_DATA_DIR)
    print("Outputs directory:", OUTPUTS_DIR)

