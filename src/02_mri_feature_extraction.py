"""
=========================================================
02_mri_feature_extraction.py

MSc Artificial Intelligence Dissertation
Birmingham City University

Author: Farah Iqbal

Purpose
-------
Inspects the raw Multiple Sclerosis MRI dataset, identifies
MRI sequences and lesion segmentation masks, extracts
quantitative lesion biomarkers and saves the resulting
patient-level MRI feature dataset.

Planned Outputs
---------------
mri_features_flair.csv
mri_features_all_modalities.csv
mri_data_quality.csv

=========================================================
"""

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scipy import ndimage

# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from config import (
    RAW_MRI_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    LOGS_DIR,
)

from utils import (
    setup_logger,
    save_dataframe,
    print_section,
)

# =========================================================
# INITIALISE LOGGER
# =========================================================

logger = setup_logger(
    logger_name="mri_feature_extraction",
    log_file=LOGS_DIR / "mri_feature_extraction.log",
)

# =========================================================
# START PIPELINE
# =========================================================

print_section("MRI Feature Extraction")

logger.info("=" * 70)
logger.info("MRI feature extraction started.")
logger.info("=" * 70)

# =========================================================
# VALIDATE MRI DIRECTORY
# =========================================================

if not RAW_MRI_DIR.exists():
    raise FileNotFoundError(
        f"Raw MRI directory was not found:\n{RAW_MRI_DIR}"
    )

patient_folders = sorted(
    [
        folder
        for folder in RAW_MRI_DIR.iterdir()
        if folder.is_dir()
        and folder.name.lower().startswith("patient")
    ],
    key=lambda folder: int(
        "".join(character for character in folder.name if character.isdigit())
    ),
)

print("\nRaw MRI directory:")
print(RAW_MRI_DIR)

print("\nNumber of patient folders found:")
print(len(patient_folders))

print("\nFirst five patient folders:")
for folder in patient_folders[:5]:
    print("-", folder.name)

print("\nLast five patient folders:")
for folder in patient_folders[-5:]:
    print("-", folder.name)

logger.info(
    "%s patient MRI folders were identified.",
    len(patient_folders),
)

if len(patient_folders) != 60:
    logger.warning(
        "Expected 60 patient folders but found %s.",
        len(patient_folders),
    )

# =========================================================
# INSPECT FIRST PATIENT FOLDER
# =========================================================

first_patient_folder = patient_folders[0]

print("\nInspecting folder:")
print(first_patient_folder)

first_patient_files = sorted(
    [
        item
        for item in first_patient_folder.rglob("*")
        if item.is_file()
    ]
)

print("\nFiles found inside the first patient folder:")

for file_path in first_patient_files:
    relative_path = file_path.relative_to(first_patient_folder)

    print(
        f"- {relative_path} "
        f"[extension: {''.join(file_path.suffixes)}]"
    )

logger.info(
    "%s files were found inside %s.",
    len(first_patient_files),
    first_patient_folder.name,
)

# =========================================================
# CREATE MRI FILE INVENTORY
# =========================================================

inventory_records = []

for patient_folder in patient_folders:
    patient_number_text = "".join(
        character
        for character in patient_folder.name
        if character.isdigit()
    )

    patient_id = (
        int(patient_number_text)
        if patient_number_text
        else None
    )

    patient_files = [
        item
        for item in patient_folder.rglob("*")
        if item.is_file()
    ]

    for file_path in patient_files:
        inventory_records.append({
            "Patient_ID": patient_id,
            "Patient_Folder": patient_folder.name,
            "Relative_Path": str(
                file_path.relative_to(patient_folder)
            ),
            "File_Name": file_path.name,
            "File_Extension": "".join(file_path.suffixes),
            "File_Size_Bytes": file_path.stat().st_size,
        })

mri_inventory_df = pd.DataFrame(inventory_records)

print("\nMRI inventory shape:")
print(mri_inventory_df.shape)

print("\nFile extensions found:")
print(
    mri_inventory_df["File_Extension"]
    .value_counts(dropna=False)
)

print("\nExample inventory records:")
print(mri_inventory_df.head(20).to_string(index=False))

# =========================================================
# SAVE MRI FILE INVENTORY
# =========================================================

mri_inventory_output = (
    TABLES_DIR / "mri_file_inventory.csv"
)

save_dataframe(
    dataframe=mri_inventory_df,
    output_path=mri_inventory_output,
    index=False,
)

logger.info(
    "MRI file inventory saved to %s.",
    mri_inventory_output,
)

print("\nMRI inventory saved to:")
print(mri_inventory_output)

# =========================================================
# DEFINE LESION FEATURE EXTRACTION FUNCTION
# =========================================================

def extract_lesion_features(
    mask_path: Path,
) -> dict:
    """
    Extract patient-level lesion biomarkers from a
    three-dimensional lesion segmentation mask.

    Parameters
    ----------
    mask_path:
        Path to a NIfTI lesion segmentation mask.

    Returns
    -------
    dict
        Dictionary containing lesion volume, lesion count
        and lesion-size distribution features.
    """

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Lesion mask was not found:\n{mask_path}"
        )

    mask_image = nib.load(str(mask_path))
    mask_data = mask_image.get_fdata()

    if mask_data.ndim != 3:
        raise ValueError(
            f"Expected a 3D lesion mask but found "
            f"{mask_data.ndim} dimensions:\n{mask_path}"
        )

    # Convert any non-zero lesion label to a binary mask
    binary_mask = mask_data > 0

    # Voxel dimensions are stored in millimetres
    voxel_dimensions = mask_image.header.get_zooms()[:3]

    voxel_volume_mm3 = float(
        np.prod(voxel_dimensions)
    )

    lesion_voxels = int(binary_mask.sum())

    lesion_volume_mm3 = (
        lesion_voxels * voxel_volume_mm3
    )

    # Use 26-neighbour connectivity in three dimensions
    connectivity_structure = np.ones(
        (3, 3, 3),
        dtype=int,
    )

    labelled_mask, lesion_count = ndimage.label(
        binary_mask,
        structure=connectivity_structure,
    )

    if lesion_count > 0:
        component_sizes = np.bincount(
            labelled_mask.ravel()
        )[1:]

        component_volumes_mm3 = (
            component_sizes * voxel_volume_mm3
        )

        average_lesion_size = float(
            component_volumes_mm3.mean()
        )

        largest_lesion_size = float(
            component_volumes_mm3.max()
        )

        smallest_lesion_size = float(
            component_volumes_mm3.min()
        )

        # Operational size bands used for exploratory analysis
        small_lesions = int(
            (component_volumes_mm3 < 10).sum()
        )

        medium_lesions = int(
            (
                (component_volumes_mm3 >= 10)
                & (component_volumes_mm3 <= 50)
            ).sum()
        )

        large_lesions = int(
            (component_volumes_mm3 > 50).sum()
        )

    else:
        average_lesion_size = 0.0
        largest_lesion_size = 0.0
        smallest_lesion_size = 0.0
        small_lesions = 0
        medium_lesions = 0
        large_lesions = 0

    return {
        "Image_Shape_X": int(mask_data.shape[0]),
        "Image_Shape_Y": int(mask_data.shape[1]),
        "Image_Shape_Z": int(mask_data.shape[2]),
        "Voxel_Size_X_mm": float(voxel_dimensions[0]),
        "Voxel_Size_Y_mm": float(voxel_dimensions[1]),
        "Voxel_Size_Z_mm": float(voxel_dimensions[2]),
        "Voxel_Volume_mm3": voxel_volume_mm3,
        "Lesion_Voxels": lesion_voxels,
        "Lesion_Volume_mm3": lesion_volume_mm3,
        "Lesion_Count": int(lesion_count),
        "Average_Lesion_Size": average_lesion_size,
        "Largest_Lesion_Size": largest_lesion_size,
        "Smallest_Lesion_Size": smallest_lesion_size,
        "Small_Lesions": small_lesions,
        "Medium_Lesions": medium_lesions,
        "Large_Lesions": large_lesions,
    }

# =========================================================
# TEST FEATURE EXTRACTION ON PATIENT 1
# =========================================================

patient_1_flair_mask = (
    RAW_MRI_DIR
    / "Patient-1"
    / "1-LesionSeg-Flair.nii"
)

patient_1_features = extract_lesion_features(
    patient_1_flair_mask
)

print("\nPatient 1 FLAIR lesion features:")

for feature_name, feature_value in patient_1_features.items():
    print(f"- {feature_name}: {feature_value}")

# =========================================================
# VALIDATE MRI AND MASK DIMENSIONS
# =========================================================

def validate_scan_mask_pair(
    scan_path: Path,
    mask_path: Path,
) -> dict:
    """
    Confirm that an MRI scan and its segmentation mask have
    matching dimensions and voxel spacing.
    """

    scan_image = nib.load(str(scan_path))
    mask_image = nib.load(str(mask_path))

    scan_shape = scan_image.shape
    mask_shape = mask_image.shape

    scan_zooms = scan_image.header.get_zooms()[:3]
    mask_zooms = mask_image.header.get_zooms()[:3]

    shapes_match = scan_shape == mask_shape

    voxel_spacing_matches = np.allclose(
        scan_zooms,
        mask_zooms,
        rtol=1e-5,
        atol=1e-5,
    )

    return {
        "Scan_Shape": str(scan_shape),
        "Mask_Shape": str(mask_shape),
        "Shapes_Match": bool(shapes_match),
        "Scan_Voxel_Spacing": str(scan_zooms),
        "Mask_Voxel_Spacing": str(mask_zooms),
        "Voxel_Spacing_Matches": bool(
            voxel_spacing_matches
        ),
    }

# =========================================================
# EXTRACT MRI FEATURES FOR ALL PATIENTS AND MODALITIES
# =========================================================

modalities = {
    "FLAIR": {
        "scan_suffix": "Flair.nii",
        "mask_suffix": "LesionSeg-Flair.nii",
    },
    "T1": {
        "scan_suffix": "T1.nii",
        "mask_suffix": "LesionSeg-T1.nii",
    },
    "T2": {
        "scan_suffix": "T2.nii",
        "mask_suffix": "LesionSeg-T2.nii",
    },
}

all_modality_records = []
validation_records = []
extraction_errors = []

for patient_folder in patient_folders:
    patient_number_text = "".join(
        character
        for character in patient_folder.name
        if character.isdigit()
    )

    patient_id = int(patient_number_text)

    for modality_name, file_patterns in modalities.items():
        scan_path = (
            patient_folder
            / f"{patient_id}-{file_patterns['scan_suffix']}"
        )

        mask_path = (
            patient_folder
            / f"{patient_id}-{file_patterns['mask_suffix']}"
        )

        try:
            if not scan_path.exists():
                raise FileNotFoundError(
                    f"MRI scan not found: {scan_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Lesion mask not found: {mask_path}"
                )

            validation_result = validate_scan_mask_pair(
                scan_path=scan_path,
                mask_path=mask_path,
            )

            validation_records.append({
                "Patient_ID": patient_id,
                "Modality": modality_name,
                "Scan_File": scan_path.name,
                "Mask_File": mask_path.name,
                **validation_result,
            })

            if not validation_result["Shapes_Match"]:
                raise ValueError(
                    "MRI scan and lesion mask shapes do not match."
                )

            if not validation_result[
                "Voxel_Spacing_Matches"
            ]:
                logger.warning(
                    "Voxel spacing mismatch for Patient %s, %s.",
                    patient_id,
                    modality_name,
                )

            lesion_features = extract_lesion_features(
                mask_path
            )

            all_modality_records.append({
                "Patient_ID": patient_id,
                "Modality": modality_name,
                "Scan_File": scan_path.name,
                "Mask_File": mask_path.name,
                **lesion_features,
            })

        except Exception as error:
            logger.exception(
                "Feature extraction failed for Patient %s, %s.",
                patient_id,
                modality_name,
            )

            extraction_errors.append({
                "Patient_ID": patient_id,
                "Modality": modality_name,
                "Error_Type": type(error).__name__,
                "Error_Message": str(error),
            })

# =========================================================
# CREATE MRI FEATURE DATAFRAMES
# =========================================================

mri_features_all_df = pd.DataFrame(
    all_modality_records
)

mri_validation_df = pd.DataFrame(
    validation_records
)

mri_errors_df = pd.DataFrame(
    extraction_errors
)

print("\nAll-modality MRI feature dataset shape:")
print(mri_features_all_df.shape)

print("\nRecords by modality:")
print(
    mri_features_all_df["Modality"]
    .value_counts()
)

print("\nScan-mask shape mismatches:")
print(
    (
        ~mri_validation_df["Shapes_Match"]
    ).sum()
)

print("\nScan-mask voxel-spacing mismatches:")
print(
    (
        ~mri_validation_df["Voxel_Spacing_Matches"]
    ).sum()
)

print("\nFeature-extraction errors:")
print(len(mri_errors_df))

# =========================================================
# CREATE PRIMARY FLAIR FEATURE DATASET
# =========================================================

mri_features_flair_df = (
    mri_features_all_df.loc[
        mri_features_all_df["Modality"] == "FLAIR"
    ]
    .copy()
    .sort_values("Patient_ID")
    .reset_index(drop=True)
)

print("\nFLAIR feature dataset shape:")
print(mri_features_flair_df.shape)

print("\nFLAIR lesion feature summary:")
print(
    mri_features_flair_df[
        [
            "Lesion_Volume_mm3",
            "Lesion_Count",
            "Average_Lesion_Size",
            "Largest_Lesion_Size",
        ]
    ].describe()
)

# =========================================================
# VALIDATE LESION SIZE GROUP COUNTS
# =========================================================

mri_features_all_df[
    "Lesion_Size_Group_Total"
] = (
    mri_features_all_df["Small_Lesions"]
    + mri_features_all_df["Medium_Lesions"]
    + mri_features_all_df["Large_Lesions"]
)

size_group_mismatches = (
    mri_features_all_df[
        "Lesion_Size_Group_Total"
    ]
    != mri_features_all_df["Lesion_Count"]
)

print("\nLesion-size group count mismatches:")
print(size_group_mismatches.sum())

if size_group_mismatches.any():
    logger.warning(
        "%s lesion-size group count mismatches were found.",
        size_group_mismatches.sum(),
    )

# =========================================================
# SAVE MRI FEATURE DATASETS
# =========================================================

all_modalities_output = (
    PROCESSED_DATA_DIR
    / "mri_features_all_modalities.csv"
)

flair_output = (
    PROCESSED_DATA_DIR
    / "mri_features_flair.csv"
)

validation_output = (
    TABLES_DIR
    / "mri_scan_mask_validation.csv"
)

errors_output = (
    TABLES_DIR
    / "mri_feature_extraction_errors.csv"
)

save_dataframe(
    dataframe=mri_features_all_df,
    output_path=all_modalities_output,
    index=False,
)

save_dataframe(
    dataframe=mri_features_flair_df,
    output_path=flair_output,
    index=False,
)

save_dataframe(
    dataframe=mri_validation_df,
    output_path=validation_output,
    index=False,
)

if not mri_errors_df.empty:
    save_dataframe(
        dataframe=mri_errors_df,
        output_path=errors_output,
        index=False,
    )

logger.info(
    "MRI feature extraction completed successfully."
)

logger.info(
    "All-modality records extracted: %s.",
    len(mri_features_all_df),
)

logger.info(
    "FLAIR records extracted: %s.",
    len(mri_features_flair_df),
)

logger.info(
    "Extraction errors: %s.",
    len(mri_errors_df),
)

print_section("MRI Feature Extraction Complete")

print("\nAll-modalities dataset saved to:")
print(all_modalities_output)

print("\nFLAIR dataset saved to:")
print(flair_output)

print("\nMRI validation table saved to:")
print(validation_output)

