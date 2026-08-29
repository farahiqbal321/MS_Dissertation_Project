"""
Reusable utility functions for the MS dissertation project.

These functions support file validation, dataset checking,
figure saving, table saving and reproducible project logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# =========================================================
# LOGGING
# =========================================================

def setup_logger(
    logger_name: str,
    log_file: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes messages to both the terminal
    and a log file.

    Parameters
    ----------
    logger_name:
        Name used to identify the logger.

    log_file:
        Path where the log file will be saved.

    level:
        Logging level, such as logging.INFO or logging.DEBUG.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers if the function is called again
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =========================================================
# FILE VALIDATION
# =========================================================

def check_file_exists(file_path: Path) -> None:
    """
    Raise a clear error if a required file does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )


def check_directory_exists(directory_path: Path) -> None:
    """
    Raise a clear error if a required directory does not exist.
    """

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Required directory was not found:\n{directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"The supplied path is not a directory:\n{directory_path}"
        )


# =========================================================
# DATASET VALIDATION
# =========================================================

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Confirm that a dataframe contains all required columns.
    """

    required_columns = list(required_columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def report_dataset_quality(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Create a summary table describing data types, missing values
    and unique values for every column.
    """

    quality_report = pd.DataFrame({
        "Column": dataframe.columns,
        "Data_Type": dataframe.dtypes.astype(str).values,
        "Missing_Values": dataframe.isna().sum().values,
        "Missing_Percentage": (
            dataframe.isna().mean().mul(100).round(2).values
        ),
        "Unique_Values": dataframe.nunique(dropna=False).values,
    })

    print(f"\n{dataset_name}")
    print("=" * len(dataset_name))
    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(f"Duplicate rows: {dataframe.duplicated().sum()}")
    print(f"Total missing values: {dataframe.isna().sum().sum()}")

    return quality_report


# =========================================================
# SAVING OUTPUTS
# =========================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
    index: bool = False,
) -> None:
    """
    Save a dataframe as CSV and create its parent folder
    automatically if required.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        output_path,
        index=index,
    )

    print(f"Table saved to:\n{output_path}")


def save_figure(
    output_path: Path,
    dpi: int = 300,
    close_figure: bool = True,
) -> None:
    """
    Save the currently active Matplotlib figure.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(f"Figure saved to:\n{output_path}")

    if close_figure:
        plt.close()


# =========================================================
# FORMATTING
# =========================================================

def print_section(title: str) -> None:
    """
    Print a clear terminal section heading.
    """

    separator = "=" * 70

    print(f"\n{separator}")
    print(title.upper())
    print(separator)

