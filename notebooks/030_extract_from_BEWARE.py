# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src.settings import DATA_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR

# Paths
directory_path = RAW_DIR / "external/COWCLIP"
print("Using data directory:", DATA_DIR)

# Inputs
atoll_inputs_path = INTERIM_DIR / "Atoll_BEWARE_inputs.parquet"
BEWARE_extended_path = INTERIM_DIR / "BEWARE_Database_extended_avg.nc"
# Outputs
results_csv = PROCESSED_DIR / "Atoll_BEWARE_processed_outputs.csv"
results_parquet = PROCESSED_DIR / "Atoll_BEWARE_processed_outputs.parquet"


# %% [markdown]
# ### BEWARE Matching Script
#
# This script processes coastal flood hazard inputs by matching them against 
# the **extended BEWARE framework database**.
# The workflow identifies the best-fitting **R2pIndex** values for given hydrodynamic and geomorphic parameters across transects and scenarios.
#
# #### Key Steps
# 1. **Load BEWARE Database**:
#    Opens the extended BEWARE NetCDF database and converts it into a structured NumPy array for fast matching.
#
# 2. **Filter Inputs**:
#    Reads pre-processed transect-level input data (`Atoll_BEWARE_inputs.parquet`) and restricts it to relevant confidence–scenario combinations (e.g., `"medium-ssp585"`, `"low-ssp585"`).
#
# 3. **Efficient Matching**:
#    - For each row, eta values are pre-filtered within a ±0.1 range to reduce computation.
#    - A normalized distance score is calculated across parameters (`W_reef`, `beta_f`, `H0`, `H0L0`, `eta0`).
#    - The closest match’s **R2pIndex** is selected.
#
# 4. **Batch Processing**:
#    Vectorized functions and progress bars (`tqdm`) are used for efficient row-wise application.
#
# 5. **Output Results**:
#    Matched results are merged with metadata and saved into both `.parquet` and `.csv` formats:
#    - `Atoll_BEWARE_processed_outputs_2020-2150.parquet`
#    - `Atoll_BEWARE_processed_outputs_2020-2150.csv`
#
# #### Usage
# - Suggested to run as (`pixi run python notebooks/030_extract_from_BEWARE.py`) from project-root.
# - Designed for **high-volume computations**; input filtering ensures speed and avoids unnecessary calculations.
#
# #### Notes
# - Requires local access to the **extended BEWARE NetCDF database** (`BEWARE_Database_extended_avg.nc`).
# - Progress and runtime are printed during execution for monitoring.
#


# %%
def match_with_eta_vectorized(W_reef, beta_f, H0, H0L0, eta_value):
    """
    Find the best matching R2pIndex in beware_array based on differences
    of given parameters (W_reef, beta_f, H0, H0L0, eta_value), prioritizing H0, eta, and W_reef.
    """
    # Filter valid entries
    valid = beware_array[
        (beware_array.W_reef > 1)
        & (np.isclose(beware_array.Cf, 0.05))
        & (np.isclose(beware_array.beta_Beach, 0.10))
    ]
    if len(valid) == 0:
        return np.nan

    # Apply eta filter +- 0.1
    eta_min = eta_value - 0.1
    eta_max = eta_value + 0.1
    valid = valid[(valid.eta0 >= eta_min) & (valid.eta0 <= eta_max)]

    if len(valid) == 0:
        return np.nan

    # Calculate normalized distances for each parameter to score matches
    d_beta = np.abs(valid.beta_ForeReef - beta_f)
    d_H0 = np.abs(valid.H0 - H0)
    d_H0L0 = np.abs(valid.H0L0 - H0L0)
    d_eta = np.abs(valid.eta0 - eta_value)
    d_Wreef = np.abs(valid.W_reef - W_reef)

    scores = (
        d_beta / (np.max(d_beta) + 1e-6)
        + d_H0 / (np.max(d_H0) + 1e-6)
        + d_H0L0 / (np.max(d_H0L0) + 1e-6)
        + d_eta / (np.max(d_eta) + 1e-6)
        + d_Wreef / (np.max(d_Wreef) + 1e-6)
    )
    # Return R2pIndex with minimal combined distance
    return valid.R2pIndex[np.argmin(scores)]


# %%
# Map eta columns to their respective output column names
eta_dict = {
    "eta_combined_rp1": "R2pIndex_combined_rp1",
    "eta_combined_rp10": "R2pIndex_combined_rp10",
    "eta_combined_rp100": "R2pIndex_combined_rp100",
}


def apply_all_matches(row):
    """
    For each row, match all eta values and return corresponding R2pIndex results.
    """
    results = {
        "transect_id": row["transect_id"],
        "year": row["year"],
        "confidence": row["confidence"],
        "scenario": row["scenario"],
        "quantile": row["quantile"],
        "confidence": row["confidence"],
        "FID_GADM": row["FID_GADM"],
    }
    for eta_col, output_col in eta_dict.items():
        results[output_col] = match_with_eta_vectorized(
            row["W_reef"], row["beta_f"], row["H0"], row["H0L0"], row[eta_col]
        )
    return pd.Series(results)


# %%
def main():
    global beware_array

    # Determine script directory to load data relative to script location
    home_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = "/hdrive/all_users/moeller/MyDocuments/atoll-slr-paper/data/processed/"

    # Load BEWARE NetCDF database and convert to structured numpy array
    ds = nc.Dataset(BEWARE_extended_path)

    beware_array = np.rec.fromarrays(
        [
            ds.variables["W_reef"][:],
            ds.variables["beta_ForeReef"][:],
            ds.variables["H0"][:],
            ds.variables["eta0"][:],
            ds.variables["R2pIndex"][:],
            ds.variables["Cf"][:],
            ds.variables["beta_Beach"][:],
            ds.variables["H0L0"][:],
        ],
        names="W_reef,beta_ForeReef,H0,eta0,R2pIndex,Cf,beta_Beach,H0L0",
    )

    # Load input data (subset for speed/testing)
    inputs = pd.read_parquet(atoll_inputs_path)

    # Filter inputs for allowed confidence-scenario combinations
    allowed_combos = [
        ("medium", "baseline"),
        ("medium", "ssp585"),
        ("medium", "ssp245"),
        ("medium", "ssp370"),
        ("medium", "ssp126"),
        ("medium", "ssp119"),
        ("low", "ssp585"),
    ]

    inputs = inputs[
        inputs[["confidence", "scenario"]].apply(tuple, axis=1).isin(allowed_combos)
    ]

    # Apply matching function with progress bar
    tqdm.pandas()
    start = time.time()
    results = inputs.progress_apply(apply_all_matches, axis=1)
    print(f"Completed matching in {time.time() - start:.2f} seconds.")

    # Merge matched results back with input metadata
    results["transect_id"] = results["transect_id"].astype(int)
    outputs = pd.merge(
        inputs[
            ["transect_id", "FID_GADM", "year", "scenario", "confidence", "quantile"]
        ],
        results,
        on=["transect_id", "FID_GADM", "year", "scenario", "confidence", "quantile"],
        how="left",
    )

    outputs.round(2).to_parquet(
        results_parquet,
        index=False,
        engine="pyarrow",
    )
    outputs.round(2).to_csv(
        results_csv,
        index=False,
    )
    print("Saved output files.")
    print(
        "Started with "
        + str(len(inputs))
        + " rows, calculated "
        + str(len(outputs))
        + " rows."
    )


# %%
if __name__ == "__main__":
    main()
