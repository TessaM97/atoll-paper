# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: atoll_slr_paper
#     language: python
#     name: atoll_slr_paper
# ---

# %%
# ── BEWARE matching filter half-widths ────────────────────────────────────────
# Derived from p99 distance of input values to the nearest BEWARE grid point.
# Found by 021_find_optimal_eta_H0_H0L0_distances.py
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src.settings import DATA_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Auto-detect the most recently modified inputs parquet
_input_candidates = sorted(
    INTERIM_DIR.glob("Atoll_BEWARE_inputs_*.parquet"),
    key=lambda p: p.stat().st_mtime,
)
if not _input_candidates:
    raise FileNotFoundError(f"No Atoll_BEWARE_inputs_*.parquet found in {INTERIM_DIR}")
atoll_inputs_path = _input_candidates[-1]
print(f"Using inputs file        : {atoll_inputs_path.name}")
BEWARE_extended_path = PROCESSED_DIR / "BEWARE_Database_extended_v4.nc"
config_path = PROCESSED_DIR / "beware_matching_config.json"

# ── Load filter half-widths from config (produced by 021) ───────────────────
with open(config_path) as f:
    cfg = json.load(f)

ETA_FILTER = cfg["filter_halfwidths"]["ETA_FILTER"]
H0_FILTER = cfg["filter_halfwidths"]["H0_FILTER"]
H0L0_FILTER = cfg["filter_halfwidths"]["H0L0_FILTER"]

print(
    f"Loaded filter halfwidths: eta=±{ETA_FILTER}, H0=±{H0_FILTER}, H0L0=±{H0L0_FILTER}"
)

# beware_valid is set once in main() and used by the matching function
beware_valid = None


# %% [markdown]
# ### BEWARE Matching Script
#
# This script processes coastal flood hazard inputs by matching them against
# the **extended BEWARE framework database**.
# The workflow identifies the best-fitting **R2pIndex** values for given
# hydrodynamic and geomorphic parameters across transects and scenarios.
#
# #### Key Steps
# 1. **Load BEWARE Database**:
#    Opens the extended BEWARE NetCDF database (`BEWARE_Database_extended_v4.nc`)
#    and pre-filters to valid entries (Cf=0.05, beta_Beach=0.10, W_reef>1) once.
#
# 2. **Filter Inputs**:
#    Reads pre-processed transect-level input data and restricts it to relevant
#    confidence–scenario combinations.
#
# 3. **Efficient Matching**:
#    - For each row, candidates are pre-filtered using tight windows around
#      eta, H0, H0L0 (widths from `beware_matching_config.json`).
#    - Falls back to relaxing the H0L0 filter if no candidates are found.
#    - A normalized distance score is calculated across all parameters.
#    - The closest match's **R2pIndex** is selected.
#
# 4. **Batch Processing**:
#    joblib Parallel (256 workers) is used for efficient row-wise application.
#
# 5. **Output Results**:
#    Matched results are merged with metadata and saved into both `.parquet`
#    and `.csv` formats with a datestamp suffix.
#
# #### Usage
# - Suggested to run as (`pixi run python notebooks/030_extract_from_BEWARE.py`)
#   from project-root.
# - Requires `beware_matching_config.json` produced by
#   `021_find_optimal_eta_H0_H0L0_distances.py`.


# %%
def match_with_eta_vectorized(W_reef, beta_f, H0, H0L0, eta_value, bv):
    """
    Return the best-matching R2pIndex from bv (pre-filtered beware_valid).

    Candidates are pre-filtered by tight windows around eta, H0, H0L0
    then scored by sum of normalised distances across all five parameters.
    Falls back to relaxing the H0L0 filter if no candidates are found.
    """
    valid = bv[
        (np.abs(bv.eta0 - eta_value) <= ETA_FILTER)
        & (np.abs(bv.H0 - H0) <= H0_FILTER)
        & (np.abs(bv.H0L0 - H0L0) <= H0L0_FILTER)
    ]

    if len(valid) == 0:
        # Fallback: relax H0L0 filter
        valid = bv[
            (np.abs(bv.eta0 - eta_value) <= ETA_FILTER)
            & (np.abs(bv.H0 - H0) <= H0_FILTER)
        ]
    if len(valid) == 0:
        return np.nan

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

    return valid["R2pIndex"].iloc[np.argmin(scores)]


# %%
# Map eta columns to their respective output column names
eta_dict = {
    "eta_combined_rp1": "R2pIndex_combined_rp1",
    "eta_combined_rp10": "R2pIndex_combined_rp10",
    "eta_combined_rp100": "R2pIndex_combined_rp100",
}


def apply_all_matches(row, bv):
    """Apply matching for all three eta return periods (rp1, rp10, rp100)."""
    results = {
        "transect_id": row["transect_id"],
        "year": row["year"],
        "confidence": row["confidence"],
        "scenario": row["scenario"],
        "quantile": row["quantile"],
        "FID_GADM": row["FID_GADM"],
    }
    for eta_col, output_col in eta_dict.items():
        results[output_col] = match_with_eta_vectorized(
            row["W_reef"], row["beta_f"], row["H0"], row["H0L0"], row[eta_col], bv
        )
    return pd.Series(results)


# %%
def main():
    global beware_valid

    # ── Load BEWARE database ──────────────────────────────────────────────────
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
    print(f"BEWARE loaded            : {len(beware_array):,} entries")

    # Pre-filter once — avoids repeated filtering per function call (~5000x speedup)
    beware_df = pd.DataFrame(beware_array)
    beware_valid = beware_df[
        (beware_df.W_reef > 1)
        & (np.isclose(beware_df.Cf, 0.05))
        & (np.isclose(beware_df.beta_Beach, 0.10))
    ].reset_index(drop=True)
    print(f"After Cf/beta_Beach filter: {len(beware_valid):,} entries")

    # ── Load and filter inputs ────────────────────────────────────────────────
    inputs = pd.read_parquet(atoll_inputs_path)
    print(
        f"Inputs loaded            : {len(inputs):,} rows  |  "
        f"{inputs['transect_id'].nunique():,} transects"
    )

    allowed_combos = [
        ("medium", "baseline"),
        ("medium", "ssp119"),
        ("medium", "ssp126"),
        ("medium", "ssp245"),
        ("medium", "ssp370"),
        ("medium", "ssp585"),
        ("low", "ssp585"),
    ]
    inputs = inputs[
        inputs[["confidence", "scenario"]].apply(tuple, axis=1).isin(allowed_combos)
    ]
    print(f"After scenario filter    : {len(inputs):,} rows")

    # ── Run matching ──────────────────────────────────────────────────────────
    N_JOBS = 256
    rows = [row for _, row in inputs.iterrows()]
    print(f"Running matching with {N_JOBS} parallel workers on {len(rows):,} rows...")
    start = time.time()
    results_list = Parallel(n_jobs=N_JOBS)(
        delayed(apply_all_matches)(row, beware_valid)
        for row in tqdm(rows, mininterval=60)
    )
    elapsed = time.time() - start
    results = pd.DataFrame(results_list)
    print(f"Completed matching in {elapsed:.1f} seconds  ({elapsed/60:.1f} minutes)")

    # ── Merge and save ────────────────────────────────────────────────────────
    merge_cols = [
        "transect_id",
        "FID_GADM",
        "year",
        "scenario",
        "confidence",
        "quantile",
    ]
    results["transect_id"] = results["transect_id"].astype(int)
    outputs = pd.merge(
        inputs[merge_cols],
        results,
        on=merge_cols,
        how="left",
    )

    # Check for unmatched rows
    n_nan = outputs["R2pIndex_combined_rp1"].isna().sum()
    if n_nan > 0:
        print(f"WARNING: {n_nan:,} rows with NaN R2pIndex — check input ranges")

    date_str = datetime.now().strftime("%d%m%Y")
    out_parquet = PROCESSED_DIR / f"Atoll_BEWARE_processed_outputs_2020-2150_{date_str}.parquet"
    out_csv = PROCESSED_DIR / f"Atoll_BEWARE_processed_outputs_2020-2150_{date_str}.csv"

    outputs.round(2).to_parquet(out_parquet, index=False, engine="pyarrow")
    outputs.round(2).to_csv(out_csv, index=False)

    print(f"Saved {len(outputs):,} rows → {out_parquet}")


# %%
if __name__ == "__main__":
    main()
