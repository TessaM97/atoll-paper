# %%
# %%
# ── BEWARE matching filter half-widths ────────────────────────────────────────
# Derived from p99 distance of input values to the nearest BEWARE grid point.
# Found by 014_a_find_optimal_eta_H0_H0L0_distances.ipynb
import json
import os
import time
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

out_dir = "/hdrive/all_users/moeller/MyDocuments/atoll-slr-paper/data/processed/"
config_path = out_dir + "beware_matching_config.json"
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

    home_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = "/hdrive/all_users/moeller/MyDocuments/atoll-slr-paper/data/processed/"
    data_dir = os.path.dirname(out_dir.rstrip("/"))  # → .../atoll-slr-paper/data

    # ── Load BEWARE database ──────────────────────────────────────────────────
    beware_nc_path = os.path.join(data_dir, "BEWARE_Database_extended_v4.nc")
    ds = nc.Dataset(beware_nc_path)
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
    _input_candidates = sorted(
        Path(out_dir).glob("Atoll_BEWARE_inputs_*.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if not _input_candidates:
        raise FileNotFoundError(f"No Atoll_BEWARE_inputs_*.parquet found in {out_dir}")
    inputs_path = _input_candidates[-1]
    print(f"Using inputs file        : {inputs_path.name}")
    inputs = pd.read_parquet(inputs_path)
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
        delayed(apply_all_matches)(row, beware_valid) for row in tqdm(rows, mininterval=60)
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
    out_parquet = os.path.join(
        out_dir, f"Atoll_BEWARE_processed_outputs_2020-2150_{date_str}.parquet"
    )
    out_csv = os.path.join(
        out_dir, f"Atoll_BEWARE_processed_outputs_2020-2150_{date_str}.csv"
    )

    outputs.round(2).to_parquet(out_parquet, index=False, engine="pyarrow")
    outputs.round(2).to_csv(out_csv, index=False)

    print(f"Saved {len(outputs):,} rows → {out_parquet}")


# %%
if __name__ == "__main__":
    main()
