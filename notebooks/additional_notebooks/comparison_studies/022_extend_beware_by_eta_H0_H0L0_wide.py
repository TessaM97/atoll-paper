# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: atoll_slr_paper
#     language: python
#     name: atoll_slr_paper
# ---

# %% [markdown]
# # BEWARE Database — Average, Interpolate, Extrapolate (WIDE, local)
#
# Local re-run of `atoll-paper/notebooks/022_extend_beware_by_eta_H0_H0L0.py` (the script that
# produced `BEWARE_Database_extended_v4.nc`), with **wider manually-set ranges** instead of the
# auto-computed "optimal" ranges from `021_find_optimal_eta_H0_H0L0_distances.py`.
#
# **Why:** `021` set the v4 extension ranges (`H0: 1.0-3.2`, `eta0: 0.15-7.6`,
# `H0L0: 0.007-0.029`) to match only the range of values seen across the main scenario pipeline's
# 5,784 transects — it was never meant to cover the literature validation events
# (`Dataset_S5_Comparison_BEWARE_runup_with_other_studies.xlsx`), which include larger storm wave
# heights (up to 5.9 m) and very low water levels (down to 0.05 m).
#
# **Algorithm is unchanged** from the server script (average duplicates → interpolate/extrapolate
# eta → interpolate H0 → interpolate H0L0 → save NetCDF); only `ETA_MIN`, `H0_MAX`, and `H0L0_MIN`/
# `H0L0_MAX` are widened, clamped to the **raw** BEWARE grid's own bounds:
#
# | | v4 (old, narrow) | v4_wide (this notebook) | raw grid bound |
# |---|---|---|---|
# | `H0`   | 1.0 – 3.2 | 1.0 – **5.0** | 1.0 – 5.0 (hard limit — no H0 extrapolation in the algorithm) |
# | `eta0` | 0.15 – 7.6 | **-1.0** – 7.6 (widened, extrapolated as before) | -1.0 – 3.0 raw, extrapolated beyond |
# | `H0L0` | 0.007 – 0.029 | **0.005** – **0.05** | 0.005 – 0.05 (hard limit — no H0L0 extrapolation) |
#
# `H0` and `H0L0` cannot be extrapolated past the raw grid (the interpolation step clips, it does
# not extrapolate) — so **Nui (H0 = 5.9 m) is still out of reach**; the raw model's own H0 ceiling
# is 5.0 m. Fixing that would require new BEWARE hydrodynamic model runs on the HPC, not just a
# wider local interpolation.
#
# **Consistency check:** all three step sizes (`ETA_STEP=0.05`, `H0_STEP=0.1`, `H0L0_STEP=0.001`)
# are kept identical to the original, and the new lower bounds are exact multiples of those steps
# below the old bounds (`0.15 - (-1.0) = 1.15 = 23×0.05`; `0.029 - 0.05`... `0.007 - 0.005 =
# 0.002 = 2×0.001`) — so the wide grid's overlapping region reproduces the *exact same grid points*
# as `v4.nc`, computed deterministically from the same source data.

# %%
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

try:
    project_root = Path(__file__).resolve().parents[3]
except NameError:
    project_root = Path().resolve().parents[2]  # Jupyter: cwd is notebooks/additional_notebooks/comparison_studies/
sys.path.append(str(project_root))
from src.settings import PROCESSED_DIR, RAW_DIR

# %%
# ── PATHS ─────────────────────────────────────────────────────────────────────
FILE_PATH = RAW_DIR / "external/BEWARE/BEWARE_Database.nc"
OUTPUT_PATH = PROCESSED_DIR / "BEWARE_Database_extended_v4_wide.nc"

# ── CONSTANTS (unchanged from the server script) ──────────────────────────────
CF_FIXED = 0.05
BETA_BEACH_FIXED = 0.10
GROUP_COLS = ["H0L0", "W_reef", "beta_ForeReef", "H0"]

ETA_STEP = 0.05
H0_STEP = 0.1
H0L0_STEP = 0.001

# ── WIDENED RANGES (manually set, clamped to raw grid bounds) ────────────────
# was: ETA_MIN=0.15, H0_MAX=3.2, H0L0_MIN=0.007, H0L0_MAX=0.029 (from beware_matching_config.json)
ETA_MIN = -1.0
ETA_MAX = 7.6      # unchanged -- already covers all 6 validation studies' eta values
H0_MIN = 1.0        # unchanged -- raw grid floor
H0_MAX = 5.0         # widened to raw grid ceiling (was 3.2)
H0L0_MIN = 0.005     # widened to raw grid floor (was 0.007)
H0L0_MAX = 0.05      # widened to raw grid ceiling (was 0.029)

print("── Extension ranges ──")
print(f"  eta  : {ETA_MIN} – {ETA_MAX} m")
print(f"  H0   : {H0_MIN} – {H0_MAX} m")
print(f"  H0L0 : {H0L0_MIN} – {H0L0_MAX}")

FILL_VALUE = -9999.0
N_EXAMPLES = 6
ETA_EXAMPLE = 0.70

# %% [markdown]
# ## 1. Load raw BEWARE NetCDF and filter to Cf=0.05, beta_Beach=0.10

# %%
ds = nc.Dataset(FILE_PATH)


def safe_var(ds, var_name):
    var = ds.variables[var_name]
    data = var[:]
    fv = var._FillValue if "_FillValue" in var.ncattrs() else None
    if fv is not None:
        data = np.where(data == fv, np.nan, data)
    return np.array(data, dtype=float)


df_raw = pd.DataFrame(
    {
        "H0L0": safe_var(ds, "H0L0"),
        "W_reef": safe_var(ds, "W_reef"),
        "beta_ForeReef": safe_var(ds, "beta_ForeReef"),
        "H0": safe_var(ds, "H0"),
        "Cf": safe_var(ds, "Cf"),
        "beta_Beach": safe_var(ds, "beta_Beach"),
        "eta": safe_var(ds, "eta0"),
        "R2p": safe_var(ds, "R2pIndex"),
    }
).dropna()

print(f"Raw rows loaded : {len(df_raw):,}")

df = (
    df_raw[
        np.isclose(df_raw["Cf"], CF_FIXED)
        & np.isclose(df_raw["beta_Beach"], BETA_BEACH_FIXED)
    ]
    .drop(columns=["Cf", "beta_Beach"])
    .copy()
)

print(f"After Cf={CF_FIXED}, beta_Beach={BETA_BEACH_FIXED} filter : {len(df):,} rows")
print(f"Unique groups  : {df.groupby(GROUP_COLS).ngroups:,}")
print(f"Unique H0      : {sorted(df['H0'].unique())}")
print(f"Unique H0L0    : {sorted(df['H0L0'].unique())}")

# %% [markdown]
# ## 2. Average 4 duplicate entries per (group + eta)

# %%
df_avg = df.groupby(GROUP_COLS + ["eta"], as_index=False)["R2p"].mean()
print(f"Rows after averaging : {len(df_avg):,}")
print(f"Duplicates remaining : {df_avg.duplicated(subset=GROUP_COLS + ['eta']).sum()}")

all_group_keys = list(df_avg.groupby(GROUP_COLS).groups.keys())
step = max(1, len(all_group_keys) // N_EXAMPLES)
example_keys = all_group_keys[::step][:N_EXAMPLES]


def get_group(df, keys, cols=GROUP_COLS, sort_col="eta"):
    mask = np.ones(len(df), dtype=bool)
    for col, val in zip(cols, keys):
        mask &= np.isclose(df[col], val)
    return df[mask].sort_values(sort_col)


def group_label(keys):
    return f"H0L0={keys[0]:.3f} W={keys[1]:.0f} " f"\u03b2F={keys[2]:.2f} H0={keys[3]:.2f}"

# %% [markdown]
# ## 3. Interpolate eta to 0.05 m grid; extrapolate beyond original range
#
# Same per-group global-slope-anchored-at-boundary extrapolation as the server script.

# %%
eta_grid = np.round(np.arange(ETA_MIN, ETA_MAX + ETA_STEP / 2, ETA_STEP), 2)
print(f"Eta grid: {len(eta_grid)} points from {eta_grid[0]} to {eta_grid[-1]}")

interp_rows = []
for group_keys, group_df in df_avg.groupby(GROUP_COLS):
    grp = group_df.sort_values("eta")
    eta_vals = grp["eta"].values
    r2p_vals = grp["R2p"].values
    if len(eta_vals) < 2:
        continue

    eta_lo = eta_vals.min()
    eta_hi = eta_vals.max()

    slope, _ = np.polyfit(eta_vals, r2p_vals, 1)

    for eta_val in eta_grid:
        if eta_val > eta_hi:
            r2p = r2p_vals[-1] + slope * (eta_val - eta_hi)
        elif eta_val < eta_lo:
            r2p = r2p_vals[0] + slope * (eta_val - eta_lo)
        else:
            r2p = float(np.interp(eta_val, eta_vals, r2p_vals))
        interp_rows.append(dict(zip(GROUP_COLS, group_keys), eta=eta_val, R2p=r2p))

df_interp = pd.DataFrame(interp_rows)
print(f"Rows after eta interpolation/extrapolation : {len(df_interp):,}")

# %% [markdown]
# ## 4. Interpolate H0 to 0.1 m resolution (independent dimension, clipped -- not extrapolated)

# %%
NON_H0_COLS = ["H0L0", "W_reef", "beta_ForeReef"]

H0_orig = np.array(sorted(df_interp["H0"].unique()))
H0_grid = np.round(np.arange(H0_MIN, H0_MAX + H0_STEP / 2, H0_STEP), 1)
print(f"H0 original ({len(H0_orig)} values) : {H0_orig.tolist()}")
print(
    f"H0 output grid ({len(H0_grid)} values, clipped to {H0_MIN}\u2013{H0_MAX}) : "
    f"{H0_grid.tolist()}"
)

df_pivot_h0 = df_interp.pivot_table(
    index=NON_H0_COLS + ["eta"], columns="H0", values="R2p", aggfunc="mean"
)
R2p_h0 = df_pivot_h0.values
idx_df_h0 = df_pivot_h0.index.to_frame(index=False)
print(f"Pivot shape : {df_pivot_h0.shape}")

frames_h0 = []
for h0_val in H0_grid:
    idx = np.clip(
        np.searchsorted(H0_orig, h0_val, side="right") - 1, 0, len(H0_orig) - 2
    )
    frac = np.clip(
        (h0_val - H0_orig[idx]) / (H0_orig[idx + 1] - H0_orig[idx] + 1e-12), 0, 1
    )
    r2p = R2p_h0[:, idx] + frac * (R2p_h0[:, idx + 1] - R2p_h0[:, idx])
    frame = idx_df_h0.copy()
    frame["H0"] = h0_val
    frame["R2p"] = r2p
    frames_h0.append(frame)

df_h0_interp = pd.concat(frames_h0, ignore_index=True)
print(f"Rows after H0 interpolation : {len(df_h0_interp):,}")
print(f"Unique H0 values            : {sorted(df_h0_interp['H0'].unique())}")

# %% [markdown]
# ## 5. Interpolate H0L0 to 0.001 resolution (independent dimension, clipped -- not extrapolated)

# %%
NON_H0L0_COLS = ["H0", "W_reef", "beta_ForeReef"]

H0L0_orig = np.array(sorted(df_h0_interp["H0L0"].unique()))
print(f"H0L0 original ({len(H0L0_orig)} values) : {H0L0_orig.tolist()}")

H0L0_grid = np.round(np.arange(H0L0_MIN, H0L0_MAX + H0L0_STEP / 2, H0L0_STEP), 4)
print(
    f"H0L0 output grid ({len(H0L0_grid)} values, clipped to {H0L0_MIN}\u2013{H0L0_MAX}) : "
    f"from {H0L0_grid[0]} to {H0L0_grid[-1]}"
)

df_pivot_h0l0 = df_h0_interp.pivot_table(
    index=NON_H0L0_COLS + ["eta"], columns="H0L0", values="R2p", aggfunc="mean"
)
R2p_h0l0 = df_pivot_h0l0.values
idx_df_h0l0 = df_pivot_h0l0.index.to_frame(index=False)
print(f"Pivot shape : {df_pivot_h0l0.shape}")

frames_h0l0 = []
for h0l0_val in H0L0_grid:
    idx = np.clip(
        np.searchsorted(H0L0_orig, h0l0_val, side="right") - 1, 0, len(H0L0_orig) - 2
    )
    frac = np.clip(
        (h0l0_val - H0L0_orig[idx]) / (H0L0_orig[idx + 1] - H0L0_orig[idx] + 1e-12),
        0,
        1,
    )
    r2p = R2p_h0l0[:, idx] + frac * (R2p_h0l0[:, idx + 1] - R2p_h0l0[:, idx])
    frame = idx_df_h0l0.copy()
    frame["H0L0"] = h0l0_val
    frame["R2p"] = r2p
    frames_h0l0.append(frame)

df_h0l0_interp = pd.concat(frames_h0l0, ignore_index=True)
print(f"Rows after H0L0 interpolation : {len(df_h0l0_interp):,}")

# %% [markdown]
# ## 6. Save as NetCDF

# %%
df_save = df_h0l0_interp.rename(columns={"eta": "eta0", "R2p": "R2pIndex"})

df_save["Cf"] = CF_FIXED
df_save["beta_Beach"] = BETA_BEACH_FIXED

df_save["eta0"] = df_save["eta0"].round(2)
df_save["H0"] = df_save["H0"].round(1)
df_save["H0L0"] = df_save["H0L0"].round(4)
df_save = df_save.fillna(FILL_VALUE)

NC_VARS = ["H0", "H0L0", "eta0", "Cf", "beta_ForeReef", "beta_Beach", "W_reef", "R2pIndex"]

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

date_created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

with nc.Dataset(OUTPUT_PATH, "w", format="NETCDF4") as ds_new:
    ds_new.createDimension("ID", len(df_save))
    for var in NC_VARS:
        v = ds_new.createVariable(var, "f8", ("ID",), fill_value=FILL_VALUE)
        v[:] = df_save[var].values.astype("f8")
        v.units = "unknown"
    ds_new.description = (
        f"BEWARE database filtered to Cf={CF_FIXED}, beta_Beach={BETA_BEACH_FIXED}. "
        f"Averaged 4 duplicates per group+eta. "
        f"Interpolated: eta to {ETA_STEP} m grid, H0 to {H0_STEP} m grid, "
        f"H0L0 to {H0L0_STEP} grid. "
        f"Extrapolated to eta in [{ETA_MIN}, {ETA_MAX}] m using per-group linear fit. "
        f"WIDE variant: H0/H0L0 widened to raw-grid bounds (H0 up to 5.0 m, H0L0 down to "
        f"0.005) vs. the pipeline-optimized v4 table, to cover literature validation events. "
        f"Generated: {date_created}."
    )
    ds_new.source = "022_extend_beware_by_eta_H0_H0L0_wide.py (local re-run of atoll-paper/notebooks/022_extend_beware_by_eta_H0_H0L0.py with widened ranges)"
    ds_new.date_created = date_created

print(f"Saved \u2192 {OUTPUT_PATH}  ({len(df_save):,} rows)")

ds_check = nc.Dataset(OUTPUT_PATH)
n_rows = len(ds_check.dimensions["ID"])
print("\n\u2500\u2500 Saved NetCDF summary \u2500\u2500")
print(f"  Total rows : {n_rows:,}")
for var in ["H0", "H0L0", "eta0", "W_reef", "beta_ForeReef"]:
    vals = np.array(ds_check.variables[var][:])
    vals = vals[vals > -9000]
    print(
        f"    {var:15s} : {len(np.unique(vals)):4d} unique  "
        f"({np.unique(vals).min():.4f} \u2013 {np.unique(vals).max():.4f})"
    )
ds_check.close()

