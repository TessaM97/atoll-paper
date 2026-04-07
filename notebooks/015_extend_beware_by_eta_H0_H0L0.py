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

# %% [markdown]
# # BEWARE Database — Average, Interpolate, Extrapolate
# Filtered to Cf=0.05, beta_Beach=0.10 only (matches matching script hardcoded filter).
#
# Steps:
# 1. Load raw BEWARE NetCDF and filter to Cf=0.05, beta_Beach=0.10
# 2. Average 4 duplicate entries per (group + eta)
# 3. Plot 6 example groups (averaged raw data)
# 4. Interpolate to 0.05 m eta grid; extrapolate beyond original range
# 5. Plot 6 example groups (full extended eta curve)
# 6. Interpolate H0 to 0.1 m resolution (independent dimension)
# 7. Plot 6 example groups (H0 interpolation check)
# 8. Interpolate H0L0 to 0.001 resolution (independent dimension)
# 9. Plot 6 example groups (H0L0 interpolation check)
# 10. Save as new NetCDF
#
# Note: Run `014_find_optimal_eta_H0_H0L0_distances.py` first to generate
# `beware_matching_config.json`, which sets the extension ranges used here.

# %%
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import FIG_DIR, PROCESSED_DIR, RAW_DIR

print("Using processed directory:", PROCESSED_DIR)

# %%
# ── PATHS ─────────────────────────────────────────────────────────────────────
FILE_PATH   = RAW_DIR / "external/BEWARE/BEWARE_Database.nc"
OUTPUT_PATH = PROCESSED_DIR / "BEWARE_Database_extended_v4.nc"
CONFIG_PATH = PROCESSED_DIR / "beware_matching_config.json"

# ── LOAD CONFIG ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

rng = cfg["beware_extension_ranges"]

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CF_FIXED         = 0.05
BETA_BEACH_FIXED = 0.10
GROUP_COLS       = ["H0L0", "W_reef", "beta_ForeReef", "H0"]

ETA_STEP  = 0.05   # eta interpolation resolution (m)
H0_STEP   = 0.1    # H0 interpolation resolution (m)
H0L0_STEP = 0.001  # H0L0 interpolation resolution

# ── RANGES FROM CONFIG (set by 014) ─────────────────────────────────────────
ETA_MIN  = rng["ETA_MIN"]
ETA_MAX  = rng["ETA_MAX"]
H0_MIN   = rng["H0_MIN"]
H0_MAX   = rng["H0_MAX"]
H0L0_MIN = rng["H0L0_MIN"]
H0L0_MAX = rng["H0L0_MAX"]

print("── Loaded extension ranges from config ──")
print(f"  eta  : {ETA_MIN} – {ETA_MAX} m")
print(f"  H0   : {H0_MIN} – {H0_MAX} m")
print(f"  H0L0 : {H0L0_MIN} – {H0L0_MAX}")

# ── OTHER CONSTANTS ───────────────────────────────────────────────────────────
FILL_VALUE  = -9999.0
N_EXAMPLES  = 6
ETA_EXAMPLE = 0.70

# %%
# ── 1. LOAD AND FILTER ────────────────────────────────────────────────────────
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

# Filter to Cf and beta_Beach values used by the matcher
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

# %%
# ── 2. AVERAGE 4 DUPLICATE ENTRIES PER (GROUP + ETA) ─────────────────────────
df_avg = df.groupby(GROUP_COLS + ["eta"], as_index=False)["R2p"].mean()
print(f"Rows after averaging : {len(df_avg):,}")
print(f"Duplicates remaining : {df_avg.duplicated(subset=GROUP_COLS + ['eta']).sum()}")

# %%
# ── HELPERS ───────────────────────────────────────────────────────────────────
all_group_keys = list(df_avg.groupby(GROUP_COLS).groups.keys())
step = max(1, len(all_group_keys) // N_EXAMPLES)
example_keys = all_group_keys[::step][:N_EXAMPLES]


def get_group(df, keys, cols=GROUP_COLS, sort_col="eta"):
    mask = np.ones(len(df), dtype=bool)
    for col, val in zip(cols, keys):
        mask &= np.isclose(df[col], val)
    return df[mask].sort_values(sort_col)


def group_label(keys):
    return f"H0L0={keys[0]:.3f} W={keys[1]:.0f} " f"βF={keys[2]:.2f} H0={keys[3]:.2f}"


# %%
# ── 3. PLOT — AVERAGED RAW DATA ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Step 1 — Averaged raw BEWARE data\n(one R2p per eta per group)", fontsize=13
)

for ax, keys in zip(axes.flat, example_keys):
    grp = get_group(df_avg, keys)
    ax.plot(grp["eta"], grp["R2p"], "o-", ms=4, lw=1.2, color="steelblue")
    ax.set_title(group_label(keys), fontsize=8)
    ax.set_xlabel("η₀ (m)")
    ax.set_ylabel("R2pIndex (m)")
    ax.grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(FIG_DIR / "step1_averaged_raw.png", dpi=100, bbox_inches="tight")
plt.show()

# %%
# ── 4. INTERPOLATE ETA TO 0.05 m GRID + EXTRAPOLATE ──────────────────────────
# Within original eta range → np.interp (exact at averaged nodes)
# Outside original eta range → linear extrapolation via np.polyfit

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

    # Global slope (all points) — but anchored to the boundary nodes
    slope, _ = np.polyfit(eta_vals, r2p_vals, 1)

    for eta_val in eta_grid:
        if eta_val > eta_hi:
            # Anchored to last original point — zero offset at seam by construction
            r2p = r2p_vals[-1] + slope * (eta_val - eta_hi)
        elif eta_val < eta_lo:
            # Anchored to first original point
            r2p = r2p_vals[0] + slope * (eta_val - eta_lo)
        else:
            r2p = float(np.interp(eta_val, eta_vals, r2p_vals))
        interp_rows.append(dict(zip(GROUP_COLS, group_keys), eta=eta_val, R2p=r2p))

df_interp = pd.DataFrame(interp_rows)
print(f"Rows after eta interpolation/extrapolation : {len(df_interp):,}")

# %%
# ── 5. PLOT — FULL EXTENDED ETA CURVE ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    "Step 2 — Full extended eta curve\n"
    "(interpolated within original range, extrapolated outside)",
    fontsize=13,
)

for ax, keys in zip(axes.flat, example_keys):
    grp_avg = get_group(df_avg, keys)
    grp_interp = get_group(df_interp, keys)
    eta_lo, eta_hi = grp_avg["eta"].min(), grp_avg["eta"].max()
    within = grp_interp[(grp_interp["eta"] >= eta_lo) & (grp_interp["eta"] <= eta_hi)]
    extrap = grp_interp[(grp_interp["eta"] < eta_lo) | (grp_interp["eta"] > eta_hi)]

    ax.plot(
        grp_avg["eta"],
        grp_avg["R2p"],
        "o",
        ms=5,
        color="steelblue",
        label="Averaged raw",
        zorder=3,
    )
    ax.plot(
        within["eta"],
        within["R2p"],
        "-",
        lw=1.5,
        color="darkorange",
        label="Interpolated",
    )
    ax.plot(
        extrap["eta"], extrap["R2p"], "--", lw=1.2, color="tomato", label="Extrapolated"
    )
    ax.axvline(eta_lo, color="grey", lw=0.8, ls=":")
    ax.axvline(eta_hi, color="grey", lw=0.8, ls=":")
    ax.set_title(group_label(keys), fontsize=8)
    ax.set_xlabel("η₀ (m)")
    ax.set_ylabel("R2pIndex (m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig(FIG_DIR / "step2_extended_eta_curves.png", dpi=100, bbox_inches="tight")

# %%
# ── 6. INTERPOLATE H0 TO 0.1 m RESOLUTION (vectorized) ───────────────────────
# Use ALL original H0 values (1,2,3,4,5) as anchor points for interpolation.
# Only the OUTPUT grid is clipped to the input data range.

NON_H0_COLS = ["H0L0", "W_reef", "beta_ForeReef"]

H0_orig = np.array(sorted(df_interp["H0"].unique()))  # full df, all 5 anchors
H0_grid = np.round(
    np.arange(H0_MIN, H0_MAX + H0_STEP / 2, H0_STEP), 1
)  # clipped output
print(f"H0 original ({len(H0_orig)} values) : {H0_orig.tolist()}")
print(
    f"H0 output grid ({len(H0_grid)} values, clipped to {H0_MIN}–{H0_MAX}) : {H0_grid.tolist()}"
)

df_pivot_h0 = df_interp.pivot_table(  # full df, NOT clipped
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

# %%
# ── 7. PLOT — H0 INTERPOLATION CHECK ─────────────────────────────────────────
non_h0_keys = list(df_h0_interp.groupby(NON_H0_COLS).groups.keys())
step_nh = max(1, len(non_h0_keys) // N_EXAMPLES)
example_nh_keys = non_h0_keys[::step_nh][:N_EXAMPLES]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    f"Step 3 — H0 interpolation to {H0_STEP} m resolution\n"
    f"(shown at η₀ = {ETA_EXAMPLE} m)",
    fontsize=13,
)

for ax, keys in zip(axes.flat, example_nh_keys):
    mask_o = np.ones(len(df_interp), dtype=bool)
    mask_i = np.ones(len(df_h0_interp), dtype=bool)
    for col, val in zip(NON_H0_COLS, keys):
        mask_o &= np.isclose(df_interp[col], val)
        mask_i &= np.isclose(df_h0_interp[col], val)
    mask_o &= np.isclose(df_interp["eta"], ETA_EXAMPLE)
    mask_i &= np.isclose(df_h0_interp["eta"], ETA_EXAMPLE)

    grp_orig = df_interp[mask_o].sort_values("H0")
    grp_interp = df_h0_interp[mask_i].sort_values("H0")
    if grp_orig.empty:
        continue

    ax.plot(
        grp_interp["H0"],
        grp_interp["R2p"],
        "-",
        lw=1.5,
        color="darkorange",
        label=f"Interpolated {H0_STEP} m",
    )
    ax.plot(
        grp_orig["H0"],
        grp_orig["R2p"],
        "o",
        ms=8,
        color="steelblue",
        label="Original H0",
        zorder=3,
    )
    ax.set_title(f"H0L0={keys[0]:.3f} W={keys[1]:.0f} βF={keys[2]:.2f}", fontsize=8)
    ax.set_xlabel("H0 (m)")
    ax.set_ylabel("R2pIndex (m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(FIG_DIR / "step3_extended_H0_curves.png", dpi=100, bbox_inches="tight")
plt.close()

# %%
# ── 8. INTERPOLATE H0L0 TO 0.001 RESOLUTION (vectorized) ─────────────────────
# IMPORTANT: pivot must use ALL original H0L0 values as anchor points (0.005, 0.025, 0.05).
# Only the OUTPUT grid is clipped to the input data range.

NON_H0L0_COLS = ["H0", "W_reef", "beta_ForeReef"]

H0L0_orig = np.array(sorted(df_h0_interp["H0L0"].unique()))
print(f"H0L0 original ({len(H0L0_orig)} values) : {H0L0_orig.tolist()}")

H0L0_grid = np.round(np.arange(H0L0_MIN, H0L0_MAX + H0L0_STEP / 2, H0L0_STEP), 4)
print(
    f"H0L0 output grid ({len(H0L0_grid)} values, clipped to {H0L0_MIN}–{H0L0_MAX}) : "
    f"from {H0L0_grid[0]} to {H0L0_grid[-1]}"
)

df_pivot_h0l0 = df_h0_interp.pivot_table(  # full df, NOT clipped
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

# Verify snapping
H0_new = np.array(sorted(df_h0l0_interp["H0"].unique()))
H0L0_new = np.array(sorted(df_h0l0_interp["H0L0"].unique()))
print("\nSnapping verification:")
for val in [1.0, 1.4, 2.0]:
    nearest = H0_new[np.argmin(np.abs(H0_new - val))]
    print(f"  H0={val:.1f}   → snaps to {nearest:.1f}")
for val in [0.0146, 0.0162, 0.0174]:
    nearest = H0L0_new[np.argmin(np.abs(H0L0_new - val))]
    print(f"  H0L0={val:.4f} → snaps to {nearest:.4f}")

# %%
# ── 9. PLOT — H0L0 INTERPOLATION CHECK ───────────────────────────────────────
non_h0l0_keys = list(df_h0l0_interp.groupby(NON_H0L0_COLS).groups.keys())
step_nl = max(1, len(non_h0l0_keys) // N_EXAMPLES)
example_nl_keys = non_h0l0_keys[::step_nl][:N_EXAMPLES]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    f"Step 4 — H0L0 interpolation to {H0L0_STEP} resolution\n"
    f"(shown at η₀ = {ETA_EXAMPLE} m)",
    fontsize=13,
)

for ax, keys in zip(axes.flat, example_nl_keys):
    mask_o = np.ones(len(df_h0_interp), dtype=bool)
    mask_i = np.ones(len(df_h0l0_interp), dtype=bool)
    for col, val in zip(NON_H0L0_COLS, keys):
        mask_o &= np.isclose(df_h0_interp[col], val)
        mask_i &= np.isclose(df_h0l0_interp[col], val)
    mask_o &= np.isclose(df_h0_interp["eta"], ETA_EXAMPLE)
    mask_i &= np.isclose(df_h0l0_interp["eta"], ETA_EXAMPLE)

    grp_orig = df_h0_interp[mask_o].sort_values("H0L0")
    grp_interp = df_h0l0_interp[mask_i].sort_values("H0L0")
    if grp_orig.empty:
        continue

    ax.plot(
        grp_interp["H0L0"],
        grp_interp["R2p"],
        "-",
        lw=1.5,
        color="darkorange",
        label=f"Interpolated {H0L0_STEP}",
    )
    ax.plot(
        grp_orig["H0L0"],
        grp_orig["R2p"],
        "o",
        ms=8,
        color="steelblue",
        label="Original H0L0",
        zorder=3,
    )
    ax.set_title(f"H0={keys[0]:.1f} W={keys[1]:.0f} βF={keys[2]:.2f}", fontsize=8)
    ax.set_xlabel("H0L0 (-)")
    ax.set_ylabel("R2pIndex (m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(FIG_DIR / "step4_extended_H0L0_curves.png", dpi=100, bbox_inches="tight")
plt.close()

# %%
# ── 10. SAVE AS NETCDF ────────────────────────────────────────────────────────
df_save = df_h0l0_interp.rename(columns={"eta": "eta0", "R2p": "R2pIndex"})

# Re-add fixed columns so NetCDF schema matches original
df_save["Cf"] = CF_FIXED
df_save["beta_Beach"] = BETA_BEACH_FIXED

df_save["eta0"] = df_save["eta0"].round(2)
df_save["H0"] = df_save["H0"].round(1)
df_save["H0L0"] = df_save["H0L0"].round(4)
df_save = df_save.fillna(FILL_VALUE)

NC_VARS = [
    "H0",
    "H0L0",
    "eta0",
    "Cf",
    "beta_ForeReef",
    "beta_Beach",
    "W_reef",
    "R2pIndex",
]

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

date_created = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

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
        f"Generated: {date_created}."
    )
    ds_new.source = "015_extend_beware_by_eta_H0_H0L0.py"
    ds_new.date_created = date_created

print(f"Saved → {OUTPUT_PATH}  ({len(df_save):,} rows)")

# ── Verify saved NetCDF ───────────────────────────────────────────────────────
ds_check = nc.Dataset(OUTPUT_PATH)
n_rows = len(ds_check.dimensions["ID"])

print("\n── Saved NetCDF summary ──")
print(f"  Total rows : {n_rows:,}")
print(f"  Variables  : {list(ds_check.variables.keys())}")
print("\n  Unique values per dimension:")
for var in ["H0", "H0L0", "eta0", "W_reef", "beta_ForeReef"]:
    vals = np.array(ds_check.variables[var][:])
    vals = vals[vals > -9000]
    print(
        f"    {var:15s} : {len(np.unique(vals)):4d} unique  "
        f"({np.unique(vals).min():.4f} – {np.unique(vals).max():.4f})"
    )

print("\n  Fixed dimensions:")
for var in ["Cf", "beta_Beach"]:
    vals = np.unique(np.array(ds_check.variables[var][:]))
    print(f"    {var:15s} : {vals}")

print(f"\n  Description: {ds_check.description}")
ds_check.close()

# %%
