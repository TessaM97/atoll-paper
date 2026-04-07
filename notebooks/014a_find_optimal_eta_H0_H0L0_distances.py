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
# # 014a — Find Optimal BEWARE Extension Ranges and Filter Half-Widths
#
# This notebook must be run before `014_extend_beware_by_eta_H0_H0L0.py`.
# It determines:
# 1. The optimal filter half-widths for matching inputs to the BEWARE grid
#    (based on p99 distance of input values to nearest grid point).
# 2. The extension ranges (eta, H0, H0L0) needed to cover all input values.
#
# Results are saved to `beware_matching_config.json` in PROCESSED_DIR,
# which is read by both `014_extend_beware_by_eta_H0_H0L0.py` and
# `030_extract_from_BEWARE.py`.

# %%
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import INTERIM_DIR, PROCESSED_DIR

print("Using processed directory:", PROCESSED_DIR)

# %% [markdown]
# ### 1) Filters to have in the BEWARE matching code, optimise the matching

# %%
# ── LOAD ──────────────────────────────────────────────────────────────────────
inputs = pd.read_parquet(INTERIM_DIR / "Atoll_BEWARE_inputs.parquet")

# Load BEWARE database to get actual grid values
ds = nc.Dataset(PROCESSED_DIR / "BEWARE_Database_extended_v4.nc")
beware_eta = np.array(ds.variables["eta0"][:])
beware_H0 = np.array(ds.variables["H0"][:])
beware_H0L0 = np.array(ds.variables["H0L0"][:])

# Unique grid values
eta_grid = np.unique(beware_eta[beware_eta > -9999])
H0_grid = np.unique(beware_H0[beware_H0 > -9999])
H0L0_grid = np.unique(beware_H0L0[beware_H0L0 > -9999])

print("BEWARE grid resolution:")
print(f"  eta  : {np.diff(eta_grid).mean():.4f} m  ({len(eta_grid)} values)")
print(f"  H0   : {np.diff(H0_grid).mean():.4f} m  ({len(H0_grid)} values)")
print(f"  H0L0 : {np.diff(H0L0_grid).mean():.5f}   ({len(H0L0_grid)} values)")

# ── FILTER VALID INPUTS ───────────────────────────────────────────────────────
inp = inputs[inputs["H0L0"] > 0].copy()

ETA_COLS = ["eta_combined_rp1", "eta_combined_rp10", "eta_combined_rp100"]


# ── FOR EACH INPUT VALUE, FIND DISTANCE TO NEAREST BEWARE GRID POINT ─────────
def nearest_dist(values, grid):
    """For each value, return distance to nearest grid point."""
    values = np.array(values)
    idx = np.searchsorted(grid, values, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    idx_lo = np.maximum(idx - 1, 0)
    dist = np.minimum(np.abs(values - grid[idx]), np.abs(values - grid[idx_lo]))
    return dist


# Eta distances across all eta columns
eta_vals = pd.concat([inp[c] for c in ETA_COLS]).dropna().values
eta_dists = nearest_dist(eta_vals, eta_grid)

# H0 distances
h0_dists = nearest_dist(inp["H0"].values, H0_grid)

# H0L0 distances
h0l0_dists = nearest_dist(inp["H0L0"].values, H0L0_grid)

# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────
print("\n── Distance to nearest BEWARE grid point ──")
for name, dists in [("eta", eta_dists), ("H0", h0_dists), ("H0L0", h0l0_dists)]:
    p = np.percentile(dists, [50, 90, 95, 99, 100])
    print(f"\n{name}:")
    print(
        f"  p50={p[0]:.5f}  p90={p[1]:.5f}  p95={p[2]:.5f}  "
        f"p99={p[3]:.5f}  max={p[4]:.5f}"
    )

# ── RECOMMENDED FILTER HALF-WIDTHS ───────────────────────────────────────────
# Filter must be wide enough to always capture the nearest grid point.
# Use p99 of nearest-distance + half grid step as safe margin.

eta_filter = np.percentile(eta_dists, 99) + np.diff(eta_grid).mean() / 2
h0_filter = np.percentile(h0_dists, 99) + np.diff(H0_grid).mean() / 2
h0l0_filter = np.percentile(h0l0_dists, 99) + np.diff(H0L0_grid).mean() / 2

# Round up to nearest sensible value
eta_filter = np.ceil(eta_filter * 20) / 20   # round to 0.05
h0_filter = np.ceil(h0_filter * 10) / 10     # round to 0.1
h0l0_filter = np.ceil(h0l0_filter * 1000) / 1000  # round to 0.001

print("\n── Recommended filter half-widths ──")
print(f"  eta  : ±{eta_filter:.3f} m")
print(f"  H0   : ±{h0_filter:.3f} m")
print(f"  H0L0 : ±{h0l0_filter:.4f}")

# ── VERIFY: what fraction of inputs fall within these filters? ────────────────
print("\n── Coverage check (fraction of inputs within filter) ──")
print(f"  eta  : {(eta_dists  <= eta_filter).mean()*100:.2f}%")
print(f"  H0   : {(h0_dists   <= h0_filter).mean()*100:.2f}%")
print(f"  H0L0 : {(h0l0_dists <= h0l0_filter).mean()*100:.2f}%")

# ── ESTIMATE SPEEDUP ──────────────────────────────────────────────────────────
beware_valid = (
    (np.isclose(ds.variables["Cf"][:], 0.05))
    & (np.isclose(ds.variables["beta_Beach"][:], 0.10))
    & (ds.variables["W_reef"][:] > 1)
)
n_total = beware_valid.sum()

frac_eta = 2 * eta_filter / (eta_grid.max() - eta_grid.min())
frac_h0 = 2 * h0_filter / (H0_grid.max() - H0_grid.min())
frac_h0l0 = 2 * h0l0_filter / (H0L0_grid.max() - H0L0_grid.min())
n_after = int(n_total * frac_eta * frac_h0 * frac_h0l0)

print("\n── Estimated candidate set size after filtering ──")
print(f"  Total BEWARE entries (Cf/beta_Beach filtered) : {n_total:,}")
print(f"  After eta+H0+H0L0 filters                    : ~{n_after:,}")
print(f"  Speedup factor                                : ~{n_total//max(n_after,1)}x")

# ── PLOT: distance distributions ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Distance to nearest BEWARE grid point per input value", fontsize=12)

for ax, name, dists, filt, color in zip(
    axes,
    ["eta", "H0", "H0L0"],
    [eta_dists, h0_dists, h0l0_dists],
    [eta_filter, h0_filter, h0l0_filter],
    ["steelblue", "darkorange", "seagreen"],
):
    ax.hist(dists, bins=60, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(
        filt, color="tomato", ls="--", lw=1.8, label=f"Recommended filter = ±{filt:.4f}"
    )
    ax.set_title(name)
    ax.set_xlabel("Distance to nearest grid point")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2) Boundaries for BEWARE extension

# %%
# ── Load inputs ───────────────────────────────────────────────────────────────
inp = inputs[inputs["H0L0"] > 0].copy()

ETA_COLS = ["eta_combined_rp1", "eta_combined_rp10", "eta_combined_rp100"]

# ── Actual input ranges ───────────────────────────────────────────────────────
eta_vals = pd.concat([inp[c] for c in ETA_COLS]).dropna().values
h0_vals = inp["H0"].values
h0l0_vals = inp["H0L0"].values

print("── Actual input ranges ──")
for name, vals in [("eta", eta_vals), ("H0", h0_vals), ("H0L0", h0l0_vals)]:
    p = np.percentile(vals, [0.1, 1, 5, 95, 99, 99.9, 100])
    print(f"\n{name}:")
    print(
        f"  p0.1={p[0]:.4f}  p1={p[1]:.4f}  p5={p[2]:.4f} | "
        f"p95={p[3]:.4f}  p99={p[4]:.4f}  p99.9={p[5]:.4f}  max={p[6]:.4f}"
    )

# ── Recommended BEWARE extension ranges ──────────────────────────────────────
# Add filter half-width as margin so edge inputs always find candidates
ETA_FILTER = 0.050
H0_FILTER = 0.400
H0L0_FILTER = 0.002

# Snap to grid resolution
ETA_STEP = 0.05
H0_STEP = 0.10
H0L0_STEP = 0.001


def snap_lo(val, step):
    return np.floor(val / step) * step


def snap_hi(val, step):
    return np.ceil(val / step) * step


eta_lo = snap_lo(eta_vals.min() - ETA_FILTER, ETA_STEP)
eta_hi = snap_hi(eta_vals.max() + ETA_FILTER, ETA_STEP)
h0_lo = snap_lo(h0_vals.min() - H0_FILTER, H0_STEP)
h0_hi = snap_hi(h0_vals.max() + H0_FILTER, H0_STEP)
h0l0_lo = snap_lo(h0l0_vals.min() - H0L0_FILTER, H0L0_STEP)
h0l0_hi = snap_hi(h0l0_vals.max() + H0L0_FILTER, H0L0_STEP)

# Clamp to BEWARE original grid bounds
eta_lo = max(eta_lo, -1.0)
eta_hi = min(eta_hi, 8.5)
h0_lo = max(h0_lo, 1.0)
h0_hi = min(h0_hi, 5.0)
h0l0_lo = max(h0l0_lo, 0.005)
h0l0_hi = min(h0l0_hi, 0.05)

# Compute resulting grid sizes
n_eta = round((eta_hi - eta_lo) / ETA_STEP) + 1
n_h0 = round((h0_hi - h0_lo) / H0_STEP) + 1
n_h0l0 = round((h0l0_hi - h0l0_lo) / H0L0_STEP) + 1
N_OUTER = 36  # W_reef (12) × beta_ForeReef (3)
n_total = n_eta * n_h0 * n_h0l0 * N_OUTER

print("\n── Recommended BEWARE extension ranges ──")
print(f"  eta  : {eta_lo:.2f} – {eta_hi:.2f} m   → {n_eta}  grid points")
print(f"  H0   : {h0_lo:.2f} – {h0_hi:.2f} m    → {n_h0}  grid points")
print(f"  H0L0 : {h0l0_lo:.4f} – {h0l0_hi:.4f}  → {n_h0l0} grid points")
print(f"  Outer groups (W_reef × beta_ForeReef) : {N_OUTER}")
print(f"\n  Estimated total rows : {n_total:,}")
print(f"  Estimated file size  : ~{n_total * 8 * 8 / 1e9:.2f} GB  (8 vars × float64)")

print("\n── Paste these into 014_extend_beware_by_eta_H0_H0L0.py ──")
print(f"ETA_MIN   = {eta_lo}")
print(f"ETA_MAX   = {eta_hi}")
print(f"H0_MIN    = {h0_lo}   # clip H0 grid lower bound")
print(f"H0_MAX    = {h0_hi}   # clip H0 grid upper bound")
print(f"H0L0_MIN  = {h0l0_lo}  # clip H0L0 grid lower bound")
print(f"H0L0_MAX  = {h0l0_hi}  # clip H0L0 grid upper bound")

# %%
config = {
    "filter_halfwidths": {
        "ETA_FILTER": float(eta_filter),
        "H0_FILTER": float(h0_filter),
        "H0L0_FILTER": float(h0l0_filter),
    },
    "beware_extension_ranges": {
        "ETA_MIN": float(eta_lo),
        "ETA_MAX": float(eta_hi),
        "H0_MIN": float(h0_lo),
        "H0_MAX": float(h0_hi),
        "H0L0_MIN": float(h0l0_lo),
        "H0L0_MAX": float(h0l0_hi),
    },
}

config_path = PROCESSED_DIR / "beware_matching_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Config saved → {config_path}")
print(json.dumps(config, indent=2))

# %%
