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
# # Output Validation — BEWARE v4 Results
#
# Validates `Atoll_BEWARE_processed_outputs_2020-2150_10042026.parquet` against
# `Atoll_BEWARE_inputs_200326.parquet`.
#
# **Checks:**
# 1. Row counts & NaN rates
# 2. R2pIndex distributions
# 3. Temporal monotonicity
# 4. Quantile monotonicity
# 5. Atoll & transect-level summaries
# 6. Diagnostic plots for affected atolls (eta / H0 / H0L0 inputs)

# %% [markdown]
# ## 1. Setup
#
# > **Memory strategy:** outputs are loaded once and kept in memory. Inputs are large — they are loaded only when needed (quantile monotonicity merge and diagnostic plots), with only the required columns, then deleted immediately after.

# %%
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root (two levels up from current notebook folder)
try:
    project_root = Path(__file__).resolve().parents[1]
except NameError:
    project_root = Path.cwd().parent  # Jupyter: assumes cwd is notebooks/
sys.path.append(str(project_root))
from src.settings import DATA_DIR, INTERIM_DIR, PROCESSED_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Auto-detect the most recently modified inputs file from interim storage
_input_candidates = sorted(
    INTERIM_DIR.glob("Atoll_BEWARE_inputs_*.parquet"),
    key=lambda p: p.stat().st_mtime,
)
INPUTS_PATH = _input_candidates[-1]
print(f"Using inputs file : {INPUTS_PATH.name}")

# Auto-detect the most recently modified outputs file
_output_candidates = sorted(
    PROCESSED_DIR.glob("Atoll_BEWARE_processed_outputs_*.parquet"),
    key=lambda p: p.stat().st_mtime,
)
OUTPUT_PATH = _output_candidates[-1]
print(f"Using output file: {OUTPUT_PATH.name}")

R2P_COLS = [
    "R2pIndex_combined_rp1",
    "R2pIndex_combined_rp10",
    "R2pIndex_combined_rp100",
]
GROUP_KEYS = ["transect_id", "FID_GADM", "scenario", "year", "confidence"]
TEMPORAL_KEYS = ["transect_id", "FID_GADM", "scenario", "confidence", "quantile"]
COLORS = {
    "ssp119": "seagreen",
    "ssp126": "steelblue",
    "ssp245": "goldenrod",
    "ssp370": "darkorange",
    "ssp585": "tomato",
}
QCOLORS = {0.17: "seagreen", 0.50: "steelblue", 0.83: "tomato"}

# ── Load outputs only — stays in memory throughout ────────────────────────
outputs = pd.read_parquet(OUTPUT_PATH)

print(
    f"Outputs : {len(outputs):,} rows  |  {outputs['transect_id'].nunique():,} transects"
)
print(f"\nScenarios : {sorted(outputs['scenario'].unique())}")
print(f"Years     : {sorted(outputs['year'].unique())}")
print(f"Quantiles : {sorted(outputs['quantile'].unique())}")
print(f"Confidence: {sorted(outputs['confidence'].unique())}")


# %% [markdown]
# ## 2. NaN & Distribution Check

# %%
print("── NaN counts ───────────────────────────────────────────────")
for col in R2P_COLS:
    n = outputs[col].isna().sum()
    print(f"  {col:35s}: {n:,}  ({100*n/len(outputs):.2f}%)")

print("\n── R2pIndex_combined_rp1 distribution ───────────────────────")
desc = outputs["R2pIndex_combined_rp1"].describe()
print(
    f"  mean={desc['mean']:.3f}  median={desc['50%']:.3f}  "
    f"std={desc['std']:.3f}  min={desc['min']:.3f}  max={desc['max']:.3f}"
)

print("\n── NaN by scenario ──────────────────────────────────────────")
nan_by_scenario = outputs.groupby("scenario")["R2pIndex_combined_rp1"].apply(
    lambda x: x.isna().sum()
)
print(nan_by_scenario.to_string())

# %% [markdown]
# ## 2b. NaN Source Diagnosis
#
# For rows where R2pIndex is NaN, load the corresponding input rows and check
# which input variables are NaN or likely out-of-range for the BEWARE lookup.

# %%
INPUT_DIAG_COLS = ["eta_combined_rp1", "eta_combined_rp10", "eta_combined_rp100",
                   "H0", "H0L0"]

nan_rows = outputs[outputs["R2pIndex_combined_rp1"].isna()][
    GROUP_KEYS + ["quantile"]
].drop_duplicates()
print(f"Unique NaN output rows (by group keys): {len(nan_rows):,}")
print(f"Unique NaN transects                  : {nan_rows['transect_id'].nunique():,}")

inputs_diag = pd.read_parquet(INPUTS_PATH, columns=GROUP_KEYS + ["quantile"] + INPUT_DIAG_COLS)
merged_nan = nan_rows.merge(inputs_diag, on=GROUP_KEYS + ["quantile"], how="left")
del inputs_diag
gc.collect()

print("\n── NaN rate in inputs for NaN-output rows ───────────────────")
for col in INPUT_DIAG_COLS:
    n = merged_nan[col].isna().sum()
    print(f"  {col:30s}: {n:,} NaN  ({100*n/len(merged_nan):.1f}%)")

print("\n── Value ranges in inputs for NaN-output rows ───────────────")
for col in INPUT_DIAG_COLS:
    valid = merged_nan[col].dropna()
    if len(valid):
        print(f"  {col:30s}: min={valid.min():.4f}  max={valid.max():.4f}  "
              f"mean={valid.mean():.4f}  n_negative={( valid < 0).sum():,}")

print("\n── NaN-output transects: NaN rate per input col ─────────────")
per_transect = (
    merged_nan.groupby("transect_id")[INPUT_DIAG_COLS]
    .apply(lambda df: df.isna().any())
    .sum()
)
print(per_transect.to_string())

print("\n── Breakdown of NaN outputs by year (first input col as proxy) ──")
print(merged_nan.groupby("year")["eta_combined_rp1"].apply(
    lambda x: x.isna().sum()
).to_string())

# %% [markdown]
# ## 3. Temporal Monotonicity
#
# R2pIndex should be non-decreasing over time for each transect × quantile × scenario × confidence.

# %%
temporal_results = []
for keys, grp in outputs.groupby(TEMPORAL_KEYS):
    grp_sorted = grp.sort_values("year")
    for r2p in R2P_COLS:
        vals = grp_sorted[r2p].values
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1 and not np.all(np.diff(vals) >= 0):
            temporal_results.append(dict(zip(TEMPORAL_KEYS, keys), r2p=r2p))

df_temporal = pd.DataFrame(temporal_results)
n_groups_total = outputs.groupby(TEMPORAL_KEYS).ngroups * len(R2P_COLS)
n_temporal_fails = len(df_temporal)

print(f"Groups checked      : {n_groups_total:,}")
print(
    f"Temporal violations : {n_temporal_fails:,}  ({100*n_temporal_fails/n_groups_total:.2f}%)"
)

if n_temporal_fails > 0:
    print("\nBreakdown by return period:")
    print(df_temporal["r2p"].value_counts().to_string())
    print("\nBreakdown by scenario:")
    print(df_temporal["scenario"].value_counts().to_string())
    print("\nBreakdown by quantile:")
    print(df_temporal["quantile"].value_counts().to_string())
    print(
        f"\nAffected transects  : {df_temporal['transect_id'].nunique():,} / "
        f"{outputs['transect_id'].nunique():,}"
    )
else:
    print("\n✓ No temporal monotonicity violations.")

# %% [markdown]
# ## 4. Quantile Monotonicity
#
# R2pIndex should be ordered q0.17 ≤ q0.50 ≤ q0.83 for every transect × scenario × year × confidence group.

# %%
# ── Load inputs — only the columns needed for the merge ─────────────────
INPUT_COLS_NEEDED = GROUP_KEYS + ["quantile", "Atoll_FID"]
inputs_slim = pd.read_parquet(INPUTS_PATH, columns=INPUT_COLS_NEEDED)
print(
    f"inputs_slim : {len(inputs_slim):,} rows  |  {inputs_slim['transect_id'].nunique():,} transects"
)

merged = pd.merge(
    inputs_slim,
    outputs[GROUP_KEYS + ["quantile"] + R2P_COLS],
    on=GROUP_KEYS + ["quantile"],
    how="inner",
)
del inputs_slim
gc.collect()

pivot = merged.pivot_table(
    index=GROUP_KEYS + ["Atoll_FID"],
    columns="quantile",
    values=R2P_COLS,
    aggfunc="first",
)
del merged
gc.collect()

pivot.columns = [f"{col}_{q}" for col, q in pivot.columns]
pivot = pivot.reset_index()

for r2p in R2P_COLS:
    pivot[f"nonmono_lo_{r2p}"] = pivot[f"{r2p}_0.5"] < pivot[f"{r2p}_0.17"]
    pivot[f"nonmono_hi_{r2p}"] = pivot[f"{r2p}_0.83"] < pivot[f"{r2p}_0.5"]

nonmono_cols = [f"nonmono_lo_{r}" for r in R2P_COLS] + [
    f"nonmono_hi_{r}" for r in R2P_COLS
]
pivot["any_nonmono"] = pivot[nonmono_cols].any(axis=1)

n_total = len(pivot)
n_nonmono = pivot["any_nonmono"].sum()
print(f"Total groups   : {n_total:,}")
print(f"Non-monotonic  : {n_nonmono:,}  ({100*n_nonmono/n_total:.2f}%)")
print(
    f"Clean          : {n_total - n_nonmono:,}  ({100*(n_total-n_nonmono)/n_total:.2f}%)"
)

print("\nBreakdown by return period and direction:")
for r2p in R2P_COLS:
    n_lo = pivot[f"nonmono_lo_{r2p}"].sum()
    n_hi = pivot[f"nonmono_hi_{r2p}"].sum()
    print(f"  {r2p:35s}  median<17th: {n_lo:,}   83rd<median: {n_hi:,}")


# %% [markdown]
# ## 5. Atoll & Transect-Level Summary

# %%
# ── Transect-level summary (collapsed across scenarios) ───────────────────────
transect_summary = (
    pivot.groupby("transect_id")
    .agg(total_groups=("any_nonmono", "count"), n_nonmono=("any_nonmono", "sum"))
    .reset_index()
)
transect_summary["pct_nonmono"] = (
    100 * transect_summary["n_nonmono"] / transect_summary["total_groups"]
)
transect_summary["has_any_nonmono"] = transect_summary["n_nonmono"] > 0

# ── Transect × scenario breakdown ─────────────────────────────────────────────
transect_scenario = (
    pivot.groupby(["transect_id", "scenario"])
    .agg(total_groups=("any_nonmono", "count"), n_nonmono=("any_nonmono", "sum"))
    .reset_index()
)
transect_scenario["pct_nonmono"] = (
    100 * transect_scenario["n_nonmono"] / transect_scenario["total_groups"]
)

print("% non-monotonic groups per transect × scenario:")
print(
    transect_scenario.groupby("scenario")["pct_nonmono"].describe().round(2).to_string()
)

# ── Atoll-level summary ────────────────────────────────────────────────────────
transect_flag = (
    pivot.groupby(["transect_id", "Atoll_FID"])["any_nonmono"]
    .any()
    .reset_index()
    .rename(columns={"any_nonmono": "has_nonmono"})
)
atoll_summary = (
    transect_flag.groupby("Atoll_FID")
    .agg(
        n_transects=("transect_id", "count"),
        n_transects_affected=("has_nonmono", "sum"),
    )
    .reset_index()
)
atoll_summary["pct_transects_affected"] = (
    100 * atoll_summary["n_transects_affected"] / atoll_summary["n_transects"]
)

print(f"\nTransects total    : {len(transect_summary):,}")
print(f"Transects affected : {transect_summary['has_any_nonmono'].sum():,}")
print(f"Atolls total       : {len(atoll_summary):,}")
print(f"Atolls affected    : {(atoll_summary['n_transects_affected'] > 0).sum():,}")

affected_atolls = atoll_summary[atoll_summary["n_transects_affected"] > 0]
if not affected_atolls.empty:
    print("\nAffected atolls:")
    print(
        affected_atolls.sort_values(
            "pct_transects_affected", ascending=False
        ).to_string(index=False)
    )

# %% [markdown]
# ## 6. Non-Monotonicity Plots
#
# Only runs if violations exist. Shows distribution summary, worst transect time series, and for each affected atoll: the input eta / H0 / H0L0 fields to diagnose the cause.

# %%
if n_nonmono == 0:
    print("✓ No quantile non-monotonicity violations — no plots needed.")
else:
    # ── Plot 1: Distribution summary ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Quantile non-monotonicity summary — v4 outputs", fontsize=12)

    data = transect_summary["pct_nonmono"]
    axes[0].hist(data, bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(
        data.median(),
        color="tomato",
        ls="--",
        lw=1.5,
        label=f"median = {data.median():.1f}%",
    )
    axes[0].set_xlabel("% non-monotonic groups per transect")
    axes[0].set_ylabel("Number of transects")
    axes[0].set_title("All transects")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    data_aff = transect_summary[transect_summary["has_any_nonmono"]]["pct_nonmono"]
    axes[1].hist(data_aff, bins=50, color="tomato", edgecolor="white")
    axes[1].set_xlabel("% non-monotonic groups per transect")
    axes[1].set_ylabel("Number of transects")
    axes[1].set_title(f"Affected transects only  (n={len(data_aff):,})")
    axes[1].grid(alpha=0.3)

    axes[2].hist(
        atoll_summary["pct_transects_affected"],
        bins=40,
        color="steelblue",
        edgecolor="white",
    )
    axes[2].axvline(
        atoll_summary["pct_transects_affected"].median(),
        color="tomato",
        ls="--",
        lw=1.5,
        label=f"median = {atoll_summary['pct_transects_affected'].median():.1f}%",
    )
    axes[2].set_xlabel("% transects affected per atoll")
    axes[2].set_ylabel("Number of atolls")
    axes[2].set_title("Atoll level")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
if n_nonmono > 0:
    # ── Plot 2: Worst transect — R2pIndex time series with violations marked ───
    worst = transect_summary.sort_values("pct_nonmono", ascending=False).iloc[0]
    print(
        f"Worst transect: {worst['transect_id']}  ({worst['pct_nonmono']:.1f}% non-monotonic groups)"
    )

    t_pivot = pivot[
        (pivot["transect_id"] == worst["transect_id"])
        & (pivot["scenario"].isin(["ssp119", "ssp245", "ssp585"]))
        & (pivot["confidence"] == "medium")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"Worst transect {worst['transect_id']} — R2pIndex over time  "
        f"({worst['pct_nonmono']:.1f}% non-monotonic, × = violations)",
        fontsize=11,
    )
    plot_colors = {"ssp119": "seagreen", "ssp245": "steelblue", "ssp585": "tomato"}
    for ax, r2p in zip(axes, R2P_COLS):
        for scenario, color in plot_colors.items():
            grp = t_pivot[t_pivot["scenario"] == scenario].sort_values("year")
            if grp.empty:
                continue
            ax.plot(grp["year"], grp[f"{r2p}_0.5"], color=color, lw=1.8, label=scenario)
            ax.fill_between(
                grp["year"],
                grp[f"{r2p}_0.17"],
                grp[f"{r2p}_0.83"],
                color=color,
                alpha=0.2,
            )
            nm = grp[grp[f"nonmono_lo_{r2p}"] | grp[f"nonmono_hi_{r2p}"]]
            if not nm.empty:
                ax.scatter(
                    nm["year"],
                    nm[f"{r2p}_0.5"],
                    color=color,
                    marker="x",
                    s=80,
                    zorder=5,
                )
        ax.set_title(r2p, fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("R2pIndex (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
if n_nonmono > 0:
    INPUT_PANELS = [
        ("eta_combined_rp1", "eta (m)"),
        ("H0", "H0 (m)"),
        ("H0L0", "H0L0 (m²)"),
    ]
    INPUT_DIAG_COLS = GROUP_KEYS + ["quantile"] + [col for col, _ in INPUT_PANELS]

    def mean_by_year_quantile(df, col, transects, scenario):
        return (
            df[
                df["transect_id"].isin(transects)
                & (df["scenario"] == scenario)
                & (df["confidence"] == "medium")
            ]
            .groupby(["year", "quantile"])[col]
            .mean()
            .unstack("quantile")
            .sort_index()
        )

    for _, atoll_row in affected_atolls.iterrows():
        atoll_fid = int(atoll_row["Atoll_FID"])
        affected_transects = transect_flag[
            (transect_flag["Atoll_FID"] == atoll_fid) & transect_flag["has_nonmono"]
        ]["transect_id"].values

        print(f"\n── Atoll FID {atoll_fid} ─────────────────────────────────────")
        print(f"  Total transects : {int(atoll_row['n_transects'])}")
        print(
            f"  Affected        : {int(atoll_row['n_transects_affected'])}  "
            f"({atoll_row['pct_transects_affected']:.1f}%)"
        )

        atoll_scenario_detail = transect_scenario[
            transect_scenario["transect_id"].isin(affected_transects)
        ][["transect_id", "scenario", "pct_nonmono"]]
        atoll_pivot_scen = (
            atoll_scenario_detail.pivot(
                index="transect_id", columns="scenario", values="pct_nonmono"
            )
            .fillna(0)
            .round(1)
        )
        print("\n  % non-monotonic per transect × scenario:")
        print(atoll_pivot_scen.to_string())

        worst_scenario = (
            atoll_pivot_scen.mean().idxmax() if not atoll_pivot_scen.empty else "ssp585"
        )

        # ── Load only needed input columns for affected transects, then delete
        inputs_diag = pd.read_parquet(
            INPUTS_PATH,
            columns=INPUT_DIAG_COLS,
        )
        inputs_diag = inputs_diag[inputs_diag["transect_id"].isin(affected_transects)]

        n_cols = len(INPUT_PANELS) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        fig.suptitle(
            f"Atoll FID {atoll_fid} — {worst_scenario} — inputs & output\n"
            f"({len(affected_transects)} affected transects, medium confidence)",
            fontsize=10,
        )

        for ax, (col, ylabel) in zip(axes[: n_cols - 1], INPUT_PANELS):
            data = mean_by_year_quantile(
                inputs_diag, col, affected_transects, worst_scenario
            )
            for q, color in QCOLORS.items():
                if q in data.columns:
                    ax.plot(data.index, data[q], color=color, lw=1.5, label=f"q={q}")
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        del inputs_diag
        gc.collect()

        ax = axes[-1]
        data = mean_by_year_quantile(
            outputs, "R2pIndex_combined_rp1", affected_transects, worst_scenario
        )
        for q, color in QCOLORS.items():
            if q in data.columns:
                ax.plot(data.index, data[q], color=color, lw=1.5, label=f"q={q}")
        nonmono_years = pivot[
            pivot["transect_id"].isin(affected_transects)
            & (pivot["scenario"] == worst_scenario)
            & (
                pivot["nonmono_lo_R2pIndex_combined_rp1"]
                | pivot["nonmono_hi_R2pIndex_combined_rp1"]
            )
        ]["year"].unique()
        if len(nonmono_years) and 0.5 in data.columns:
            ax.scatter(
                nonmono_years,
                data.loc[data.index.isin(nonmono_years), 0.5],
                color="tomato",
                marker="x",
                s=80,
                zorder=5,
                label="violation",
            )
        ax.set_title("R2pIndex_combined_rp1", fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("R2pIndex (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# %% [markdown]
# ## 7. Global Aggregate Time Series
#
# Mean R2pIndex across all transects per scenario — quick sanity check that the output is physically reasonable.

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
fig.suptitle(
    "R2pIndex — mean across all transects — medium confidence, q50 line + q17–q83 shading",
    fontsize=11,
)

for ax, r2p in zip(axes, R2P_COLS):
    for scenario, color in COLORS.items():
        grp = (
            outputs[
                (outputs["scenario"] == scenario) & (outputs["confidence"] == "medium")
            ]
            .groupby(["year", "quantile"])[r2p]
            .mean()
            .unstack("quantile")
            .sort_index()
        )
        if grp.empty or 0.50 not in grp.columns:
            continue
        ax.plot(grp.index, grp[0.50], color=color, lw=1.8, label=scenario)
        if {0.17, 0.83}.issubset(grp.columns):
            ax.fill_between(
                grp.index, grp[0.17], grp[0.83], color=color, alpha=0.15, lw=0
            )

    ax.set_title(r2p, fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean R2pIndex (m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
