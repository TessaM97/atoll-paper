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
import itertools
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from matplotlib import cm

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, FIG_DIR, PROCESSED_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)
# Path
file_path = RAW_DIR / "BEWARE_Database.nc"
output_path = PROCESSED_DIR / "BEWARE_Database_extended_avg.nc"
fig_path = Path(FIG_DIR) / "BEWARE_Extension"

# %% [markdown]
# ### Load and extent BEWARE framework
# BEWARE entries are grouped by H0L0, Cf, Beta_Beach, W_reef, beta_ForeReef, and H0.
# For each group, if multiple entries exist, their values are averaged.
# The BEWARE dataset is then extended by fitting a linear relationship 
# for each group with respect to η (eta), up to η = 8 m.
# The results are interpolated with a step size of 0.05 m in η.

# %%
# Load dataset
ds = nc.Dataset(file_path)


# Helper to extract variable with optional _FillValue masking
def safe_var(var_name):
    var = ds.variables[var_name]
    data = var[:]
    fill_val = var._FillValue if "_FillValue" in var.ncattrs() else None
    if fill_val is not None:
        data = np.where(data == fill_val, np.nan, data)
    return data


# Extract variables
W_reef = safe_var("W_reef")
beta_ForeReef = safe_var("beta_ForeReef")
H0 = safe_var("H0")
eta0 = safe_var("eta0")
R2p = safe_var("R2pIndex")
Cf = safe_var("Cf")
beta_beach = safe_var("beta_Beach")
H0L0 = safe_var("H0L0")

# Flatten into DataFrame
df = pd.DataFrame(
    {
        "H0L0": H0L0,
        "Cf": Cf,
        "Beta_Beach": beta_beach,
        "W_reef": W_reef,
        "beta_ForeReef": beta_ForeReef,
        "H0": H0,
        "eta": eta0,
        "R2p": R2p,
    }
)
df.dropna(inplace=True)

# Target eta range
eta_extended = np.arange(-1.0, 8.5, 0.05)

# Grouping columns
group_cols = ["H0L0", "Cf", "Beta_Beach", "W_reef", "beta_ForeReef", "H0"]

# Step 1: average R2p per group+eta
df_avg = df.groupby(group_cols + ["eta"], as_index=False)["R2p"].mean()

# Step 2: linear fit and extrapolation
extended_rows = []
grouped_avg = df_avg.groupby(group_cols)

for group_keys, group_df in grouped_avg:
    group_df_sorted = group_df.sort_values("eta")
    eta_vals = group_df_sorted["eta"].values
    r2p_vals = group_df_sorted["R2p"].values

    if len(eta_vals) < 2:
        continue  # Need at least two points for a fit

    # Fit a straight line: R2p = a*eta + b
    slope, intercept = np.polyfit(eta_vals, r2p_vals, 1)

    # Extrapolate over full eta range
    for eta_val in eta_extended:
        R2p_pred = slope * eta_val + intercept
        extended_rows.append(
            dict(zip(group_cols, group_keys), eta=eta_val, R2p=R2p_pred)
        )

# Create DataFrame with extrapolated data
df_extended = pd.DataFrame(extended_rows)

df_extended

# %%
# --- Plotting: Example from one group ---
# Choose a representative group for visualization
example_key = list(grouped_avg.groups.keys())[0]
example_group_orig = grouped_avg.get_group(example_key)
example_group_ext = df_extended[
    (df_extended.H0L0 == example_key[0])
    & (df_extended.Cf == example_key[1])
    & (df_extended.Beta_Beach == example_key[2])
    & (df_extended.W_reef == example_key[3])
    & (df_extended.beta_ForeReef == example_key[4])
    & (df_extended.H0 == example_key[5])
]

plt.figure(figsize=(10, 6))
plt.plot(example_group_ext["eta"], example_group_ext["R2p"], "x--", label="Extended")
plt.plot(example_group_orig["eta"], example_group_orig["R2p"], "o-", label="Original")
plt.title("R2p vs eta for one parameter group")
plt.xlabel("eta (m)")
plt.ylabel("R2p")
plt.legend()
plt.grid(True)
plt.show()


# %%
# Define fixed colormap
color_norm = mcolors.Normalize(vmin=0, vmax=1)
scalar_map = cm.ScalarMappable(
    norm=color_norm, cmap="tab10"
)  # or 'viridis', 'plasma', etc.

# Grouping values
H0L0_vals = [0.005, 0.025, 0.05]
Cf_vals = [0.01, 0.05, 0.1]
Beta_Beach_vals = [0.05, 0.1, 0.2]
Beta_ForeReef_vals = [0.05, 0.1, 0.5]

# Loop over all combinations
for h0l0_val, cf_val, beta_beach_val, beta_fore_val in itertools.product(
    H0L0_vals, Cf_vals, Beta_Beach_vals, Beta_ForeReef_vals
):
    # Filter data
    base_mask = (
        np.isclose(df["H0L0"], h0l0_val)
        & np.isclose(df["Cf"], cf_val)
        & np.isclose(df["Beta_Beach"], beta_beach_val)
        & np.isclose(df["beta_ForeReef"], beta_fore_val)
    )
    df_filtered = df[base_mask]
    df_ext_filtered = df_extended[
        (df_extended["H0L0"] == h0l0_val)
        & (df_extended["Cf"] == cf_val)
        & (df_extended["Beta_Beach"] == beta_beach_val)
        & (df_extended["beta_ForeReef"] == beta_fore_val)
    ]

    if df_filtered.empty:
        continue

    group_cols_local = ["H0", "W_reef"]
    grouped_orig = df_filtered.groupby(group_cols_local)
    grouped_ext = df_ext_filtered.groupby(group_cols_local)

    # Create consistent color mapping for each (H0, W_reef) group
    all_keys = list(grouped_orig.groups.keys())
    color_dict = {
        key: scalar_map.to_rgba(i / max(1, len(all_keys) - 1))  # avoid div-by-zero
        for i, key in enumerate(all_keys)
    }

    plt.figure(figsize=(10, 6))

    for group_key, group_df_orig in grouped_orig:
        h, w = group_key
        color = color_dict[group_key]

        group_df_orig = group_df_orig.sort_values("eta")
        plt.plot(
            group_df_orig["eta"],
            group_df_orig["R2p"],
            "o",
            label=f"H₀={h:.2f}, W={w:.0f}",
            alpha=0.6,
            color=color,
        )

        if group_key in grouped_ext.groups:
            group_df_ext = grouped_ext.get_group(group_key).sort_values("eta")
            plt.plot(
                group_df_ext["eta"], group_df_ext["R2p"], "-", alpha=0.8, color=color
            )

    plt.xlabel("η₀ (m)", fontsize=12)
    plt.ylabel("R2pIndex (m)", fontsize=12)
    plt.suptitle("R2pIndex vs η₀ (by H₀, W_reef)", fontsize=14)
    plt.title(
        f"H0/L0={h0l0_val}, Cf={cf_val}, β_Beach={beta_beach_val}, β_ForeReef={beta_fore_val}",
        fontsize=10,
    )
    plt.grid(True)
    # plt.legend(fontsize=8)  # optionally re-enable legend
    plt.tight_layout()
    plt.show()


# %%
# Subplots according to H0
plot_count = 0

# Define fixed colormap
color_norm = mcolors.Normalize(vmin=0, vmax=1)
scalar_map = cm.ScalarMappable(
    norm=color_norm, cmap="tab10"
)  # or 'viridis', 'plasma', etc.

# Grouping values
H0L0_vals = [0.005, 0.025, 0.05]
Cf_vals = [0.01, 0.05, 0.1]
Beta_Beach_vals = [0.05, 0.1, 0.2]
Beta_ForeReef_vals = [0.05, 0.1, 0.5]

# Loop over all parameter combinations
for h0l0_val, cf_val, beta_beach_val, beta_fore_val in itertools.product(
    H0L0_vals, Cf_vals, Beta_Beach_vals, Beta_ForeReef_vals
):
    # Filter base and extended data
    base_mask = (
        np.isclose(df["H0L0"], h0l0_val)
        & np.isclose(df["Cf"], cf_val)
        & np.isclose(df["Beta_Beach"], beta_beach_val)
        & np.isclose(df["beta_ForeReef"], beta_fore_val)
    )
    df_filtered = df[base_mask]
    df_ext_filtered = df_extended[
        (df_extended["H0L0"] == h0l0_val)
        & (df_extended["Cf"] == cf_val)
        & (df_extended["Beta_Beach"] == beta_beach_val)
        & (df_extended["beta_ForeReef"] == beta_fore_val)
    ]

    if df_filtered.empty:
        continue

    # Unique H0 values for subplotting
    unique_H0s = sorted(df_filtered["H0"].unique())
    n_H0s = len(unique_H0s)

    fig, axes = plt.subplots(1, n_H0s, figsize=(5 * n_H0s, 5), sharey=True, sharex=True)
    if n_H0s == 1:
        axes = [axes]  # ensure iterable

    # Get consistent color mapping for W_reef
    unique_W_reef = sorted(df_filtered["W_reef"].unique())
    cmap = cm.get_cmap("viridis", len(unique_W_reef))
    color_dict = {w: cmap(i) for i, w in enumerate(unique_W_reef)}

    for ax, h_val in zip(axes, unique_H0s):
        # Filter for this H0
        df_h = df_filtered[df_filtered["H0"] == h_val]
        df_ext_h = df_ext_filtered[df_ext_filtered["H0"] == h_val]

        # Group by W_reef
        grouped_orig = df_h.groupby("W_reef")
        grouped_ext = df_ext_h.groupby("W_reef")

        for w_reef, group_df_orig in grouped_orig:
            color = color_dict[w_reef]
            group_df_orig = group_df_orig.sort_values("eta")
            ax.plot(
                group_df_orig["eta"],
                group_df_orig["R2p"],
                "o",
                label=f"W={w_reef:.0f}",
                alpha=0.6,
                color=color,
            )

            if w_reef in grouped_ext.groups:
                group_df_ext = grouped_ext.get_group(w_reef).sort_values("eta")
                ax.plot(
                    group_df_ext["eta"],
                    group_df_ext["R2p"],
                    "-",
                    alpha=0.8,
                    color=color,
                )

        ax.set_title(f"H₀ = {h_val:.2f}", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("R2pIndex (m)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("η₀ (m)", fontsize=12)

    fig.suptitle(
        f"R2pIndex vs η₀ by H₀\nH0/L0={h0l0_val}, Cf={cf_val}, β_Beach={beta_beach_val}, β_ForeReef={beta_fore_val}",
        fontsize=14,
    )
    axes[-1].legend(title="W_reef", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave space for suptitle
    plot_count += 1
    plt.show()


# %%
print(f"Total number of figures plotted: {plot_count}")

# %%
# Define fixed colormap
color_norm = mcolors.Normalize(vmin=0, vmax=1)
scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab10")

# Parameter groups
H0L0_vals = [0.005, 0.025, 0.05]
Cf_vals = [0.01, 0.05, 0.1]
Beta_Beach_vals = [0.05, 0.1, 0.2]
Beta_ForeReef_vals = [0.05, 0.1, 0.5]

plot_count = 0

# Loop over all combinations
for h0l0_val, cf_val, beta_beach_val, beta_fore_val in itertools.product(
    H0L0_vals, Cf_vals, Beta_Beach_vals, Beta_ForeReef_vals
):
    # 🔴 Skip unless Cf == 0.05 and Beta_Beach == 0.10
    if not (np.isclose(cf_val, 0.05) and np.isclose(beta_beach_val, 0.10)):
        continue

    # Filter base and extended data
    base_mask = (
        np.isclose(df["H0L0"], h0l0_val)
        & np.isclose(df["Cf"], cf_val)
        & np.isclose(df["Beta_Beach"], beta_beach_val)
        & np.isclose(df["beta_ForeReef"], beta_fore_val)
    )
    df_filtered = df[base_mask]
    df_ext_filtered = df_extended[
        (df_extended["H0L0"] == h0l0_val)
        & (df_extended["Cf"] == cf_val)
        & (df_extended["Beta_Beach"] == beta_beach_val)
        & (df_extended["beta_ForeReef"] == beta_fore_val)
    ]

    if df_filtered.empty:
        continue

    # Unique H0 values for subplotting
    unique_H0s = sorted(df_filtered["H0"].unique())
    n_H0s = len(unique_H0s)

    fig, axes = plt.subplots(1, n_H0s, figsize=(5 * n_H0s, 5), sharey=True, sharex=True)
    if n_H0s == 1:
        axes = [axes]  # ensure iterable

    # Get consistent color mapping for W_reef
    unique_W_reef = sorted(df_filtered["W_reef"].unique())
    cmap = cm.get_cmap("viridis", len(unique_W_reef))
    color_dict = {w: cmap(i) for i, w in enumerate(unique_W_reef)}

    for ax, h_val in zip(axes, unique_H0s):
        # Filter for this H0
        df_h = df_filtered[df_filtered["H0"] == h_val]
        df_ext_h = df_ext_filtered[df_ext_filtered["H0"] == h_val]

        # Group by W_reef
        grouped_orig = df_h.groupby("W_reef")
        grouped_ext = df_ext_h.groupby("W_reef")

        for w_reef, group_df_orig in grouped_orig:
            color = color_dict[w_reef]
            group_df_orig = group_df_orig.sort_values("eta")
            ax.plot(
                group_df_orig["eta"],
                group_df_orig["R2p"],
                "o",
                label=f"W={w_reef:.0f}",
                alpha=0.6,
                color=color,
            )

            if w_reef in grouped_ext.groups:
                group_df_ext = grouped_ext.get_group(w_reef).sort_values("eta")
                ax.plot(
                    group_df_ext["eta"],
                    group_df_ext["R2p"],
                    "-",
                    alpha=0.8,
                    color=color,
                )

        ax.set_title(f"H₀ = {h_val:.2f}", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("R2pIndex (m)", fontsize=12)
    for ax in axes:
        ax.set_xlabel("η₀ (m)", fontsize=12)

    fig.suptitle(
        f"R2pIndex vs η₀ by H₀\nH0/L0={h0l0_val}, Cf={cf_val}, β_Beach={beta_beach_val}, β_ForeReef={beta_fore_val}",
        fontsize=14,
    )
    axes[-1].legend(title="W_reef", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    # === Save figure as PDF ===
    filename = f"plot_H0L0_{h0l0_val}_Cf_{cf_val}_BetaBeach_{beta_beach_val}_BetaForeReef_{beta_fore_val}.pdf"
    fig.savefig(fig_path / filename, bbox_inches="tight")
    plt.show()

    plot_count += 1

print(f"Total number of figures plotted: {plot_count}")


# %% [markdown]
# ### Save extended BEWARE dataframe as new netcdf file

# %%
# Step 1: Define variables to keep
vars_to_keep = [
    "H0",
    "H0L0",
    "eta",
    "Cf",
    "beta_ForeReef",
    "Beta_Beach",
    "W_reef",
    "R2p",
]
fill_value = -9999.0  # Fill value compatible with Panoply

# Step 2: Rename columns to match NetCDF convention
df_original_trimmed = df[df["eta"] <= 3].copy()
df_extended_trimmed = df_extended[df_extended["eta"] > 3].copy()

# Rename columns to match NetCDF variable names
rename_map = {"eta": "eta0", "R2p": "R2pIndex", "Beta_Beach": "beta_Beach"}
df_original_trimmed = df_original_trimmed.rename(columns=rename_map)
df_extended_trimmed = df_extended_trimmed.rename(columns=rename_map)

# Add missing variables to extended df if not present
for var in rename_map.values():
    if var not in df_extended_trimmed.columns:
        df_extended_trimmed[var] = fill_value

# Concatenate the two dataframes
df_combined = pd.concat([df_original_trimmed, df_extended_trimmed], ignore_index=True)

# Fill any remaining NaNs with fill value
df_combined = df_combined.fillna(fill_value)

# Step 3: Create NetCDF file
with nc.Dataset(output_path, "w", format="NETCDF4") as ds_new:
    # Create dimension
    n_rows = df_combined.shape[0]
    ds_new.createDimension("ID", n_rows)

    # Write variables
    for var in [
        "H0",
        "H0L0",
        "eta0",
        "Cf",
        "beta_ForeReef",
        "beta_Beach",
        "W_reef",
        "R2pIndex",
    ]:
        var_data = df_combined[var].values.astype("f8")  # ensure float64
        var_out = ds_new.createVariable(var, "f8", ("ID",), fill_value=fill_value)
        var_out[:] = var_data
        var_out.units = "unknown"  # optionally set units or other attributes

    # Optionally add global attributes
    ds_new.description = (
        "Extended BEWARE database with extrapolated R2pIndex values for eta0 > 3"
    )
    ds_new.source = "Generated by extrapolation script"

# %% [markdown]
# ### Save extended as new dataframe

# %%
df_extended

# %%
# Step 1: Define variables to keep
vars_to_keep = [
    "H0",
    "H0L0",
    "eta",
    "Cf",
    "beta_ForeReef",
    "Beta_Beach",
    "W_reef",
    "R2p",
]
fill_value = -9999.0  # Fill value compatible with Panoply

# Rename columns to match NetCDF variable names
rename_map = {"eta": "eta0", "R2p": "R2pIndex", "Beta_Beach": "beta_Beach"}
df_extended_trimmed = df_extended.rename(columns=rename_map)

# Add missing variables to extended df if not present
for var in rename_map.values():
    if var not in df_extended_trimmed.columns:
        df_extended_trimmed[var] = fill_value

# Fill any remaining NaNs with fill value
df_extended_trimmed = df_extended_trimmed.fillna(fill_value)

# Step 3: Create NetCDF file
with nc.Dataset(output_path, "w", format="NETCDF4") as ds_new:
    # Create dimension
    n_rows = df_extended_trimmed.shape[0]
    ds_new.createDimension("ID", n_rows)

    # Write variables
    for var in [
        "H0",
        "H0L0",
        "eta0",
        "Cf",
        "beta_ForeReef",
        "beta_Beach",
        "W_reef",
        "R2pIndex",
    ]:
        var_data = df_extended_trimmed[var].values.astype("f8")  # ensure float64
        var_out = ds_new.createVariable(var, "f8", ("ID",), fill_value=fill_value)
        var_out[:] = var_data
        var_out.units = "unknown"  # optionally set units or other attributes

    # Optionally add global attributes
    ds_new.description = (
        "Extended BEWARE database with extrapolated R2pIndex values for eta0 > 3"
    )
    ds_new.source = "Generated by extrapolation script"
