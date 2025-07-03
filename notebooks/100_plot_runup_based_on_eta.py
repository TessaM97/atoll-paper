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
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib import cm

# Load BEWARE dataset
file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"
dataset = nc.Dataset(file_path)

# Extract variables
eta0 = dataset.variables["eta0"][:]
R2pIndex = dataset.variables["R2pIndex"][:]

# Mask invalid values
# fill_eta = dataset.variables['eta0']._FillValue
# fill_r2p = dataset.variables['R2pIndex']._FillValue

# mask = (eta0 != fill_eta) & (R2pIndex != fill_r2p)

# Filter valid data
# eta0_valid = eta0[mask]
# R2p_valid = R2pIndex[mask]

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(eta0, R2pIndex, alpha=0.3, s=10, color="teal")
plt.xlabel("η₀ (m)", fontsize=12)
plt.ylabel("R2pIndex (m)", fontsize=12)
plt.title("R2pIndex as a function of η₀ from BEWARE", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Close dataset
dataset.close()


# %%
import matplotlib.pyplot as plt
import netCDF4 as nc

# Load dataset
file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"
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

# Apply filters
valid_mask = (
    ~np.isnan(W_reef)
    & ~np.isnan(beta_ForeReef)
    & ~np.isnan(H0)
    & ~np.isnan(eta0)
    & ~np.isnan(R2p)
    & (np.isclose(Cf, 0.05))
    & (np.isclose(beta_beach, 0.10))
    & (np.isclose(H0L0, 0.025))
)

# Apply mask
W_reef = W_reef[valid_mask]
beta = beta[valid_mask]
H0 = H0[valid_mask]
eta0 = eta0[valid_mask]
R2p = R2p[valid_mask]


# Group by beta, H0, W_reef with 5% tolerance
def round_tol(arr, tol=0.05):
    return np.round(arr / (tol * np.maximum(arr, 1e-6))) * (tol * np.maximum(arr, 1e-6))


group_keys = list(zip(round_tol(beta), round_tol(H0), round_tol(W_reef)))

from collections import defaultdict

grouped = defaultdict(list)

for key, eta, r2p in zip(group_keys, eta0, R2p):
    grouped[key].append((eta, r2p))

# Plot
plt.figure(figsize=(10, 6))
for i, ((b, h, w), vals) in enumerate(grouped.items()):
    vals = np.array(vals)
    if len(vals) < 5:
        continue
    sorted_idx = np.argsort(vals[:, 0])
    eta_sorted = vals[sorted_idx, 0]
    r2p_sorted = vals[sorted_idx, 1]
    plt.plot(
        eta_sorted, r2p_sorted, label=f"β={b:.2f}, H₀={h:.2f}, W={w:.0f}", alpha=0.7
    )

plt.xlabel("η₀ (m)", fontsize=12)
plt.ylabel("R2pIndex (m)", fontsize=12)
plt.suptitle("R2pIndex per η₀(grouped by β_ForeReef, H₀, W_reef)", fontsize=14)
plt.title("Assumptions: cf=0.05, β_Beach=0.10 and H0L0=0.025", fontsize=10)
# plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

ds.close()


# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Load dataset
file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"
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
beta = safe_var("beta_ForeReef")
H0 = safe_var("H0")
eta0 = safe_var("eta0")
R2p = safe_var("R2pIndex")
Cf = safe_var("Cf")
beta_beach = safe_var("beta_Beach")

# Apply filters
valid_mask = (
    ~np.isnan(W_reef)
    & ~np.isnan(beta)
    & ~np.isnan(H0)
    & ~np.isnan(eta0)
    & ~np.isnan(R2p)
    & (np.isclose(Cf, 0.05))
    & (np.isclose(beta_beach, 0.10))
)

# Apply mask
W_reef = W_reef[valid_mask]
beta = beta[valid_mask]
H0 = H0[valid_mask]
eta0 = eta0[valid_mask]
R2p = R2p[valid_mask]

# Group data
grouped = defaultdict(list)
for b, h, w, eta, r2p in zip(beta, H0, W_reef, eta0, R2p):
    grouped[(b, h, w)].append((eta, r2p))

# Unique values
unique_betas = sorted(set(beta))
unique_H0s = sorted(set(H0))
nrows = len(unique_betas)
ncols = len(unique_H0s)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows), sharex=True, sharey=True
)
axes = np.array(axes)

# Color map for W_reef
unique_W = sorted(set(W_reef))
colors = cm.viridis(np.linspace(0, 1, len(unique_W)))
color_map = dict(zip(unique_W, colors))

for i, b in enumerate(unique_betas):
    for j, h in enumerate(unique_H0s):
        ax = axes[i, j] if nrows > 1 and ncols > 1 else axes[max(i, j)]
        # Filter for (b, h)
        for (b_g, h_g, w), vals in grouped.items():
            if b_g == b and h_g == h and len(vals) >= 5:
                vals = np.array(vals)
                sorted_idx = np.argsort(vals[:, 0])
                eta_sorted = vals[sorted_idx, 0]
                r2p_sorted = vals[sorted_idx, 1]
                ax.plot(
                    eta_sorted,
                    r2p_sorted,
                    color=color_map[w],
                    label=f"W={w:.0f}",
                    alpha=0.7,
                )

        if i == nrows - 1:
            ax.set_xlabel("η₀ (m)")
        if j == 0:
            ax.set_ylabel("R2pIndex (m)")
        ax.set_title(f"β={b:.2f}, H₀={h:.2f}")
        ax.grid(True)

        # Legend (only on diagonal to reduce clutter)
        if i == j:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="upper right", title="W_reef")

plt.suptitle("R2pIndex vs η₀\n(rows = β, columns = H₀)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

ds.close()


# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from matplotlib import cm

# Load dataset
file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"
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
beta = safe_var("beta_ForeReef")
H0 = safe_var("H0")
eta0 = safe_var("eta0")
R2p = safe_var("R2pIndex")
Cf = safe_var("Cf")
beta_beach = safe_var("beta_Beach")

# Apply filters
valid_mask = (
    ~np.isnan(W_reef)
    & ~np.isnan(beta)
    & ~np.isnan(H0)
    & ~np.isnan(eta0)
    & ~np.isnan(R2p)
    & (np.isclose(Cf, 0.05))
    & (np.isclose(beta_beach, 0.10))
)

# Apply mask
W_reef = W_reef[valid_mask]
beta = beta[valid_mask]
H0 = H0[valid_mask]
eta0 = eta0[valid_mask]
R2p = R2p[valid_mask]

# Group data
grouped = defaultdict(list)
for b, h, w, eta, r2p in zip(beta, H0, W_reef, eta0, R2p):
    grouped[(b, h, w)].append((eta, r2p))

# Unique values
unique_betas = sorted(set(beta))
unique_H0s = sorted(set(H0))
nrows = len(unique_betas)
ncols = len(unique_H0s)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4 * nrows), sharex=True, sharey=True
)
axes = np.array(axes)

# Color map for W_reef
unique_W = sorted(set(W_reef))
colors = cm.viridis(np.linspace(0, 1, len(unique_W)))
color_map = dict(zip(unique_W, colors))

for i, b in enumerate(unique_betas):
    for j, h in enumerate(unique_H0s):
        ax = axes[i, j] if nrows > 1 and ncols > 1 else axes[max(i, j)]
        # Filter for (b, h)
        for (b_g, h_g, w), vals in grouped.items():
            if b_g == b and h_g == h and len(vals) >= 5:
                vals = np.array(vals)
                sorted_idx = np.argsort(vals[:, 0])
                eta_sorted = vals[sorted_idx, 0]
                r2p_sorted = vals[sorted_idx, 1]
                ax.plot(
                    eta_sorted,
                    r2p_sorted,
                    color=color_map[w],
                    label=f"W={w:.0f}",
                    alpha=0.7,
                )

        if i == nrows - 1:
            ax.set_xlabel("η₀ (m)")
        if j == 0:
            ax.set_ylabel("R2pIndex (m)")
        ax.set_title(f"β={b:.2f}, H₀={h:.2f}")
        ax.grid(True)

        # Legend (only on diagonal to reduce clutter)
        if i == j:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="upper right", title="W_reef")

plt.suptitle("R2pIndex vs η₀\n(rows = β, columns = H₀)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

ds.close()

# %%
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# Load dataset
file_path = "../data/BEWARE_Database.nc"
ds = nc.Dataset(file_path)


# Helper to extract variable with optional _FillValue masking
def safe_var(var_name):
    var = ds.variables[var_name]
    data = var[:]
    fill_val = var._FillValue if "_FillValue" in var.ncattrs() else None
    if fill_val is not None:
        data = np.where(data == fill_val, np.nan, data)
    return data


# Extract all variables once
W_reef = safe_var("W_reef")
beta_ForeReef = safe_var("beta_ForeReef")
H0 = safe_var("H0")
eta0 = safe_var("eta0")
R2p = safe_var("R2pIndex")
Cf = safe_var("Cf")
beta_beach = safe_var("beta_Beach")
H0L0 = safe_var("H0L0")


# Tolerance-based rounding for grouping
def round_tol(arr, tol=0.05):
    return np.round(arr / (tol * np.maximum(arr, 1e-6))) * (tol * np.maximum(arr, 1e-6))


# Value combinations
H0L0_vals = [0.005, 0.025, 0.05]
Cf_vals = [0.01, 0.05, 0.1]
Beta_Beach_vals = [0.05, 0.1, 0.2]

# Loop over all combinations
for h0l0_val, cf_val, beta_beach_val in itertools.product(
    H0L0_vals, Cf_vals, Beta_Beach_vals
):
    # Filter data
    valid_mask = (
        ~np.isnan(W_reef)
        & ~np.isnan(beta_ForeReef)
        & ~np.isnan(H0)
        & ~np.isnan(eta0)
        & ~np.isnan(R2p)
        & np.isclose(Cf, cf_val)
        & np.isclose(beta_beach, beta_beach_val)
        & np.isclose(H0L0, h0l0_val)
    )

    if not np.any(valid_mask):
        continue  # skip empty cases

    # Apply mask
    W_reef_f = W_reef[valid_mask]
    beta_f = beta_ForeReef[valid_mask]
    H0_f = H0[valid_mask]
    eta0_f = eta0[valid_mask]
    R2p_f = R2p[valid_mask]

    # Group by (β, H₀, W_reef)
    group_keys = list(zip(round_tol(beta_f), round_tol(H0_f), round_tol(W_reef_f)))

    grouped = defaultdict(list)
    for key, eta, r2p in zip(group_keys, eta0_f, R2p_f):
        grouped[key].append((eta, r2p))

    # Plot for this combination
    plt.figure(figsize=(10, 6))
    for (b, h, w), vals in grouped.items():
        vals = np.array(vals)
        if len(vals) < 5:
            continue
        sorted_idx = np.argsort(vals[:, 0])
        eta_sorted = vals[sorted_idx, 0]
        r2p_sorted = vals[sorted_idx, 1]
        plt.plot(
            eta_sorted, r2p_sorted, label=f"β={b:.2f}, H₀={h:.2f}, W={w:.0f}", alpha=0.7
        )

    plt.xlabel("η₀ (m)", fontsize=12)
    plt.ylabel("R2pIndex (m)", fontsize=12)
    plt.suptitle("R2pIndex vs η₀ (grouped by β_ForeReef, H₀, W_reef)", fontsize=14)
    plt.title(f"H0/L0={h0l0_val}, Cf={cf_val}, β_Beach={beta_beach_val}", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    # Uncomment to save each figure:
    # out_dir = "plots"
    # os.makedirs(out_dir, exist_ok=True)
    # plt.savefig(f"{out_dir}/plot_H0L0_{h0l0_val}_Cf_{cf_val}_BetaBeach_{beta_beach_val}.png")
    plt.show()

ds.close()


# %%
