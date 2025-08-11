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
import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

# %% [markdown]
# ### Load and Prepare BEWARE
#
# #### Select Cf = 0.05
# #### Select beta_Beach = 0.10

# %%
# Enable tqdm with pandas apply
tqdm.pandas()

# Load BEWARE dataset
file_path = "/Users/tessamoller/Documents/atoll-slr-paper-data/data/BEWARE_Database.nc"
dataset = nc.Dataset(file_path)

# Extract variables
W_reef_db = dataset.variables["W_reef"][:]
beta_foreReef_db = dataset.variables["beta_ForeReef"][:]
H0_db = dataset.variables["H0"][:]
eta0_db = dataset.variables["eta0"][:]
R2pIndex_db = dataset.variables["R2pIndex"][:]
Cf_db = dataset.variables["Cf"][:]
beta_beach_db = dataset.variables["beta_Beach"][:]
H0L0_db = dataset.variables["H0L0"][:]

# Convert to structured array
beware_array = np.rec.fromarrays(
    [
        W_reef_db,
        beta_foreReef_db,
        H0_db,
        eta0_db,
        R2pIndex_db,
        Cf_db,
        beta_beach_db,
        H0L0_db,
    ],
    names="W_reef,beta_ForeReef,H0,eta0,R2pIndex,Cf,beta_Beach,H0L0",
)

# Define eta columns and result column names
eta_dict = {
    "eta_SLR": "R2pIndex_SLR",
    "eta_combined_rp1": "R2pIndex_combined_rp1",
    "eta_combined_rp10": "R2pIndex_combined_rp10",
    "eta_combined_rp100": "R2pIndex_combined_rp100",
}


# Matching function for a given eta value
def match_with_eta(row, eta_value):
    valid = beware_array[
        (beware_array.W_reef > 1)
        & (np.isclose(beware_array.Cf, 0.05))
        & (np.isclose(beware_array.beta_Beach, 0.10))
    ]

    if len(valid) == 0:
        return np.nan, np.nan

    d_beta = np.abs(valid.beta_ForeReef - row["beta_f"])
    d_H0 = np.abs(valid.H0 - row["H0"])
    d_H0L0 = np.abs(valid.H0L0 - row["H0L0"])
    d_eta = np.abs(valid.eta0 - eta_value)
    d_Wreef = np.abs(valid.W_reef - row["W_reef"])

    scores = (
        d_beta / (np.max(d_beta) + 1e-6)
        + d_H0 / (np.max(d_H0) + 1e-6)
        + d_H0L0 / (np.max(d_H0L0) + 1e-6)
        + d_eta / (np.max(d_eta) + 1e-6)
        + d_Wreef / (np.max(d_Wreef) + 1e-6)
    )

    best_idx = np.argmin(scores)
    return (
        valid.R2pIndex[best_idx],
        scores[best_idx],
        valid.Cf[best_idx],
        valid.beta_Beach[best_idx],
        valid.H0L0[best_idx],
        row.get("transect_id", np.nan),
    )


# Apply function with progress bar
def apply_all_matches(row):
    results = {}

    for eta_col, output_col in eta_dict.items():
        eta_val = row[eta_col]
        r2p, score, cf, beta_beach, h0l0, transect_i = match_with_eta(row, eta_val)

        # Base output name, e.g., from "R2pIndex_combined_rp1" → "rp1"
        label = output_col.split("_")[-1]

        results["transect_id"] = row["transect_id"]
        results[output_col] = r2p
        results[f"score_{label}"] = score
        results['year'] = row["year"]
        results['scenario'] = row["scenario"]
        results['confidence'] = row["confidence"]
        results['quantile'] = row["quantile"]
        
    # results[f"Cf_{label}"] = cf
    # results[f"beta_Beach_{label}"] = beta_beach
    # results[f"H0L0_{label}"] = h0l0
    #  results[f"transect_i_{label}"] = transect_i

    return pd.Series(results)


# %%
# Load combined input data

#inputs = pd.read_parquet("../data/Atoll_BEWARE_inputs.parquet")
inputs = pd.read_parquet("/Users/tessamoller/Documents/atoll-slr-paper-data/data/Atoll_BEWARE_inputs.parquet")

# Select one scenario/timeframe
inputs = inputs[
    (inputs["scenario"] == "ssp119")
    & (inputs["year"].isin([2080,2090]))
    & (inputs["quantile"] == 0.50)
]
# inputs = inputs.iloc[:10, ]

# Apply BEWARE to input data

# Apply matching function
results = inputs.progress_apply(apply_all_matches, axis=1, result_type="expand")
results["transect_id"] = results["transect_id"].astype(int)
results
# Add to GeoDataFrame


outputs = pd.merge(
    inputs[["transect_id", "FID_GADM", "Atoll_FID", "year", "scenario", "confidence", "quantile"]],
    results,
    on=["transect_id", "FID_GADM", "Atoll_FID", "year", "scenario", "confidence", "quantile"],
    how="left",
)
outputs

# Close dataset
dataset.close()

# %%
#print(results.columns, inputs.columns)
outputs

# %%
#outputs
#79604
#11372
outputs[outputs['transect_id']==0]

# %%
## Check how much it deviates

# Ensure R2pIndex_SLR is numeric
outputs["R2pIndex_SLR"] = pd.to_numeric(outputs["R2pIndex_SLR"], errors="coerce")

# Group by Atoll_FID and compute stats on R2pIndex_SLR
r2p_summary = (
    outputs.groupby("Atoll_FID")["R2pIndex_SLR"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
        range=lambda x: x.max() - x.min(),
    )
    .reset_index()
)

# Sort by deviation (range or std)
r2p_summary_sorted = r2p_summary.sort_values(by="std", ascending=False)

# Display
print(r2p_summary_sorted.head(10))  # Top 10 atolls with largest variation

# %%
# Check how much it deviates

# Ensure R2pIndex_SLR is numeric
outputs["R2pIndex_SLR"] = pd.to_numeric(outputs["R2pIndex_SLR"], errors="coerce")

# Group by Atoll_FID and compute stats on R2pIndex_SLR
r2p_summary = (
    outputs.groupby("FID_GADM")["R2pIndex_SLR"]
    .agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
        range=lambda x: x.max() - x.min(),
    )
    .reset_index()
)

# Sort by deviation (range or std)
r2p_summary_sorted = r2p_summary.sort_values(by="std", ascending=False)

# Display
print(r2p_summary_sorted.head(10))  # Top 10 atolls with largest variation

# %%
outputs.round(2)

# %%
outputs.round(2).to_parquet(
    "../data/processed/Atoll_BEWARE_processed_outputs_2020-2150.parquet", index=False
)

# %%
outputs.round(2).to_csv(
    "../data/processed/Atoll_BEWARE_processed_outputs_2020-2150.csv", index=False
)

# %%
