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
import glob
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, INTERIM_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# %% [markdown]
# ## Combine input datasets to create input dataframe

# %% [markdown]
# ### Load Shapefile

# %%
shapefile_path = os.path.join(INTERIM_DIR, "Shapefiles/Atoll_transects_centroids.shp")

gdf = gpd.read_file(shapefile_path)
gdf

# %% [markdown]
# ### Load COWCLIP

# %%
# Load datasets

COWCLIP_H0_path = os.path.join(
    INTERIM_DIR, "external/COWCLIP/COWCLIP_ensemble_mean_Hs_1995_2014.nc"
)

COWCLIP_Tm_path = os.path.join(
    INTERIM_DIR, "external/COWCLIP//COWCLIP_ensemble_mean_Tm_1995_2014.nc"
)


COWCLIP_H0 = xr.open_dataset(COWCLIP_H0_path)
COWCLIP_Tm = xr.open_dataset(COWCLIP_Tm_path)

# Extract lat/lon grid
lat = COWCLIP_H0["latitude"].values
lon = COWCLIP_H0["longitude"].values

# Create a dictionary of quantile names and their corresponding suffixes
quantile_map = {
    0.17: ("hs_p17", "tm_p17"),
    0.5: ("hs_p50", "tm_p50"),
    0.83: ("hs_p83", "tm_p83"),
}


# Compute wavelength using deepwater linear wave theory
def calculate_wave_length_L0(Tm, g=9.81):
    """Compute deepwater wavelength (L0) using linear wave theory."""
    return float((g * Tm**2) / (2 * np.pi))


# Function to extract nearest value for all centroids
def extract_wave_quantiles(gdf, quantile_map, COWCLIP_H0, COWCLIP_Tm, lat, lon):
    def find_nearest_index(array, value):
        return np.abs(array - value).argmin()

    coords = np.array([(geom.y, geom.x) for geom in gdf.geometry])
    records = []

    for i, (lat_pt, lon_pt) in enumerate(coords):
        lat_idx = find_nearest_index(lat, lat_pt)
        lon_idx = find_nearest_index(lon, lon_pt)

        for q, (hs_var, tm_var) in quantile_map.items():
            hs_val = COWCLIP_H0[hs_var].values[lat_idx, lon_idx]
            tm_val = COWCLIP_Tm[tm_var].values[lat_idx, lon_idx]
            l0_val = calculate_wave_length_L0(tm_val)
            records.append(
                {
                    "transect_i": gdf.iloc[i]["transect_i"],
                    "quantile": q,
                    "H0": hs_val,
                    "Tm": tm_val,
                    "L0": l0_val,
                    "H0L0": hs_val / l0_val if tm_val > 0 else np.nan,
                }
            )

    return pd.DataFrame.from_records(records)


# Apply extraction
wave_df = extract_wave_quantiles(gdf, quantile_map, COWCLIP_H0, COWCLIP_Tm, lat, lon)
wave_df

# %% [markdown]
# ### Load IPCC SLR Projections

# %%
# Base path to search in

base_dir = os.path.join(
    RAW_DIR, "external/AR6_regional_SLR_projections/confidence_output_files"
)

# Recursive glob pattern to find files starting with "total_ssp" and ending in ".nc"
pattern = os.path.join(base_dir, "**", "total_ssp*.nc")
nc_files = glob.glob(pattern, recursive=True)

# Extract metadata
file_records = []

for file_path in nc_files:
    filename = os.path.basename(file_path)

    # Ensure filename contains expected scenario structure
    if not filename.startswith("total_ssp"):
        continue  # extra guard

    if "rates" in file_path:
        continue

    # Extract scenario (e.g., ssp126, ssp585)
    scenario = next((p for p in filename.split("_") if p.startswith("ssp")), "unknown")

    # Extract confidence from directory path
    confidence = "unknown"
    for level in ["low_confidence", "medium_confidence"]:
        if level in file_path:
            confidence = level.replace("_confidence", "")
            break

    file_records.append(
        {"file_path": file_path, "confidence": confidence, "scenario": scenario}
    )

# Convert to a DataFrame for convenience
slr_file_df = pd.DataFrame(file_records)

# Display
print(f"Found {len(slr_file_df)} matching SLR files:")

# %%
# Combine IPCC SLR with GDF

# Extract (lat, lon) of centroids
centroid_coords = np.array([(geom.y, geom.x) for geom in gdf.geometry])
transect_ids = gdf["transect_i"].values

# Quantiles of interest
target_quantiles = [0.17, 0.5, 0.83]
target_years = [
    2020,
    2030,
    2040,
    2050,
    2060,
    2070,
    2080,
    2090,
    2100,
    2110,
    2120,
    2130,
    2140,
    2150,
]

# Container to hold all rows
records = []

for _, row in slr_file_df.iterrows():
    file_path = row["file_path"]
    scenario = row["scenario"]
    confidence = row["confidence"]

    print(f"Processing: {scenario} ({confidence})")
    # Load NetCDF
    ds = xr.open_dataset(file_path)

    # Coordinates
    slr_lat = ds["lat"].values
    slr_lon = ds["lon"].values
    slr_coords = np.column_stack((slr_lat, slr_lon))

    # KDTree for fast nearest-neighbor lookup
    tree = cKDTree(slr_coords)
    _, nearest_idx = tree.query(centroid_coords)

    # Extract years and quantiles
    years = ds["years"].values
    quantiles = ds["quantiles"].values
    slr_values = ds["sea_level_change"]  # shape: (quantiles, years, locations)

    # Loop over desired quantiles and years
    for q in target_quantiles:
        if q not in quantiles:
            print(f"  Warning: Quantile {q} not in file {file_path}")
            continue
        q_idx = np.where(np.isclose(quantiles, q))[0][0]

        for y_idx, year in enumerate(years):
            if year not in target_years:
                continue  # 🔴 Skip unwanted years

            slr_slice = slr_values.isel(quantiles=q_idx, years=y_idx).values.astype(
                float
            )

            # Handle fill values if needed
            fill_val = slr_values.attrs.get("_FillValue", np.nan)
            slr_slice = np.where(slr_slice == fill_val, np.nan, slr_slice)

            # Collect SLR at each transect location
            for i, tid in enumerate(transect_ids):
                slr_mm = slr_slice[nearest_idx[i]]
                slr_m = slr_mm / 1000.0 if np.isfinite(slr_mm) else np.nan
                records.append(
                    {
                        "transect_i": tid,
                        "scenario": scenario,
                        "confidence": confidence,
                        "year": year,
                        "quantile": q,
                        "eta_SLR": slr_m,
                    }
                )

    ds.close()  # Free memory

# Convert to DataFrame
df_slr_all = pd.DataFrame.from_records(records)

# Preview
print(df_slr_all.head())


# %%
# Step 1: Filter original DataFrame
filtered = df_slr_all[
    (df_slr_all["confidence"] == "medium")
    & (df_slr_all["scenario"] == "ssp119")
    & (df_slr_all["year"] == 2020)
    & (df_slr_all["quantile"] == 0.50)
]

# Step 2: Create modified copy
modified = filtered.copy()
modified["year"] = 2005
modified["eta_SLR"] = 0.0
modified["scenario"] = "baseline"
modified
# Step 3: Append to original DataFrame

df_slr_all = pd.concat([df_slr_all, modified], ignore_index=True)


# %% [markdown]
# ### Load COAST-RP

# %%
# Open COAST-RP dataset
COASTRP_path = os.path.join(RAW_DIR, "external/COAST-RP/COAST-RP.nc")

COASTRP = xr.open_dataset(COASTRP_path)


# Coordinates of storm tide stations
x = COASTRP["station_x_coordinate"].values
y = COASTRP["station_y_coordinate"].values
station_coords = np.column_stack((y, x))  # lat, lon

# Build KDTree for fast nearest-neighbor lookup
tree = cKDTree(station_coords)

# Coordinates of transects
centroid_coords = np.array([(geom.y, geom.x) for geom in gdf.geometry])
transect_ids = gdf["transect_i"].values

# Map storm tide values from each return period
storm_tide_rps = {
    "storm_tide_rp1": "storm_tide_rp_0001",
    "storm_tide_rp10": "storm_tide_rp_0010",
    "storm_tide_rp100": "storm_tide_rp_0100",
}

# Prepare dictionary to hold storm tide values per return period
storm_tide_data = {}

for label, var_name in storm_tide_rps.items():
    print(f"Processing: {label} ({var_name})")

    tide_values = COASTRP[var_name].values

    # Handle _FillValue
    fill_value = COASTRP[var_name].attrs.get("_FillValue", np.nan)
    tide_values = np.where(tide_values == fill_value, np.nan, tide_values)

    # Map nearest tide value to each transect
    mapped_values = tide_values[tree.query(centroid_coords)[1]]
    storm_tide_data[label] = mapped_values

    # Optional: add to gdf for inspection
    gdf[label + "_m"] = mapped_values

# Add storm tide to df_slr_all by transect_i
for label, values in storm_tide_data.items():
    tide_map = dict(zip(transect_ids, values))
    tide_map = {int(k): float(v) for k, v in tide_map.items()}
    df_slr_all["transect_i"] = df_slr_all["transect_i"].astype(int)
    df_slr_all[label] = df_slr_all["transect_i"].map(tide_map)

# Compute combined water level for each return period
df_slr_all["eta_combined_rp1"] = df_slr_all["eta_SLR"] + df_slr_all["storm_tide_rp1"]
df_slr_all["eta_combined_rp10"] = df_slr_all["eta_SLR"] + df_slr_all["storm_tide_rp10"]
df_slr_all["eta_combined_rp100"] = (
    df_slr_all["eta_SLR"] + df_slr_all["storm_tide_rp100"]
)

# Merge with wave data
df_slr_all = df_slr_all.merge(wave_df, on=["transect_i", "quantile"], how="left")

# Preview
df_slr_all.head()


# %%
# gdf['dist_to_station_km'].max()
# gdf['dist_to_station_km'].max()

# %% [markdown]
# ### Prepare for BEWARE
#
#
# #### eta = offshore water levels : combine IPCC SLR + COAST_RP [m]
# #### Wreef = reef width : beach_width [m]
# #### betaf = fore reef slope: fore_reef_width / 25
# #### Hs = offshore significant wave height : COWCLIP [m]

# %%
gdf["W_reef"] = gdf["beach_widt"]
gdf["beta_f"] = 25 / gdf["Fore_reef_"]
# gdf
gdf

# %%
df_inputs = gdf[["transect_i", "Atoll_FID", "FID_GADM", "geometry", "W_reef", "beta_f"]]
df_all = df_slr_all.merge(df_inputs, on="transect_i", how="left")
df_all


# %%
def round_to_sigfig(x, sigfigs):
    if pd.isnull(x) or x == 0:
        return x
    # decimals = sigfigs - int(np.floor(np.log10(abs(x)))) - 1
    return round(x, sigfigs)


# Define columns and required significant figures
sigfig_map = {
    "eta_SLR": 2,
    "eta_combined_rp1": 2,
    "eta_combined_rp10": 2,
    "eta_combined_rp100": 2,
    "beta_f": 3,
    "H0": 1,
    "H0L0": 4,
    "w_reef": 1,
}

BEWARE_inputs = df_all[
    [
        "transect_i",
        "FID_GADM",
        "Atoll_FID",
        "scenario",
        "year",
        "confidence",
        "quantile",
        "eta_SLR",
        "eta_combined_rp1",
        "eta_combined_rp10",
        "eta_combined_rp100",
        "H0",
        "H0L0",
        "W_reef",
        "beta_f",
    ]
].copy()


# Apply formatting
for col, sig in sigfig_map.items():
    if col in BEWARE_inputs.columns:
        BEWARE_inputs[col] = BEWARE_inputs[col].apply(lambda x: round_to_sigfig(x, sig))

BEWARE_inputs.rename(columns={"transect_i": "transect_id"}, inplace=True)


# %%
BEWARE_inputs

# %%
output_path = os.path.join(
    INTERIM_DIR,
    "Atoll_BEWARE_inputs.parquet",
)

BEWARE_inputs.to_parquet(output_path, index=False, engine="pyarrow")
