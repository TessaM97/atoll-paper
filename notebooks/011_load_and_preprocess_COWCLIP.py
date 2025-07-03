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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.io import loadmat

# %%
mat = loadmat("../data/COWCLIP/ensMeans_robustnessMthdB2/robust_hs_p50_DJF.mat")
mat

# %%
# Extract variables
X = mat["Xq"]  # shape: (1, N)
Y = mat["Yq"]  # shape: (M, 1)
muEns = mat["muEns"]  # shape: (M, N)

# Inspect coordinate arrays
X = mat["Xq"]
Y = mat["Yq"]

# Print summary stats
print("X (longitude) range:", np.min(X), "to", np.max(X))
print("Y (latitude) range:", np.min(Y), "to", np.max(Y))

print("muEns shape:", muEns.shape)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# %%
# Transform into netcdf for further processing


# Extract 1D coordinate arrays from the 2D grids
lat_1d = Y[:, 0]  # shape: (361,)
lon_1d = X[0, :]  # shape: (721,)

# Make sure lon is in -180 to 180 range
# lon_1d = np.where(lon_1d > 180, lon_1d - 360, lon_1d)

# Sort lon if necessary
# if not np.all(np.diff(lon_1d) > 0):
#  lon_idx_sorted = np.argsort(lon_1d)
#  lon_1d = lon_1d[lon_idx_sorted]
# muEns = muEns[:, lon_idx_sorted]

# Create xarray Dataset
ds = xr.Dataset(
    {"muEns": (["lat", "lon"], muEns)},
    coords={"lat": lat_1d, "lon": lon_1d},
    attrs={
        "title": "Significant wave height (Hs) trends based on the 7-member ensemble for 1980–2014. DFJ season.",
        "source": "Converted from robust_hs_p50_DJF.mat",
    },
)


# %%
# Plot using pcolormesh
muEns = ds["muEns"].values  # shape (lat, lon)
lat = ds["lat"].values  # shape (lat,)
lon = ds["lon"].values  # shape (lon,)

# Create 2D meshgrid for pcolormesh
X, Y = np.meshgrid(lon, lat)

# Plot using pcolormesh
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(X, Y, muEns, shading="auto", cmap="viridis")

# Add colorbar and labels
plt.colorbar(c, label="Hs (m/yr)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    "Significant wave height (Hs) trends based on the 7-member ensemble for 1980–2014. DFJ season."
)
plt.tight_layout()
plt.show()

# %%
# Transform into netcdf for further processing


# Extract 1D coordinate arrays from the 2D grids
lat_1d = Y[:, 0]  # shape: (361,)
lon_1d = X[0, :]  # shape: (721,)

# Make sure lon is in -180 to 180 range

lon_1d = np.where(lon_1d > 180, lon_1d - 360, lon_1d)

# Sort lon if necessary
if not np.all(np.diff(lon_1d) > 0):
    lon_idx_sorted = np.argsort(lon_1d)
    lon_1d = lon_1d[lon_idx_sorted]
    muEns = muEns[:, lon_idx_sorted]

# Create xarray Dataset
ds = xr.Dataset(
    {"muEns": (["lat", "lon"], muEns)},
    coords={"lat": lat_1d, "lon": lon_1d},
    attrs={
        "title": "Significant wave height (Hs) trends based on the 7-member ensemble for 1980–2014. DFJ season.",
        "source": "Converted from robust_hs_p50_DJF.mat",
    },
)

# %%
# Transformed to -180 to 180
# Plot using pcolormesh
muEns = ds["muEns"].values  # shape (lat, lon)
lat = ds["lat"].values  # shape (lat,)
lon = ds["lon"].values  # shape (lon,)

# Print ranges to understand coordinate "shape"
print(f"\nLatitude range: {lat.min():.2f} to {lat.max():.2f}")
print(f"Longitude range: {lon.min():.2f} to {lon.max():.2f}")

# %%
# Create 2D meshgrid for pcolormesh
X, Y = np.meshgrid(lon, lat)

# Plot using pcolormesh
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(X, Y, muEns, shading="auto", cmap="viridis")

# Add colorbar and labels
plt.colorbar(c, label="Hs (m/yr)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    "Significant wave height (Hs) trends based on the 7-member ensemble for 1980–2014. DFJ season."
)
plt.tight_layout()
plt.show()

# %%
# Save to NetCDF for further processing

# Define the output file path
output_path = Path("../data/processed/COWCLIP_grid.nc")

# Delete the existing file if it exists
if output_path.exists():
    output_path.unlink()

# Save to NetCDF
ds.to_netcdf(output_path)
