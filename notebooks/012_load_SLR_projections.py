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
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# %% [markdown]
# ### Load and plot one IPCC regional SLR file
# Note that you need to have first downloaded the data.
# To do so, you can execute `102_download_AR6_SLR_projections.ipnyb`
# under `/additional_notebooks`

# %%
# Load NetCDF
nc_path = (
    RAW_DIR
    / "external/AR6_regional_SLR_projections"
    / "confidence_output_files/low_confidence"
    / "ssp585/total_ssp585_low_confidence_values.nc"
)
nc = Dataset(nc_path, mode="r")

# Extract variables
lats = nc.variables["lat"][:]
lons = nc.variables["lon"][:]
years = nc.variables["years"][:]
quantiles = nc.variables["quantiles"][:]
slr_raw = nc.variables["sea_level_change"]

print(f"\nLatitude range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")


# %%
# Get index for 2150 and median quantile (0.5)
year_idx = np.where(years == 2150)[0][0]
quantile_idx = np.where(quantiles == 0.50)[0][0]

# Extract SLR data
slr_data = slr_raw[quantile_idx, year_idx, :].astype(float)
slr_data[slr_data == slr_raw._FillValue] = np.nan

# Convert to meters
slr_data_m = slr_data / 1000

# Set color scale limits
vmin = -1
vmax = np.nanmax(slr_data_m)
vmax = 4

# Plot
plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
sc = ax.scatter(
    lons,
    lats,
    c=slr_data_m,
    cmap="viridis",
    s=2,
    transform=ccrs.PlateCarree(),
    vmin=vmin,
    vmax=vmax,
)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.set_global()
cb = plt.colorbar(sc, label="SLR (m)")
# cb.set_clim(vmin, vmax)  # optional; usually handled by `scatter()` already
plt.title("Sea Level Rise (m) in 2150 - Median (0.5 Quantile)")
plt.show()

# %% [markdown]
# Note: IPCC SLR Projections are not on a lat/lon 2D grid but instead lat/lon are both
# 1D arrays of length locations
