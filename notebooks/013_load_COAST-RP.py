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
import netCDF4 as nc
import numpy as np
from netCDF4 import Dataset

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# %% [markdown]
# ### Load and plot COAST-RP return periods
# Note that you need to have first downloaded the data. 
# To do so, you can execute `103_download_COAST-RP.ipnyb` 
# under `/additional_notebooks`

# %%
# Load COAST-RP NetCDF file
nc_path = RAW_DIR / "external/COAST-RP/COAST-RP.nc"
nc = Dataset(nc_path, mode="r")

# Extract coordinates and values
x = nc.variables["station_x_coordinate"][:]  # longitudes
y = nc.variables["station_y_coordinate"][:]  # latitudes
storm_tide_var = nc.variables["storm_tide_rp_0050"][:]  # 50-year return period


print(f"\nLatitude range: {y.min():.2f} to {y.max():.2f}")
print(f"Longitude range: {x.min():.2f} to {x.max():.2f}")

# %%
storm_tide = storm_tide_var.astype(float)
if hasattr(storm_tide_var, "_FillValue"):
    storm_tide[storm_tide == storm_tide_var._FillValue] = np.nan
elif hasattr(storm_tide_var, "missing_value"):
    storm_tide[storm_tide == storm_tide_var.missing_value] = np.nan


# vmin = 0.0
# vmax = np.nanmax(storm_tide)


plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

sc = ax.scatter(
    x,
    y,
    c=storm_tide,
    cmap="viridis",
    s=2,
    transform=ccrs.PlateCarree(),
    # vmin=vmin,
    # vmax=vmax
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")

# If you want to show the whole globe, keep set_global();
# otherwise, comment it out to let Cartopy choose a default view.
ax.set_global()

plt.colorbar(sc, label="Storm Tide (50-yr RP in m)")
plt.title("COAST-RP: 50-Year Return Period Storm Tide")
plt.show()
