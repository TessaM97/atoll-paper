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

import numpy as np
import xarray as xr

# -------- SETTINGS --------
data_folder = "/Users/tessamoller/Documents/atoll-slr-paper-data/data/large_datasets/ORAS5_sossheig"  # 🔁 Update this path
variable_name = "sossheig"  # ORAS5 sea surface height variable

# Site coordinates: (latitude, longitude)
sites = {
    "Tarawa, Kiribati": (1.1940, 172.5837),
    "Nanumea, Tuvalu": (-5.6612, 176.1035),
    "Funafuti, Tuvalu": (-8.5155, 179.113),
    "Nukulaelae, Tuvalu": (-9.3872, 179.8424),
    "Nui, Tuvalu": (-7.2258, 177.1546),
}

# --------------------------

# Step 1: Load all NetCDF files
all_files = sorted(glob.glob(f"{data_folder}/*.nc"))
ds = xr.open_mfdataset(all_files, combine="by_coords", parallel=True)

# Step 2: Define baseline (<= 2014) and target (March 2015) periods
baseline_ds = ds.sel(time_counter=ds.time_counter.dt.year <= 2014)
march_2015_ds = ds.sel(
    time_counter=(ds.time_counter.dt.year == 2015) & (ds.time_counter.dt.month == 3)
)

# Step 3: Compute overall mean across all months for 1995–2014
baseline_mean = baseline_ds[variable_name].mean(dim="time_counter")

# Step 4: Extract March 2015 value (assumes one time step for March 2015)
march_2015 = march_2015_ds[variable_name].isel(time_counter=0)

# Step 5: Compute anomaly (March 2015 minus baseline mean)
anomaly = march_2015 - baseline_mean


# Step 6: Function to find nearest grid point index in 2D lat/lon arrays
def find_nearest_index(lat_arr, lon_arr, site_lat, site_lon):
    dist = (lat_arr - site_lat) ** 2 + (lon_arr - site_lon) ** 2
    return np.unravel_index(dist.argmin(), dist.shape)  # returns (y_idx, x_idx)


# Step 7: Extract anomaly at each site
print(
    "March 2015 Sea Surface Height Anomaly (m) relative to 1995–2014 full-period mean:"
)
for site_name, (lat, lon) in sites.items():
    y_idx, x_idx = find_nearest_index(ds.nav_lat.values, ds.nav_lon.values, lat, lon)
    try:
        value = anomaly.values[y_idx, x_idx]
        print(f"{site_name}: {value:.4f} m")
    except Exception as e:
        print(f"{site_name}: Error: {e}")


# %%
