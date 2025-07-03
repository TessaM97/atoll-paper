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
import netCDF4 as nc
import numpy as np
import xarray as xr
from netCDF4 import Dataset, num2date

# %% [markdown]
# ## Load ERA5 data and transform longitude and time

# %%
# Load NetCDF

nc_path = "../data/ERA5/data_0.nc"
nc = Dataset(nc_path, mode="r")

# Extract variables
lats = nc.variables["latitude"][:]
lat_1d = nc.variables["latitude"][:]
lons = nc.variables["longitude"][:]
time = nc.variables["valid_time"][:]
swh = nc.variables["swh"][:]  # significant wave height
pp1d = nc.variables["pp1d"][:]  # peak wave period

print(f"\nLatitude range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")


print("lat_1d shape:", lats.shape)
print("lon_1d shape:", lons.shape)
print("time shape:", time.shape)
print("swh shape:", swh.shape)
print("pp1 shape:", swh.shape)

# %%
# Convert lons from [0, 360] to [-180, 180]
lons = np.where(lons > 180, lons - 360, lons)

# Sort longitudes and reorder data accordingly
lon_sorted_idx = np.argsort(lons)
lons_sorted = lons[lon_sorted_idx]

# Reorder swh and pp1d along the longitude axis (last dimension)
swh_sorted = swh[:, :, lon_sorted_idx]
pp1d_sorted = pp1d[:, :, lon_sorted_idx]


# %%
# Read time variable and its units
time_var = nc.variables["valid_time"]
time_units = time_var.units  # e.g., 'seconds since 1970-01-01 00:00:00'
calendar = time_var.calendar if hasattr(time_var, "calendar") else "standard"

# Convert time to datetime objects
time_dates = num2date(time[:], units=time_units, calendar=calendar)

# Optional: convert to np.datetime64 if using xarray
time_np = np.array(time_dates).astype("datetime64[ns]")

# %%
# Transformed dataframe
ds = xr.Dataset(
    {
        "swh": (["time", "lat", "lon"], swh_sorted),
        "pp1d": (["time", "lat", "lon"], pp1d_sorted),
    },
    coords={"lat": lats, "lon": lons_sorted, "time": time_np},
    attrs={
        "title": "Monthly Significant wave height (swh) and Peak wave period (pp1d) for 1990–2005 from ERA5 reanalysis.",
        "source": "Converted from data_0.nc",
    },
)


# Optional: Check new lon range
print(
    f"Transformed Longitude range: {ds.lon.min().item():.2f} to {ds.lon.max().item():.2f}"
)

# %% [markdown]
# ### Plotting as test

# %%
# Select data for January 2000 and remove extra time dimension
swh_jan2000 = ds.sel(time="2000-01")["swh"].squeeze().values  # shape: (361, 720)

lat = ds["lat"].values
lon = ds["lon"].values

# Create 2D meshgrid
X, Y = np.meshgrid(lon, lat)

# Plot
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(X, Y, swh_jan2000, shading="auto", cmap="viridis")

# Add colorbar and labels
plt.colorbar(c, label="Significant Wave Height (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Significant Wave Height (swh) - January 2000")
plt.tight_layout()
plt.show()


# %%
# Extract variables
lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
time = ds.variables["time"][:]
swh = ds.variables["swh"][:]  # significant wave height
pp1d = ds.variables["pp1d"][:]  # peak wave period

print(f"\nLatitude range: {lat.min():.2f} to {lat.max():.2f}")
print(f"Longitude range: {lon.min():.2f} to {lon.max():.2f}")


print("lat_1d shape:", lat.shape)
print("lon_1d shape:", lon.shape)
print("time shape:", time.shape)
print("swh shape:", swh.shape)
print("pp1 shape:", swh.shape)

# %% [markdown]
# ### Select maximum in each year for shw and select maximum as annual value

# %%
# 1. Identify land points (all-NaN) so we can restore them as NaN later
mask = ~np.all(np.isnan(ds["swh"]), axis=0)

# 2. Replace NaNs with -inf so argmax can run without error
ds2 = ds.fillna(-np.inf)

# 3. Group by year and select the time-index of max swh
#    (this yields a new Dataset with dims (year, lat, lon))
annual = ds2.groupby("time.year").apply(
    lambda x: x.isel(time=x["swh"].argmax(dim="time"))
)

# 4. Mask out the original land points (set them back to NaN)
annual = annual.where(mask)

# 5. Rename variables and coordinate as requested
annual = annual.rename(
    {"swh": "swh_max", "pp1d": "pp1d_at_swh_max", "time": "max_time"}
)

# Now `annual` has dims (year, lat, lon) with:
#   - swh_max(year,lat,lon): the annual max swh
#   - pp1d_at_swh_max(year,lat,lon): pp1d at that time
#   - max_time(year,lat,lon): the time of the max swh


# %%
# Select data for January 2000 and remove extra time dimension
annual_2000 = annual.sel(year=2000)["swh_max"].values  # shape: (361, 720)

lat = annual["lat"].values
lon = annual["lon"].values

# Create 2D meshgrid
X, Y = np.meshgrid(lon, lat)

# Plot
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(X, Y, annual_2000, shading="auto", cmap="viridis")

# Add colorbar and labels
plt.colorbar(c, label="Significant Wave Height (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Significant Wave Height (swh) - 2000")
plt.tight_layout()
plt.show()

# %%
# Select data for January 2000 and remove extra time dimension
annual_2000 = annual.sel(year=2000)["pp1d_at_swh_max"].values  # shape: (361, 720)

lat = annual["lat"].values
lon = annual["lon"].values

# Create 2D meshgrid
X, Y = np.meshgrid(lon, lat)

# Plot
plt.figure(figsize=(12, 6))
c = plt.pcolormesh(X, Y, annual_2000, shading="auto", cmap="viridis")

# Add colorbar and labels
plt.colorbar(c, label="Peak wave period (m)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Peak Wave Period (PP1D) at Significant Wave Height Maximum (swh) - 2000")
plt.tight_layout()

# %%
## Find maximum in period 1995-2014

# Select only the time range 1995-2014
ds_period = ds.sel(time=slice("1995-01-01", "2014-12-31"))

mask = ~np.all(np.isnan(ds_period["swh"]), axis=0)

ds2 = ds_period.fillna(-np.inf)

# Find index of max SWH along time
swh_argmax = ds2["swh"].argmax(dim="time")

# Select corresponding values
swh_max = ds2["swh"].isel(time=swh_argmax)
pp1d_at_swh_max = ds2["pp1d"].isel(time=swh_argmax)
max_time = ds2["time"].isel(time=swh_argmax)

swh_max = swh_max.where(mask)
pp1d_at_swh_max = pp1d_at_swh_max.where(mask)
max_time = max_time.where(mask)

extreme = xr.Dataset(
    {
        "swh_max": swh_max,
        "pp1d_at_swh_max": pp1d_at_swh_max,
    },
    coords={"lat": ds["lat"], "lon": ds["lon"], "max_time": max_time},
)

# %%
## Find maximum in period 1995-2014

# Select only the time range 1995-2014
ds_period = ds.sel(time=slice("1995-01-01", "2014-12-31"))

mask = ~np.all(np.isnan(ds_period["swh"]), axis=0)

ds2 = ds_period.fillna(-np.inf)

# Find index of max pp1d along time
swh_argmax = ds2["swh"].argmax(dim="time")
pp1d_argmax = ds2["pp1d"].argmax(dim="time")

# Select corresponding values
swh_max = ds2["swh"].isel(time=swh_argmax)
pp1d_max = ds2["pp1d"].isel(time=pp1d_argmax)

max_time_swh = ds2["time"].isel(time=swh_argmax)
max_time_pp1d = ds2["time"].isel(time=pp1d_argmax)

swh_max = swh_max.where(mask)
pp1d_max = pp1d_max.where(mask)
max_time_swh = max_time_swh.where(mask)
max_time_pp1d = max_time_pp1d.where(mask)

extreme = xr.Dataset(
    {
        "swh_max": swh_max,
        "pp1d_max": pp1d_max,
        "max_time_pp1d": max_time_pp1d,
    },
    coords={
        "lat": ds["lat"],
        "lon": ds["lon"],
        "max_time_swh": max_time_swh,
        #'max_time_pp1d': max_time_pp1d,
    },
)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
extreme["swh_max"].plot(cmap="viridis", vmin=0, cbar_kwargs={"label": "Max SWH (m)"})
plt.title("Maximum Significant Wave Height (1995–2014)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# %%
plt.figure(figsize=(12, 6))
extreme["pp1d_max"].plot(cmap="plasma", cbar_kwargs={"label": "PP1D at Max SWH (s)"})
plt.title("PP1D at Maximum SWH (1995–2014)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# %%
# Define gravitational acceleration
g = 9.81  # m/s^2

# Compute offshore wave length max
owl_max = (g * extreme["pp1d_max"] ** 2) / (2 * np.pi)

# Add to the dataset
extreme["owl_max"] = owl_max
extreme["owl_max"].attrs["units"] = "m"
extreme["owl_max"].attrs["long_name"] = "Offshore Wave Length Maximum"

# %%
plt.figure(figsize=(12, 6))
extreme["owl_max"].plot(cmap="cividis", cbar_kwargs={"label": "OWL Max (m)"})
plt.title("Offshore Wave Length Maximum (1995–2014)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# %%
output_path = Path("../data/processed/ERA5_processed.nc")
extreme.to_netcdf(output_path)

# %%
