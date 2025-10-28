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
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, INTERIM_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)
# Paths
directory_path = RAW_DIR / "external/COWCLIP/Hs"
output_dir = INTERIM_DIR / "external/COWCLIP/Hs"


# %% [markdown]
# ## Preprocess and clean Hs COWCLIP outputs, calculate ensemble average
# Note that you need to have first downloaded the data.
# To do so, you can execute `101_download_COWCLIP.ipnyb`
# under `/additional_notebooks`.

# %% [markdown]
# ### Check resolution and coordinates of input files


# %%
def find_coord_name(coords, possible_names):
    """Find the coordinate name from a list of possible names."""
    for name in possible_names:
        if name in coords:
            return name
    return None


def compute_resolution(coord_values):
    """Compute resolution as the mean of the absolute differences."""
    coord_values = np.asarray(coord_values, dtype=np.float64)

    if len(coord_values) < 2:
        return None

    diffs = np.diff(coord_values)

    if len(diffs) == 0 or not np.isfinite(diffs).all():
        return None

    return float(np.mean(np.abs(diffs)))


def inspect_netcdf_files(path, recursive=False):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".nc"):
                full_path = os.path.join(root, filename)
                try:
                    ds = xr.open_dataset(full_path, decode_times=False)
                    coords = ds.coords

                    lat_name = find_coord_name(
                        coords, ["lat", "latitude", "Latitude", "nav_lat", "y"]
                    )
                    lon_name = find_coord_name(
                        coords, ["lon", "longitude", "Longitude", "nav_lon", "x"]
                    )

                    print(f"\n📄 File: {filename}")

                    if lat_name and lon_name:
                        lat_vals = ds[lat_name].values
                        lon_vals = ds[lon_name].values

                        # Handle 2D grids
                        if lat_vals.ndim > 1:
                            lat_vals = lat_vals[:, 0]
                        if lon_vals.ndim > 1:
                            lon_vals = lon_vals[0, :]

                        lat_res = compute_resolution(lat_vals)
                        lon_res = compute_resolution(lon_vals)

                        print(
                            f"  🧭 Latitude ({lat_name}): "
                            f"min={lat_vals.min():.2f}, max={lat_vals.max():.2f}, "
                            f"res={lat_res:.4f}°"
                        )
                        print(
                            f"  🧭 Longitude ({lon_name}): "
                            f"min={lon_vals.min():.2f}, max={lon_vals.max():.2f}, "
                            f"res={lon_res:.4f}°"
                        )
                    else:
                        print("  ⚠️  Could not find lat/lon coordinates.")

                except Exception as e:
                    print(f"  ❌ Error opening {filename}: {e}")

        if not recursive:
            break


# 🔧 Example usage
inspect_netcdf_files(directory_path, recursive=True)


# %%
ds = xr.open_dataset(
    directory_path / "hs_JRC-ERAI_annual_1980-2014.nc",
    decode_times=False,
)
print(ds["longitude"].values)
print(ds["longitude"].shape)
print(ds["longitude"].attrs)

# %%
## Fix broken longitudes

n_lon = ds.dims["longitude"]  # or use len(ds['longitude'])
new_lons = np.linspace(0, 360 - 360 / n_lon, n_lon)

ds = ds.assign_coords(longitude=("longitude", new_lons))

print(ds["longitude"].values)
print(ds["longitude"].shape)
print(ds["longitude"].attrs)

ds.to_netcdf(directory_path / "hs_JRC-ERAI_annual_1980-2014_fixed.nc")


# %%
def regrid_file(input_path, output_path, lat_name="lat", lon_name="lon"):
    try:
        ds = xr.open_dataset(input_path, decode_times=False)

        # Guess coordinate names
        all_coords = list(ds.coords)
        lat_guess = [
            name
            for name in all_coords
            if name.lower() in ["lat", "latitude", "nav_lat", "y"]
        ]
        lon_guess = [
            name
            for name in all_coords
            if name.lower() in ["lon", "longitude", "nav_lon", "x"]
        ]

        if lat_guess:
            lat_name = lat_guess[0]
        if lon_guess:
            lon_name = lon_guess[0]

        # Convert longitudes from -180..180 to 0..360 if needed
        lon_vals = ds[lon_name].values
        if np.nanmin(lon_vals) < 0:
            ds[lon_name] = (ds[lon_name] + 180) % 360
            ds = ds.sortby(lon_name)  # sort to ensure interpolation is monotonic

        # Define the new uniform grid
        new_lat = np.arange(-89.75, 90, 0.5)
        new_lon = np.arange(0.25, 360, 0.5)

        ds_interp = ds.interp(
            coords={lat_name: new_lat, lon_name: new_lon}, method="linear"
        )

        ds_interp.to_netcdf(output_path)
        print(
            f"✅ Regridded: {os.path.basename(input_path)} → {os.path.basename(output_path)}"
        )

    except Exception as e:
        print(f"❌ Failed: {os.path.basename(input_path)} — {e}")


def regrid_all_in_directory(
    directory, recursive=False, output_folder="regridded", output_suffix="_regridded.nc"
):
    os.makedirs(output_dir, exist_ok=True)

    exclude_file = "hs_JRC-ERAI_annual_1980-2014.nc"
    include_file = "hs_JRC-ERAI_annual_1980-2014_fixed.nc"

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".nc"):
                # Skip the excluded file
                if fname == exclude_file:
                    continue

                # Only process the fixed version if it exists
                if fname == include_file or fname != exclude_file:
                    input_path = os.path.join(root, fname)
                    output_name = fname.replace(".nc", output_suffix)
                    output_path = os.path.join(output_dir, output_name)
                    regrid_file(input_path, output_path)

        if not recursive:
            break


# Set your input directory here
regrid_all_in_directory(directory_path)


# %%
def plot_regridded_file(filepath, var_name=None):
    ds = xr.open_dataset(filepath, decode_times=False)

    if var_name is None:
        var_name = list(ds.data_vars)[0]

    da = ds[var_name]

    # Guess lat/lon coordinate names:
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude"]

    lat_name = next((name for name in lat_candidates if name in ds.coords), None)
    lon_name = next((name for name in lon_candidates if name in ds.coords), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Latitude or Longitude coordinate not found in dataset")

    # If time dimension exists, select first time slice
    if "time" in da.dims:
        da = da.isel(time=0)

    # Mask values below -100 by setting them to NaN
    da = da.where(da >= 0)

    da.plot(x=lon_name, y=lat_name, cmap="viridis")
    plt.title(f"{var_name} from {filepath}")
    plt.show()


# %%
plot_regridded_file(output_dir / "hs_CSIRO-G1D_annual_1980-2010_regridded.nc")


# %%
def check_regridding(
    input_dir, regridded_dir, lat_name="latitude", lon_name="longitude"
):
    # List all NetCDF files in input_dir
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".nc")]

    for filename in input_files:
        input_file_path = os.path.join(input_dir, filename)
        regrid_file_path = os.path.join(
            regridded_dir, filename.replace(".nc", "_regridded.nc")
        )

        if not os.path.exists(regrid_file_path):
            print(f"Regridded file not found for {filename}, skipping.")
            continue

        ds_orig = xr.open_dataset(input_file_path, decode_times=False)
        ds_regrid = xr.open_dataset(regrid_file_path, decode_times=False)

        # Find variable to compare (assuming only one data variable)
        var_orig = list(ds_orig.data_vars)[0]
        var_regrid = list(ds_regrid.data_vars)[0]

        # Print some info on coordinate coverage and resolution
        print(f"\nChecking file: {filename}")

        print(
            f"Original {lat_name} range: {ds_orig[lat_name].min().values} to {ds_orig[lat_name].max().values}"
        )
        print(
            f"Regridded {lat_name} range: {ds_regrid[lat_name].min().values} to {ds_regrid[lat_name].max().values}"
        )

        print(
            f"Original {lon_name} range: {ds_orig[lon_name].min().values} to {ds_orig[lon_name].max().values}"
        )
        print(
            f"Regridded {lon_name} range: {ds_regrid[lon_name].min().values} to {ds_regrid[lon_name].max().values}"
        )

        # Check resolution by computing mean difference between coords
        orig_lat_res = np.diff(ds_orig[lat_name]).mean()
        regrid_lat_res = np.diff(ds_regrid[lat_name]).mean()

        orig_lon_res = np.diff(ds_orig[lon_name]).mean()
        regrid_lon_res = np.diff(ds_regrid[lon_name]).mean()

        print(f"Original lat resolution: {orig_lat_res}")
        print(f"Regridded lat resolution: {regrid_lat_res}")
        print(f"Original lon resolution: {orig_lon_res}")
        print(f"Regridded lon resolution: {regrid_lon_res}")

        # Optional: check data shape differences
        print(f"Original data shape: {ds_orig[var_orig].shape}")
        print(f"Regridded data shape: {ds_regrid[var_regrid].shape}")

        # Close datasets
        ds_orig.close()
        ds_regrid.close()


# %%
check_regridding(
    directory_path,
    output_dir,
)


# %%
def plot_regridded_year(
    regridded_dir,
    year=1990,
    var_to_plot="hs_avg",
    lat_name="latitude",
    lon_name="longitude",
):
    files = [
        f for f in os.listdir(regridded_dir) if f.endswith(".nc") and "_regridded" in f
    ]

    n_files = len(files)
    ncols = 3
    nrows = (n_files + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
    axs = axs.flatten()

    for i, fname in enumerate(files):
        path = os.path.join(regridded_dir, fname)
        try:
            ds = xr.open_dataset(path, decode_times=False)

            # Check for desired variable
            if var_to_plot not in ds.data_vars:
                print(f"{var_to_plot} not found in {fname}, skipping.")
                axs[i].axis("off")
                continue

            if "time" not in ds.coords:
                print(f"No 'time' coordinate in {fname}, skipping.")
                axs[i].axis("off")
                continue

            # Extract year array manually
            time_units = ds.time.attrs.get("units", "")
            if "since 1980" in time_units:
                years = 1980 + ds["time"].values.astype(int)
            else:
                raise ValueError(f"Unexpected time units in {fname}: {time_units}")

            # Find index for requested year
            if year not in years:
                print(f"Year {year} not found in {fname}, skipping.")
                axs[i].axis("off")
                continue

            year_idx = list(years).index(year)
            data = (
                ds[var_to_plot]
                .isel(time=year_idx)
                .where(ds[var_to_plot].isel(time=year_idx) > 0)
            )

            # Plot
            im = data.plot(
                ax=axs[i], cmap="viridis", cbar_kwargs={"label": var_to_plot}
            )
            axs[i].set_title(fname.replace("_regridded.nc", ""))
            axs[i].set_xlabel("Longitude")
            axs[i].set_ylabel("Latitude")

        except Exception as e:
            print(f"Failed to plot {fname}: {e}")
            axs[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
plot_regridded_year(output_dir, year=1990, var_to_plot="hs_avg")


# %%
def check_time_extents(regridded_dir, time_var="time"):
    time_coverage = []

    for fname in sorted(os.listdir(regridded_dir)):
        if not fname.endswith(".nc"):
            continue

        fpath = os.path.join(regridded_dir, fname)

        try:
            ds = xr.open_dataset(fpath, decode_times=False)
            if time_var not in ds:
                print(f"Skipping {fname}: no '{time_var}' variable found.")
                continue

            # Extract raw time values
            raw_time = ds[time_var].values
            units = ds[time_var].attrs["units"]

            # Parse origin from units like 'years since 1980-01-01'
            if "years since" in units:
                base_year = int(units.split("since")[1].strip().split("-")[0])
                years = base_year + raw_time
                start = int(years[0])
                end = int(years[-1])
            else:
                raise ValueError(f"Unrecognized time units: {units}")

            time_coverage.append((fname, start, end))
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    print("\nTime extent of regridded files:")
    for fname, start, end in time_coverage:
        print(f"{fname}: {start} to {end}")

    return time_coverage


time_coverage = check_time_extents(output_dir)

# %%
# Variables to average
ensemble_mean.close()
target_vars = ["hs_avg", "hs_p10", "hs_p50", "hs_p90", "hs_p95", "hs_p99", "hs_max"]


def get_dataset_means(fname, time_var="time", variables=target_vars):
    """Extracts and averages selected variables over correct time range, masking values > 100."""
    ds = xr.open_dataset(fname, decode_times=False)

    # Parse base year from units like 'years since 1980-01-01'
    units = ds[time_var].attrs.get("units", "")
    base_year = int(units.split("since")[1].strip().split("-")[0])
    years = base_year + ds[time_var].values

    # Time mask based on model
    if "CSIRO-G1D" in fname:
        year_mask = (years >= 1995) & (years <= 2009)
    else:
        year_mask = (years >= 1995) & (years <= 2014)

    if not np.any(year_mask):
        raise ValueError(f"No valid years in {fname} for requested period")

    # Select, mask, and average each target variable
    averaged = []
    for var in variables:
        if var in ds:
            data = ds[var].isel({time_var: year_mask})
            data = data.where(data < 100)  # ⬅️ Mask values >= 100
            averaged.append(data.mean(dim=time_var))
        else:
            raise ValueError(
                f"Variable '{var}' not found in {fname}. Available: {list(ds.data_vars)}"
            )

    return xr.merge(averaged)


def compute_ensemble_average(folder, variables=target_vars):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".nc")])
    model_means = []

    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            model_mean = get_dataset_means(fpath, variables=variables)
            model_means.append(model_mean)
            print(f"✅ Included: {fname}")
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    if not model_means:
        raise RuntimeError("No datasets loaded successfully.")

    # Combine and average across the new 'ensemble' dimension
    ensemble_mean = xr.concat(model_means, dim="ensemble").mean(dim="ensemble")
    return ensemble_mean


def interpolate_percentiles(ds, base_percentiles=None, target_percentiles=None):
    if base_percentiles is None:
        base_percentiles = [10, 50, 90, 95, 99]
    if target_percentiles is None:
        target_percentiles = [17, 83]

    # Step 1: Stack base percentile variables into one DataArray
    try:
        data_vars = [ds[f"hs_p{p}"] for p in base_percentiles]
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Missing expected variable: {missing} in dataset")

    hs_stacked = xr.concat(
        data_vars,
        dim=xr.DataArray(base_percentiles, dims="percentile", name="percentile"),
    )

    # Step 2: Interpolate and add each target percentile to the dataset
    for p in target_percentiles:
        interpolated = hs_stacked.interp(percentile=p)
        ds[f"hs_p{p}"] = interpolated

    # Step 3: Remove scalar percentile coordinate if present
    ds = ds.drop_vars("percentile", errors="ignore")

    return ds


# === Run it ===
output_path = INTERIM_DIR / "external/COWCLIP/COWCLIP_ensemble_mean_Hs_1995_2014.nc"

ensemble_mean = compute_ensemble_average(output_dir)
ensemble_mean_interpolated = interpolate_percentiles(ensemble_mean)


# Delete the output file if it exists
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"🗑️ Removed existing file at {output_path}")

# Save the result
ensemble_mean_interpolated.to_netcdf(output_path)
print(f"✅ Ensemble mean saved to '{output_path}'")


# %%
# Load the ensemble mean file
ds = xr.open_dataset(output_path)

# Mask invalid values (<= -100)
masked_ds = ds.where(ds > 0)

# Set up plots
n_vars = len(masked_ds.data_vars)
ncols = 3
nrows = (n_vars + ncols - 1) // ncols

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
axs = axs.flatten()

for i, var in enumerate(masked_ds.data_vars):
    da = masked_ds[var]
    im = da.plot(ax=axs[i], cmap="viridis", cbar_kwargs={"label": var})
    axs[i].set_title(var)
    axs[i].set_xlabel("Longitude")
    axs[i].set_ylabel("Latitude")

# Hide unused axes if any
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.show()
