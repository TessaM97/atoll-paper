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

# %%
import netCDF4 as nc
import numpy as np


def validate_extended_netcdf(file_path):
    print(f"Opening NetCDF file: {file_path}")
    ds = nc.Dataset(file_path)

    # 1. List available variables and dimensions
    print("\n📦 Variables in file:")
    for var in ds.variables:
        print(f"  - {var}")

    print("\n📏 Dimensions:")
    for dim in ds.dimensions:
        print(f"  - {dim} ({len(ds.dimensions[dim])})")

    # 2. Preview values from key variables
    preview_vars = ["eta0", "R2pIndex"]
    for var in preview_vars:
        print(f"\n🔍 Preview of '{var}':")
        data = ds.variables[var][:]
        print(f"  - Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
        print(f"  - Sample: {data[:10]}")

    # 3. Check for fill values (e.g., -9999)
    fill_val = -9999
    key_vars = ["H0", "H0L0", "Cf", "beta_Beach", "beta_ForeReef", "W_reef"]
    print("\n🚨 Checking for fill values (-9999):")
    for var in key_vars:
        data = ds.variables[var][:]
        missing_count = np.sum(data == fill_val)
        print(f"  - {var}: {missing_count} fill values")

    # 4. Plot R2pIndex vs eta0
    eta = ds.variables["eta0"][:]
    r2p = ds.variables["R2pIndex"][:]

    print("\n📈 Plotting R2pIndex vs η₀...")
    plt.figure(figsize=(8, 5))
    plt.scatter(eta, r2p, s=5, alpha=0.5, c="steelblue")
    plt.axvline(3.0, color="red", linestyle="--", label="Extrapolation Start (η₀ = 3)")
    plt.xlabel("η₀ (m)")
    plt.ylabel("R2pIndex (m)")
    plt.title("Validation: R2pIndex vs η₀")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. Check extrapolated range
    extended_mask = eta > 3
    print(f"\n📊 Extended points (η₀ > 3): {np.sum(extended_mask)}")
    print(
        f"  - R2pIndex in extended range: min={np.nanmin(r2p[extended_mask])}, max={np.nanmax(r2p[extended_mask])}"
    )

    ds.close()
    print("\n✅ Validation complete.")


# Example usage:
validate_extended_netcdf("../data/processed/BEWARE_Database_extended.nc")


# %%
def compare_netcdf_eta_le_3(original_path, extended_path, tolerance=1e-6):
    print("📁 Loading NetCDF files...")
    ds_orig = nc.Dataset(original_path)
    ds_ext = nc.Dataset(extended_path)

    # Variables to compare
    vars_to_check = [
        "H0",
        "H0L0",
        "eta0",
        "Cf",
        "beta_ForeReef",
        "beta_Beach",
        "W_reef",
        "R2pIndex",
    ]

    print("\n🔍 Filtering values where η₀ ≤ 3...")
    eta_orig = ds_orig.variables["eta0"][:]
    eta_ext = ds_ext.variables["eta0"][:]

    mask_orig = eta_orig <= 3
    mask_ext = eta_ext <= 3

    if np.sum(mask_orig) != np.sum(mask_ext):
        print(
            f"⚠️ Warning: Number of values with η₀ ≤ 3 differ between datasets! ({np.sum(mask_orig)} vs {np.sum(mask_ext)})"
        )

    # Create index-based mapping
    print("\n🔬 Comparing values...")
    for var in vars_to_check:
        if var not in ds_orig.variables or var not in ds_ext.variables:
            print(f"⚠️ Variable '{var}' missing in one of the datasets.")
            continue

        data_orig = ds_orig.variables[var][:]
        data_ext = ds_ext.variables[var][:]

        values_orig = data_orig[mask_orig]
        values_ext = data_ext[mask_ext]

        if len(values_orig) != len(values_ext):
            print(
                f"❌ {var}: Length mismatch ({len(values_orig)} vs {len(values_ext)})"
            )
            continue

        if np.allclose(values_orig, values_ext, atol=tolerance, equal_nan=True):
            print(f"✅ {var}: Match within tolerance (±{tolerance})")
        else:
            diff = np.abs(values_orig - values_ext)
            max_diff = np.nanmax(diff)
            mismatch_count = np.sum(diff > tolerance)
            print(f"❌ {var}: {mismatch_count} mismatches (max diff = {max_diff:.2e})")

    # Close files
    ds_orig.close()
    ds_ext.close()
    print("\n✅ Comparison complete.")


# Example usage:
compare_netcdf_eta_le_3(
    "../data/BEWARE_Database.nc", "../data/processed/BEWARE_Database_extended.nc"
)
