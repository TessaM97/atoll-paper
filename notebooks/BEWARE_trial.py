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

# %%

# Load the .nc file

file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE2data.nc"
dataset = nc.Dataset(file_path)

# Extract necessary variables
cross_shore_positions = dataset.variables["CrossShorePosition"][:]
rrp_elevations = dataset.variables["RRPElevation"][
    :, :
]  # Shape: (nCrossShorePoints, nProfiles)

# Threshold for reef presence
threshold = -30.0  # Example: mean sea level

# Handle NaNs in the elevation data by replacing them with a fill value (-30.0 is the _FillValue here)
fill_value = -30.0
rrp_elevations = np.nan_to_num(rrp_elevations, nan=fill_value)

# Initialize an array to store reef widths
n_profiles = rrp_elevations.shape[1]
reef_widths = np.zeros(n_profiles)

# Loop through each profile to calculate reef width
for i in range(n_profiles):
    # Get elevation for the current profile
    elevation = rrp_elevations[:, i]

    # Find indices where elevation exceeds the threshold
    above_threshold_indices = np.where(elevation > threshold)[0]

    if above_threshold_indices.size > 0:
        # Calculate reef width as the distance between the first and last points above the threshold
        reef_width = (
            cross_shore_positions[above_threshold_indices[-1]]
            - cross_shore_positions[above_threshold_indices[0]]
        )
    else:
        # If no points exceed the threshold, set reef width to 0
        reef_width = 0.0

    # Store the result
    reef_widths[i] = reef_width / 2  # SOMEHOW need this correcting factor

# Output reef widths
for i, width in enumerate(reef_widths):
    print(f"Profile {i + 1}: Reef Width = {width:.2f} m")

# Close the dataset
dataset.close()


# %%
# Load the .nc file
file_path = "/Users/tessamoeller/Documents/atoll_paper/data/BEWARE2data.nc"
dataset = nc.Dataset(file_path)

# Extract necessary variables
cross_shore_positions = dataset.variables["CrossShorePosition"][:]
rrp_elevations = dataset.variables["RRPElevation"][
    :, :
]  # Shape: (nCrossShorePoints, nProfiles)
hs = dataset.variables["Hs"][:]  # Offshore wave height
tp = dataset.variables["Tp"][:]  # Offshore wave period
wl = dataset.variables["WL"][:]  # Offshore still water level

# Handle NaNs in the elevation data
fill_value = -30.0
rrp_elevations = np.nan_to_num(rrp_elevations, nan=fill_value)

# Handle NaNs in the elevation data by replacing them with a fill value (-30.0 is the _FillValue here)
fill_value = -30.0
rrp_elevations = np.nan_to_num(rrp_elevations, nan=fill_value)


# Calculate reef widths for each profile
n_profiles = rrp_elevations.shape[1]
reef_widths = np.zeros(n_profiles)

threshold = -30  # Mean sea level threshold

for i in range(n_profiles):
    # Get elevation for the current profile
    elevation = rrp_elevations[:, i]

    # Replace fill values with NaN for processing
    elevation[elevation == fill_value] = np.nan

    # Find indices where elevation exceeds the threshold
    above_threshold_indices = np.where(elevation > threshold)[0]

    if above_threshold_indices.size > 0:
        # Calculate reef width
        reef_width = (
            cross_shore_positions[above_threshold_indices[-1]]
            - cross_shore_positions[above_threshold_indices[0]]
        )
    else:
        reef_width = 0.0

    reef_widths[i] = reef_width

# Filter profiles based on criteria
criteria_indices = np.where(
    (reef_widths > 1000) & (hs == 10) & (tp == 8) & (wl == 100)
)[0]

# Output the indices and details of profiles meeting the criteria
if criteria_indices.size > 0:
    print(
        "Profiles meeting the criteria (reef_width > 1000, Hs = 10, Tp = 8, WL = 100):"
    )
    for idx in criteria_indices:
        print(f"Profile {idx + 1}: Reef Width = {reef_widths[idx]:.2f} m")
else:
    print("No profiles meet the criteria.")

# Close the dataset
dataset.close()


# %%

dataset = nc.Dataset(file_path)

# Extract necessary variables
cross_shore_positions = dataset.variables["CrossShorePosition"][:]
rrp_elevations = dataset.variables["RRPElevation"][
    :, :
]  # Shape: (nCrossShorePoints, nProfiles)
hs = dataset.variables["Hs"][:]  # Offshore wave height (shape: 440)
tp = dataset.variables["Tp"][:]  # Offshore wave period (shape: 440)
wl = dataset.variables["WL"][:]  # Offshore still water level (shape: 440)
r2data = dataset.variables["R2data"][:, :, :]  # Shape: (5, 440, 195)


# Handle NaNs in the elevation data
fill_value = -30.0
rrp_elevations = np.nan_to_num(rrp_elevations, nan=fill_value)

# Calculate reef widths for each profile
n_profiles = rrp_elevations.shape[1]
reef_widths = np.zeros(n_profiles)

threshold = -30  # Mean sea level threshold

for i in range(n_profiles):
    # Get elevation for the current profile
    elevation = rrp_elevations[:, i]

    # Replace fill values with NaN for processing
    elevation[elevation == fill_value] = np.nan

    # Find indices where elevation exceeds the threshold
    above_threshold_indices = np.where(elevation > threshold)[0]

    if above_threshold_indices.size > 0:
        # Calculate reef width
        reef_width = (
            cross_shore_positions[above_threshold_indices[-1]]
            - cross_shore_positions[above_threshold_indices[0]]
        )
    else:
        reef_width = 0.0

    reef_widths[i] = reef_width

# Filter combinations of profiles and conditions
matching_combinations = []

for condition_idx in range(len(hs)):  # Loop through all forcing conditions
    if hs[condition_idx] == 6 and tp[condition_idx] == 10 and wl[condition_idx] == 3:
        # Check profiles where reef_width > 1000 for the matching condition
        for profile_idx in range(n_profiles):
            if reef_widths[profile_idx] > 1000:
                matching_combinations.append((profile_idx + 1, condition_idx + 1))

# Output the results
if matching_combinations:
    print(
        "Profiles and conditions meeting the criteria (reef_width > 1000, Hs = 6, Tp = 10, WL = 3):"
    )
    for profile_idx, condition_idx in matching_combinations:
        # Extract R2data for the given profile and condition
        r2_values = r2data[
            :, condition_idx - 1, profile_idx - 1
        ]  # Subtract 1 to convert to zero-based indexing
        r2_values = r2_values[r2_values != -999.0]  # Exclude fill values (-999.0)
        print(f"Profile {profile_idx}, Condition {condition_idx}: R2data = {r2_values}")
else:
    print("No profiles and conditions meet the criteria.")
# Close the dataset
dataset.close()


# %%
import netCDF4 as nc
import numpy as np

# Load the dataset
file_path = " /Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"  # Update this path
dataset = nc.Dataset(file_path)

# Extract variables
W_reef = dataset.variables["W_reef"][:]  # 1D array
beta_foreReef = dataset.variables["beta_ForeReef"][:]  # 1D array
H0 = dataset.variables["H0"][:]  # 1D array
eta0 = dataset.variables["eta0"][:]  # 1D array
R2pIndex = dataset.variables["R2pIndex"][:]  # 1D array

# Define your criteria
W_threshold = 1
beta_target = 0.05
H0_target = 1
eta0_target = 1

# Create a boolean mask
mask = (
    (W_reef > W_threshold)
    & (np.isclose(beta_foreReef, beta_target))
    & (np.isclose(H0, H0_target))
    & (np.isclose(eta0, eta0_target))
)

# Apply the mask
matching_indices = np.where(mask)[0]

# Display results
if matching_indices.size > 0:
    print("Matching R2pIndex values for given criteria:")
    for idx in matching_indices:
        print(f"Index {idx}: R2pIndex = {R2pIndex[idx]}")
else:
    print("No matching entries found.")

# Close the dataset
dataset.close()


# %%
# Load the dataset
file_path = " /Users/tessamoeller/Documents/atoll_paper/data/BEWARE_Database.nc"  # Update this path
dataset = nc.Dataset(file_path)

# Extract variables
W_reef = dataset.variables["W_reef"][:]
beta_foreReef = dataset.variables["beta_ForeReef"][:]
H0 = dataset.variables["H0"][:]
eta0 = dataset.variables["eta0"][:]
R2pIndex = dataset.variables["R2pIndex"][:]
Cf = dataset.variables["Cf"][:]
beta_beach = dataset.variables["beta_Beach"][:]

# Define filtering criteria
W_threshold = 1
beta_target = 0.05
H0_target = 1
eta0_target = 1

# Create boolean mask
mask = (
    (W_reef > W_threshold)
    & (np.isclose(beta_foreReef, beta_target))
    & (np.isclose(H0, H0_target))
    & (np.isclose(eta0, eta0_target))
)

# Get matching indices
matching_indices = np.where(mask)[0]

# Display results
if matching_indices.size > 0:
    print("Matching R2pIndex values and corresponding Cf, beta_Beach:")
    for idx in matching_indices:
        print(
            f"Index {idx}: R2pIndex = {R2pIndex[idx]}, Cf = {Cf[idx]}, beta_Beach = {beta_beach[idx]}"
        )
else:
    print("No matching entries found.")

# Close the dataset
dataset.close()


# %%
