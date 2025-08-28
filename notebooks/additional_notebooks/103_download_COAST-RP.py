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

import requests

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from src.settings import DATA_DIR, RAW_DIR

# Paths
directory_path = RAW_DIR / "external/COAST-RP"
os.makedirs(directory_path, exist_ok=True)
print("Using data directory:", DATA_DIR)

# %% [markdown]
# ### Automatic Download of COAST-RP dataset
# This script automatically downloads the COAST-RP (Coastal Storm Tide Return Periods) dataset from:     https://data.4tu.nl/articles/dataset/COAST-RP_A_global_COastal_dAtaset_of_Storm_Tide_Return_Periods/13392314.
#
# #### Notes:
# Please review the dataset’s license and usage terms before use.
# Ensure to cite COAST-RP appropriately in any publications or analyses.

# %%
# Define the URLs for the dataset and README file
dataset_url = "https://doi.org/10.4121/13392314.v2"
readme_url = "https://data.4tu.nl/articles/dataset/COAST-RP_A_global_COastal_dAtaset_of_Storm_Tide_Return_Periods/13392314"

os.makedirs(directory_path, exist_ok=True)

# Define the local file paths
local_nc_file = os.path.join(directory_path, "COAST-RP.nc")
local_readme_file = os.path.join(directory_path, "README.txt")

# Download the NetCDF dataset
response = requests.get(dataset_url, stream=True)
response.raise_for_status()  # Check for request errors
with open(local_nc_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)
print(f"Dataset downloaded successfully to {local_nc_file}")

# Download the README.txt file
response = requests.get(readme_url)
response.raise_for_status()  # Check for request errors
with open(local_readme_file, "w") as file:
    file.write(response.text)
print(f"README.txt downloaded successfully to {local_readme_file}")
