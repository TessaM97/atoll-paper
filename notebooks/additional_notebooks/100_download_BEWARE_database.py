# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: atolls
#     language: python
#     name: atolls
# ---

# %%
import os
import sys
from pathlib import Path

import requests

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# project_root = Path().resolve().parents[1]
# sys.path.append(str(project_root))s
from src.settings import DATA_DIR, RAW_DIR

# Paths
directory_path = RAW_DIR
os.makedirs(directory_path, exist_ok=True)
print("Using data directory:", DATA_DIR)

# %% [markdown]
# ### Automatic Download of BEWARE Database
#
# This script automatically downloads the required **BEWARE Database** from the U.S. Geological Survey (USGS) ScienceBase platform:
# [https://www.sciencebase.gov/catalog/item/59c452c3e4b017cf313bea5c](https://www.sciencebase.gov/catalog/item/59c452c3e4b017cf313bea5c)
#
# #### Notes
# - Please review the **license and citation information** provided by USGS before using the dataset.
# - The dataset includes a **NetCDF file** named `BEWARE_Database.nc`.

# %%
url = "https://www.sciencebase.gov/catalog/file/get/59c452c3e4b017cf313bea5c?name=BEWARE_Database.nc"
output_path = os.path.join(RAW_DIR, "BEWARE_Database.nc")

response = requests.get(url)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Download completed successfully: {output_path}")
else:
    print(f"Failed to download file: {response.status_code}")

# %%
