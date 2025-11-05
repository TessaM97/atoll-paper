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
import shutil
import sys
from pathlib import Path

import requests

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from src.settings import DATA_DIR, RAW_DIR

# Paths
directory_path = RAW_DIR / "external/AR6_regional_SLR_projections"
print("Using data directory:", DATA_DIR)


# %% [markdown]
# ### Automatic Download of AR6 SLR Projection Files
#
# This script automatically downloads the **AR6 Sea Level Rise (SLR) projection data** from Zenodo, extracts the relevant files, removes unnecessary files, and organizes them into the appropriate subfolders.
#
# #### Notes
# - The original data is provided in **zip archives** containing multiple files, including redundant or unnecessary ones. Temporary storage may be required during extraction, but non-relevant files are deleted automatically.
# - Only files matching the desired criteria (e.g., containing `"total"` and `"values"`) are kept.
# - Please review the **license and citation information and additional documentation** on the Zenodo record: [https://zenodo.org/records/5914710](https://zenodo.org/records/5914710).
#


# %%
rec_id = "5914710"
api_url = f"https://zenodo.org/api/records/{rec_id}"
r = requests.get(api_url)
r.raise_for_status()
data = r.json()

files = data.get("files", [])
print(f"Total files in record: {len(files)}\n")

for f in files:
    print(f["key"])


# %%
from pathlib import Path
from zipfile import ZipFile

import requests


def download_and_extract_selected_zip(
    zip_url, output_dir, keywords=("total", "values")
):
    """
    Download a zip, extract only .nc files matching keywords, delete zip and other .nc files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the zip
    zip_name = zip_url.split("/")[-1].split("?")[0]  # remove query params
    zip_path = output_dir / zip_name

    print(f"Downloading {zip_name} …")
    with requests.get(zip_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Step 2: Extract files
    print("Extracting matching .nc files …")
    with ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith(".nc") and all(
                kw.lower() in member.lower() for kw in keywords
            ):
                member_path = output_dir / member
                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as source, open(member_path, "wb") as target:
                    target.write(source.read())
                print(f"Extracted {member}")

    # Step 3: Delete the zip
    zip_path.unlink()
    print(f"Deleted zip file {zip_name}")

    # Step 4: Delete other .nc files that don't match the pattern
    for nc_file in output_dir.rglob("*.nc"):
        if not all(kw.lower() in nc_file.name.lower() for kw in keywords):
            nc_file.unlink()
            print(f"Deleted non-matching file {nc_file.relative_to(output_dir)}")

    print("Done. Only matching .nc files remain.")


# Example usage

# output_folder = RAW_DIR / "external/AR6_regional_SLR_projections"

zip_url = (
    "https://zenodo.org/record/5914710/files/ar6-regional-confidence.zip?download=1"
)

download_and_extract_selected_zip(zip_url, directory_path)


# %%
# Base folder containing the extracted files
base_dir = Path(
    "/Users/tessamoller/Documents/atoll-paper-data_clean/raw/external/AR6_regional_SLR_projections"
)
prefix_to_remove = "ar6-regional-confidence/regional"

# Pattern keywords (must contain these in filename)
keywords = ("total", "values")

# Step 1: Move matching .nc files and preserve subfolder structure after prefix
for nc_file in base_dir.rglob("*.nc"):
    if all(kw.lower() in nc_file.name.lower() for kw in keywords):
        try:
            # Get relative path after the prefix to remove
            relative_path = nc_file.relative_to(base_dir / prefix_to_remove)
        except ValueError:
            # File is not under the prefix, skip
            continue

        # Determine new destination while keeping subfolders
        destination = base_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(nc_file), destination)
        print(
            f"Moved: {nc_file.relative_to(base_dir)} -> {destination.relative_to(base_dir)}"
        )

# Step 2: Remove the entire ar6-regional-confidence folder
prefix_path = base_dir / "ar6-regional-confidence"
if prefix_path.exists():
    shutil.rmtree(prefix_path)
    print(f"Deleted folder and all subfolders: {prefix_path.relative_to(base_dir)}")

print("All matching files moved and ar6-regional-confidence folder removed.")
