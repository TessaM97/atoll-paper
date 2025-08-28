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
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))
from src.settings import DATA_DIR, RAW_DIR

# Paths
directory_path = RAW_DIR / "external/COWCLIP"
print("Using data directory:", DATA_DIR)

# %% [markdown]
# ### Automatic Download of Raw COWCLIP Files
#
# This script automatically downloads the required **COWCLIP2 hindcast datasets** from the Australian Ocean Data Network (AODN) THREDDS server:
# [https://thredds.aodn.org.au/thredds/catalog/CSIRO/Climatology/COWCLIP2/hindcasts/Monthly/catalog.html](https://thredds.aodn.org.au/thredds/catalog/CSIRO/Climatology/COWCLIP2/hindcasts/Monthly/catalog.html)
#
# #### Notes
# - Only the relevant variables (e.g., **Tm** and **Ds**) are downloaded and organized locally.
# - Temporary storage may be needed during the download process, as the datasets include multiple files in nested directories.
# - Please review the **license and citation information** provided by CSIRO before using the datasets.
#

# %%
BASE_CATALOGS = {
    "Hs": "https://thredds.aodn.org.au/thredds/catalog/CSIRO/Climatology/COWCLIP2/hindcasts/Annual/Hs/catalog.xml",
    "Tm": "https://thredds.aodn.org.au/thredds/catalog/CSIRO/Climatology/COWCLIP2/hindcasts/Annual/Tm/catalog.xml",
}

BASE_SERVER = "https://thredds.aodn.org.au/thredds/fileServer/"
NAMESPACE = {"x": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


def fetch_catalog(url):
    with urllib.request.urlopen(url) as response:
        return ET.parse(response)


def get_dataset_urls(catalog_url):
    tree = fetch_catalog(catalog_url)
    root = tree.getroot()
    urls = []

    # Check for sub-catalog references
    for catalog_ref in root.findall(".//x:catalogRef", NAMESPACE):
        href = catalog_ref.attrib.get("xlink:href")
        if href:
            next_url = urllib.request.urljoin(catalog_url, href)
            urls.extend(get_dataset_urls(next_url))  # recurse

    # Look for actual .nc datasets
    for dataset in root.findall(".//x:dataset", NAMESPACE):
        path = dataset.attrib.get("urlPath", "")
        if path.endswith(".nc"):
            urls.append(path)

    return urls


def download_datasets(variable, catalog_url):
    print(f"🔍 Scanning catalog for {variable}...")
    urls = get_dataset_urls(catalog_url)
    print(f"✅ Found {len(urls)} .nc files for {variable}.")

    save_dir = directory_path / variable
    os.makedirs(save_dir, exist_ok=True)

    for i, path in enumerate(urls, 1):
        filename = os.path.basename(path)
        url = BASE_SERVER + path
        print(f"[{variable} {i}/{len(urls)}] Downloading {filename}")
        urllib.request.urlretrieve(url, os.path.join(save_dir, filename))


def main():
    for variable, catalog_url in BASE_CATALOGS.items():
        download_datasets(variable, catalog_url)


if __name__ == "__main__":
    main()
