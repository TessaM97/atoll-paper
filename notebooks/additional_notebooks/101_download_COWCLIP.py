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
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# %%
# Add project root (two levels up from current notebook folder)
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# project_root = Path().resolve().parents[1]
# sys.path.append(str(project_root))
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


try:
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path.cwd()  # fallback for Jupyter / interactive use

DATA_DIR = BASE_DIR / "data" / "raw" / "COWCLIP"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_catalog(url):
    """Fetch and parse an XML catalog from a THREDDS URL."""
    with urllib.request.urlopen(url) as response:
        return ET.parse(response)


def get_dataset_urls(catalog_url):
    """Recursively find all .nc dataset URLs in a THREDDS catalog."""
    tree = fetch_catalog(catalog_url)
    root = tree.getroot()
    urls = []

    # Follow sub-catalog references
    for catalog_ref in root.findall(".//x:catalogRef", NAMESPACE):
        href = catalog_ref.attrib.get("xlink:href")
        if href:
            next_url = urllib.request.urljoin(catalog_url, href)
            urls.extend(get_dataset_urls(next_url))  # recursion

    # Collect actual .nc dataset paths
    for dataset in root.findall(".//x:dataset", NAMESPACE):
        path = dataset.attrib.get("urlPath", "")
        if path.endswith(".nc"):
            urls.append(path)

    return urls


def download_file(url, dest, max_retries=3, chunk_size=8192):
    """
    Download a file with retries and chunked streaming.

    Parameters
    ----------
    url : str
        File URL to download.
    dest : Path
        Destination file path.
    max_retries : int
        Number of times to retry failed downloads.
    chunk_size : int
        Bytes per chunk when streaming.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    start_time = time.time()
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                percent = (downloaded / total) * 100
                                elapsed = time.time() - start_time
                                speed = downloaded / (elapsed + 1e-6) / 1024 / 1024
                                print(
                                    f"\r  ⏳ {percent:.1f}% ({speed:.2f} MB/s)", end=""
                                )

                print(f"\n✅ Downloaded: {dest.name} ({downloaded / 1e6:.1f} MB)")
                return

        except Exception as e:
            print(f"\n⚠️ Attempt {attempt} failed for {dest.name}: {e}")
            if attempt < max_retries:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Failed to download after {max_retries} attempts: {url}")


def download_datasets(variable, catalog_url):
    """Download all .nc datasets for a given variable from the catalog."""
    print(f"🔍 Scanning catalog for {variable}...")
    urls = get_dataset_urls(catalog_url)
    print(f"✅ Found {len(urls)} .nc files for {variable}.\n")

    save_dir = DATA_DIR / variable
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(urls, 1):
        filename = os.path.basename(path)
        url = BASE_SERVER + path
        dest = save_dir / filename
        print(f"[{variable} {i}/{len(urls)}] Downloading {filename}")
        download_file(url, dest)


def main():
    """Main entry point — download all COWCLIP2 datasets."""
    for variable, catalog_url in BASE_CATALOGS.items():
        download_datasets(variable, catalog_url)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()


# %%
