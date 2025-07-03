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
import urllib.request
import xml.etree.ElementTree as ET

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

    save_dir = f"../data/COWCLIP/{variable}"
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


# %%
