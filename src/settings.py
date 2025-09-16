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
from pathlib import Path

# %%
# Dynamically resolve project root (2 levels up from /src/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory (can be overridden by PAPER_DATA environment variable)
DATA_DIR = (
    Path(
        os.environ.get(
            "PAPER_DATA", "/Users/tessamoller/Documents/atoll-slr-paper-data_clean"
        )
    )
    .expanduser()
    .resolve()
)

# Common subfolders relative to DATA_DIR
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# Shapefile path (kept relative to project root for portability)
SHAPEFILE_PATH = PROJECT_ROOT / "data" / "Shapefiles" / "Atoll_transects_240725.shp"

# Figures directory (can also be made configurable with an env var if needed)
FIG_DIR = Path(
    os.environ.get(
        "FIG_DIR",
        PROJECT_ROOT.parent
        / "atoll-slr-paper-manuscript"
        / "Figures"
        / "Suppl_Figures",
    )
).resolve()
