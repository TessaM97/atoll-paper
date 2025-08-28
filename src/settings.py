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
# Absolute path from environment variable
# DATA_DIR = Path(os.environ["PAPER_DATA"]).expanduser().resolve()

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
# Additionnal
FIG_DIR = (
    "/Users/tessamoller/Documents/atoll-slr-paper-manuscript/Figures/Suppl_Figures"
)


# %%
