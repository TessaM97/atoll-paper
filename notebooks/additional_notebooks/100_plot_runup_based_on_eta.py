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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, RAW_DIR

print("Using data directory:", DATA_DIR)

# %%
# Load BEWARE dataset
file_path = RAW_DIR / "BEWARE_Database.nc"
dataset = nc.Dataset(file_path)

# Extract variables
eta0 = dataset.variables["eta0"][:]
R2pIndex = dataset.variables["R2pIndex"][:]

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(eta0, R2pIndex, alpha=0.3, s=10, color="teal")
plt.xlabel("η₀ (m)", fontsize=12)
plt.ylabel("R2pIndex (m)", fontsize=12)
plt.title("R2pIndex as a function of η₀ from BEWARE", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Close dataset
dataset.close()
