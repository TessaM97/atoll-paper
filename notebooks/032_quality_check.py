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

# %% [markdown]
# ### Quality check of results

# %%
import multiprocessing
import os

import numpy as np
import pandas as pd
from tqdm import tqdm



# %%
output_dir = "/Users/tessamoller/Documents/atoll-slr-paper-data/data/processed/"


# Make sure the file path is correct and accessible
file_path = output_dir + "Atoll_BEWARE_processed_outputs_2020-2150.parquet"

# Read the parquet file
df = pd.read_parquet(file_path, engine= "pyarrow")

# Display the DataFrame
df

# %%
file Atoll_BEWARE_processed_outputs_2020-2150.parquet

# %%
import os

print(os.path.exists(file_path))        # Should be True
print(os.path.isfile(file_path))        # Should also be True
print(os.path.getsize(file_path))       # Should be > 0


# %%
with open(file_path, 'rb') as f:
    start = f.read(4)
    f.seek(-4, os.SEEK_END)
    end = f.read(4)

print("Start bytes:", start)
print("End bytes:", end)


# %%
import pyarrow.parquet as pq

pf = pq.ParquetFile(file_path)
print(pf.metadata)

# %%
with open(file_path, 'rb') as f:
    print(f.read(500))  # Print first 500 bytes

# %%
df = pd.read_parquet(file_path, engine="fastparquet")


# %%
import pyarrow.parquet as pq

try:
    pqfile = pq.ParquetFile(file_path)
    num_row_groups = pqfile.num_row_groups
    print(f"Found {num_row_groups} row groups")
    
    # Try reading one row group at a time
    dfs = []
    for i in range(num_row_groups):
        print(f"Reading row group {i}")
        try:
            table = pqfile.read_row_group(i)
            dfs.append(table.to_pandas())
        except Exception as e:
            print(f"Row group {i} failed: {e}")

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print("Successfully recovered partial data.")
    else:
        df = None
        print("No usable data recovered.")

except Exception as e:
    print("Failed to read anything:", e)


# %%
