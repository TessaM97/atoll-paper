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
import pandas as pd


# %%
df = pd.read_parquet("../data/processed/Atoll_BEWARE_inputs.parquet")
df.head()

# %%
# Count unique transect_i values
num_transects = df['transect_i'].nunique()
print(f"Number of unique transects: {num_transects}")

# %%
# Count transects per Atoll_FID
transects_per_atoll = df.groupby('Atoll_FID')['transect_i'].nunique()

# Calculate the average
average_transects_per_atoll = transects_per_atoll.mean()
print(f"Average number of transects per Atoll_FID: {average_transects_per_atoll:.2f}")


# %%
# Count transects per Atoll_FID
transects_per_atoll = df.groupby('FID_GADM')['transect_i'].nunique()

# Calculate the average
average_transects_per_atoll = transects_per_atoll.mean()
print(f"Average number of transects per FID_GADM: {average_transects_per_atoll:.2f}")

# %%
key_cols = ["eta0"]

df_combined = pd.concat(
    [df, df_extended],
    ignore_index=True
).drop_duplicates(subset=key_cols, ignore_index=True)
df_combined

# %%
import pandas as pd

# %%
