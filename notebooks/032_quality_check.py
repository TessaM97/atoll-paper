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
import pyarrow.parquet as pq
import random

# %%
#output_dir = "/Users/tessamoller/Documents/atoll-slr-paper-data/data/processed/"
#file_path = output_dir + "Atoll_BEWARE_processed_outputs_2020-2150.parquet"
file_path = '/Users/tessamoller/Downloads/Atoll_BEWARE_processed_outputs_2020-2150-2.parquet'

# Read just the first 1000 rows using pyarrow
table = pq.read_table(file_path, use_threads=True)
df = table.slice(1000, 2000).to_pandas()

df


# %%
num_rows = table.num_rows
print(f"Number of rows: {num_rows}")

# %%
column_names = table.column_names

# Check for existence of 'scenario' and 'year' columns
if "scenario" in column_names and "year" in column_names:
    # Convert only the required columns to pandas Series for unique values
    df_subset = table.select(["scenario", "year", "transect_id", "confidence"]).to_pandas()

    scenarios = sorted(df_subset["scenario"].unique())
    years = sorted(df_subset["year"].unique())
    transect_ids = sorted(df_subset["transect_id"].unique())
    confidences = sorted(df_subset["confidence"].unique())

    print("\n✅ Unique scenarios found:", scenarios)
    print("📆 Unique years found:", years)
    print("📆 Unique transect_ids found:", len(transect_ids))
    print("📆 Unique confidence levels found:", confidences)
else:
    print("\n❌ 'scenario' or 'year' column not found in input data.")


# %%
df_meta = table.select(["transect_id", "scenario", "quantile", "year"]).to_pandas()

# Get unique values
transects = df_meta["transect_id"].unique()
scenarios = sorted(df_meta["scenario"].unique())
quantiles = sorted(df_meta["quantile"].unique())
years = sorted(df_meta["year"].unique())

print("\n✅ Unique scenarios found:", scenarios)
print("📆 Unique years found:", years)
print("🎯 Unique quantiles found:", quantiles)


# %%
# Check: All transects must have all combinations of scenario, quantile, and year
expected_combinations = set((s, q, y) for s in scenarios for q in quantiles for y in years)
errors = []

for tid in transects:
    df_tid = df_meta[df_meta["transect_id"] == tid]
    actual = set(df_tid[["scenario", "quantile", "year"]].itertuples(index=False, name=None))
    missing = expected_combinations - actual
    if missing:
        errors.append((tid, missing))

if errors:
    print(f"\n❌ {len(errors)} transect_ids are missing combinations. Showing first 3:\n")
    for tid, missing in errors[:3]:
        print(f"🚫 transect_id {tid} is missing {len(missing)} combinations")
else:
    print("\n✅ All transect_ids contain full scenario/quantile/year combinations.")


# %%
# --- Check values over time for 3 random transects and 3 random scenarios ---
columns_to_check = ["R2pIndex_combined_rp1", "R2pIndex_combined_rp100"]

# Read only relevant data to reduce memory usage
df = table.select(["transect_id", "scenario", "quantile", "year"] + columns_to_check).to_pandas()

# Filter for quantile == "0.5" (assuming "median" means quantile 0.5)
df = df[df["quantile"] == 0.5]

sample_transects = random.sample(list(transects), 3)
sample_scenarios = random.sample(scenarios, 3)

print(f"\n🔍 Checking monotonic increase for 3 random transect_ids and scenarios:")
for tid in sample_transects:
    for sc in sample_scenarios:
        df_sub = df[(df["transect_id"] == tid) & (df["scenario"] == sc)].sort_values("year")
        print(f"\n📊 transe# Filter using pyarrow expression
filtered_table = table.filter(
    (pc.field("transect_id") == 5510) & (pc.field("scenario") == "ssp370")
)

# Convert to pandas DataFrame
filtered_df = filtered_table.to_pandas()

# Show the result
print(filtered_df.head())
# Filter using pyarrow expression
filtered_table = table.filter(
    (pc.field("transect_id") == 5510) & (pc.field("scenario") == "ssp370")
)

# Convert to pandas DataFrame
filtered_df = filtered_table.to_pandas()

# Show the result
print(filtered_df.head())ct_id={tid}, scenario={sc}")

        for col in columns_to_check:
            values = df_sub[col].values
            is_increasing = np.all(np.diff(values) >= 0)
            trend = "✅ Increasing" if is_increasing else "⚠️ Not increasing"
            print(f"  - {col}: {trend}")

# %%
import pyarrow.compute as pc
# Load table
table = pq.read_table(file_path, use_threads=True)

# Filter using pyarrow expression
filtered_table = table.filter(
    (pc.field("transect_id") == 5510) & (pc.field("scenario") == "ssp370")
)

# Convert to pandas DataFrame
filtered_df = filtered_table.to_pandas()

# Show the result
print(filtered_df.head())


# %%
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import random

# Load data
table = pq.read_table(file_path, use_threads=True)

# Relevant columns
columns_to_check = ["R2pIndex_combined_rp1", "R2pIndex_combined_rp100"]
select_cols = ["transect_id", "scenario", "quantile", "year"] + columns_to_check

# Convert to pandas
df = table.select(select_cols).to_pandas()

# Use quantile == 0.5 for median (adjust if needed)
df = df[df["quantile"] == 0.5]

# Sample 3 transects and 3 scenarios
transects = df["transect_id"].unique()
scenarios = df["scenario"].unique()

sample_transects = random.sample(list(transects), 3)
sample_scenarios = random.sample(list(scenarios), 3)

# Collect results
results = []

for tid in sample_transects:
    for sc in sample_scenarios:
        df_sub = df[(df["transect_id"] == tid) & (df["scenario"] == sc)].sort_values("year")

        result_row = {
            "transect_id": tid,
            "scenario": sc,
        }

        for col in columns_to_check:
            values = df_sub[col].values
            is_increasing = np.all(np.diff(values) >= 0)
            result_row[f"{col}_increasing"] = is_increasing

        results.append(result_row)

# Create DataFrame of results
results_df = pd.DataFrame(results)

# Optional: pretty display
print("\n📋 Monotonicity Check Results:")
print(results_df)


# %%
filtered_df = results_df[(results_df["transect_id"] == 2892) & (results_df["scenario"] == "ssp370")]
print(filtered_df)


# %%

import pyarrow.compute as pc
file_path2 = '/Users/tessamoller/Downloads/Atoll_BEWARE_processed_outputs_2020-2150_short-4.parquet'
# Load table
table2 = pq.read_table(file_path2, use_threads=True)

# Filter using pyarrow expression
filtered_table2 = table2.filter((pc.field("transect_id") == 2892) )

# Convert to pandas DataFrame
filtered_df2 = table2.to_pandas()

# Show the result
print(filtered_df2.head())

# %%
len(filtered_df['year'].unique())*len(filtered_df['quantile'].unique())* (len(filtered_df['scenario'].unique())+1)*5686

# %%
311342616/443508

# %%
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Load table
table = pq.read_table(file_path, use_threads=True)

# Filter using pyarrow expression
filtered_table = table.filter(
    (pc.field("transect_id") == 2580) & (pc.field("scenario") == "ssp370")
)

# Convert to pandas DataFrame
filtered_df = filtered_table.to_pandas()

# Show the result
print(filtered_df.head())


# %%
filtered_df

# %%
import pyarrow.parquet as pq
import pyarrow.compute as pc

file_path2 = '/Users/tessamoller/Downloads/Atoll_BEWARE_processed_outputs_2020-2150_short-2.parquet'

# Load table
table2 = pq.read_table(file_path2, use_threads=True)

# Filter for transect_id == 2892
filtered_table2 = table2.filter(pc.field("transect_id") == 2892)

# Convert to pandas
filtered_df2 = filtered_table2.to_pandas()

# Keep only the first row for each year
filtered_unique_per_year = filtered_df2.drop_duplicates(subset="year", keep="first")

# Show result
print(filtered_unique_per_year)


# %%
#9126 for all?

# %%
# Load table
table = pq.read_table(file_path, use_threads=True)

# Filter using pyarrow expression
filtered_table = table.filter(
    (pc.field("transect_id") == 2580) & (pc.field("scenario") == "ssp370")
)

# Convert to pandas DataFrame
filtered_df = filtered_table.to_pandas()

# Keep only the first row for each year
filtered_unique_per_year = filtered_df.drop_duplicates(subset="year", keep="second")

# Show result
print(filtered_unique_per_year)


# %%
filtered_unique_per_year

# %%
# Sort by year to ensure consistent order
filtered_df_sorted = filtered_df.sort_values("year")

# Group by year and get the second entry using .nth(1) (0-based indexing)
second_per_year = filtered_df_sorted.groupby("year", as_index=False).nth(100)

# Group by year and get the second entry using .nth(1) (0-based indexing)
second_per_year

# %%
2678106 / 311342616

# %%
311342616 / 2678106

# %%
