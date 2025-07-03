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
import geopandas as gpd
import pandas as pd

# %%
results = pd.read_parquet("../data/processed/Atoll_BEWARE_inputs.parquet")
results = results[
    (results["scenario"] == "ssp119")
    & (results["year"] == 2020)
    & (results["quantile"] == 0.50)
]
results

# %%
# Read in shapefiles
shapefile_path = "../data/processed/Atoll_transects_centroids.shp"
gdf = gpd.read_file(shapefile_path)[["transect_i", "geometry", "Atoll_FID"]]
gdf

# %%
# Add Atoll_FID and geometry to results
results_all = gdf.merge(results, on=["transect_i"], how="left")
results_all

# save only 2 signs of significance for eta_SLR, eta_combined_rp1, eta_ombined_rp10, eta_combined_rp100

results_all2 = results_all.drop(columns=["H0", "L0", "H0L0", "W_reef", "beta_f"]).round(
    2
)
results_all2

# %%
selection = results[(results["scenario"] == "ssp119") & (results["year"] == 2020)]
selection

# %%
missing_from_df2 = selection[~selection["transect_i"].isin(gdf["transect_i"])]
missing_from_df2

# %%
missing_from_df1 = gdf[~gdf["transect_i"].isin(selection["transect_i"])]
missing_from_df1

# %%
gdf

# %%
selection

# %%
missing_ids_from_df2 = set(selection["transect_i"]) - set(gdf["transect_i"])
missing_ids_from_df1 = set(gdf["transect_i"]) - set(selection["transect_i"])


# %%
missing_ids_from_df2

# %%
missing_ids_from_df1

# %%
selection["transect_i"][selection["transect_i"].duplicated()].unique()

# %%
results[
    (results["scenario"] == "ssp119")
    & (results["year"] == 2020)
    & (results["transect_i"] == 538)
]

# %%
gdf = gpd.read_file(shapefile_path)
gdf[gdf["transect_i"] == 4886]

# %%
selection.columns

# %%
