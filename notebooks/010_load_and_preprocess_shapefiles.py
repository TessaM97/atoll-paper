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

import folium
import geopandas as gpd
import matplotlib.pyplot as plt

# %%
# Add project root to Python path
sys.path.append(str(Path().resolve().parent))
from src.settings import DATA_DIR, INTERIM_DIR, SHAPEFILE_PATH

print("Using data directory:", DATA_DIR)

# %%
# Load the shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)

# Display the first few rows of attribute data
print("Attribute Data:")
print(gdf.head())

# Display column names
print("\nColumns (Attributes):")
print(gdf.columns)

# Access geometry (location data)
print("\nGeometry of first item:")
print(gdf.geometry.iloc[0])

print("Shapefile CRS:")
print(gdf.crs)

# %%
# Load and reproject shapefile to get accurate centroids

gdf = gdf.to_crs(epsg=3857)  # in meters

# Finds centroid of each shape
gdf["centroid"] = gdf.geometry.centroid
gdf["geometry"] = gdf["centroid"].to_crs(epsg=4326)  # in degrees

gdf

# Extract lat/lon from geometry
lats = gdf.geometry.y.values
lons = gdf.geometry.x.values

# Print ranges to understand coordinate "shape"
print(f"\nLatitude range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")


# %%
# Plot the centroids
fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, color="blue", markersize=5, alpha=0.7)

ax.set_title("Centroid Locations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# Get the centroid of all points to center the map
center = gdf.geometry.unary_union.centroid
m = folium.Map(location=[center.y, center.x], zoom_start=4)

# Add each point as a marker
for pt in gdf.geometry:
    folium.CircleMarker(
        location=[pt.y, pt.x], radius=2, color="blue", fill=True, fill_opacity=0.7
    ).add_to(m)

m


# %%
# Saving to file
file_path_output = INTERIM_DIR / "Shapefiles/Atoll_transects_centroids.shp"
gdf = gdf.drop(columns="centroid")  # centroids is in EPSG3857, in meters
gdf.to_file(file_path_output)
