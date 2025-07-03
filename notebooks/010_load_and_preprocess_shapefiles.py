# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import folium
import geopandas as gpd
import matplotlib.pyplot as plt

# %%
# Load the shapefile
shapefile_path = "../data/Shapefiles/Atoll_transects.shp"
gdf = gpd.read_file(shapefile_path)

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
# No longer needed: Reproject lon to 0 to 360

## Reproject to EPSG:3857 for accurate centroid calculation
# gdf_reproj = gdf

## Convert centroids to WGS84 (EPSG:4326)
# centroids_WGS84 = gdf_reproj['centroid'].to_crs(epsg=4326)

# Extract and adjust longitudes to [0, 360]
# adjusted_geometries = []
# for point in centroids_WGS84:
#    lon, lat = point.x, point.y
#    lon = (lon + 360) % 360  # shift to [0, 360]
#    adjusted_geometries.append(Point(lon, lat))

# Replace geometry with adjusted geometries
# gdf_reproj = gdf_reproj.set_geometry(adjusted_geometries)

# Extract lat/lon for reporting
# lats = gdf_reproj.geometry.y.values
# lons = gdf_reproj.geometry.x.values

# Print ranges to understand coordinate "shape"
# print(f"\nLatitude range: {lats.min():.2f} to {lats.max():.2f}")
# print(f"Longitude range: {lons.min():.2f} to {lons.max():.2f}")


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
gdf = gdf.drop(columns="centroid")  # centroids is in EPSG3857, in meters
gdf.to_file("../data/processed/Atoll_transects_centroids.shp")

# %%
