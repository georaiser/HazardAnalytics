import arcpy
from arcpy.sa import *
from arcpy.ia import *
import os

# 1. Setup Workspace and Projections
# You will need to change this path to point to your actual local Geodatabase.
arcpy.env.workspace = r"C:\Path\To\CascadeProject.gdb"
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("Image")

# Set these paths to where you downloaded the raw data in Phase 2
dem_path = r"C:\RawData\Maule_DEM.tif"
sentinel_path = r"C:\RawData\Sentinel2_L2A.tif"

print("1. Calculating Topography & Terrain Analysis...")
# Slope: Rate of maximum change in z-value.
slope_raster = Slope(dem_path, "DEGREE")
slope_raster.save("Cascade_Slope")

# Aspect: The compass direction that a topographic slope faces (useful for vegetation/moisture).
print(" - Calculating Aspect...")
aspect_raster = Aspect(dem_path)
aspect_raster.save("Cascade_Aspect")

# Curvature: The second derivative of the surface (concave/convex). Useful for water pooling and landslide modeling.
print(" - Calculating Curvature...")
# The Curvature tool outputs the primary curvature, but optionally profile and planform curvature.
out_curv = Curvature(dem_path, z_factor=1, out_profile_curve="Cascade_ProfCurve", out_plan_curve="Cascade_PlanCurve")
out_curv.save("Cascade_Curvature")

# Focal Statistics: Neighborhood analysis. Let's calculate terrain roughness (standard deviation of elevation in a 3x3 window).
print(" - Calculating Terrain Roughness (Focal Statistics)...")
neighborhood = NbrRectangle(3, 3, "CELL")
roughness_raster = FocalStatistics(dem_path, neighborhood, "STD")
roughness_raster.save("Cascade_Roughness")

# Reclassify: Categorizing continuous data into discrete classes.
# Example: Slope risk classes: 0-15° (Low=1), 15-30° (Medium=2), >30° (High=3)
print(" - Reclassifying Slope into Risk Zones...")
remap = RemapRange([[0, 15, 1], [15, 30, 2], [30, 90, 3]])
slope_risk = Reclassify(slope_raster, "Value", remap)
slope_risk.save("Cascade_SlopeRisk")


print("2. Calculating Hydrology & Routing...")
filled_dem = Fill(dem_path)
flow_dir = FlowDirection(filled_dem)
flow_acc = FlowAccumulation(flow_dir)
flow_acc.save("Cascade_FlowAcc")

# Calculate distance to geological faults
# You will need to ensure the Sernageomin_Faults shapefile is loaded into your GDB
out_dist = DistanceAccumulation("Sernageomin_Faults", dem_path)
out_dist.save("Cascade_FaultDist")

print("3. Calculating Spectral Indices (Crop Health)...")
# Sentinel-2: Band 8 (NIR), Band 4 (Red), Band 3 (Green)
red_band = arcpy.Raster(f"{sentinel_path}/Band_4")
nir_band = arcpy.Raster(f"{sentinel_path}/Band_8")
green_band = arcpy.Raster(f"{sentinel_path}/Band_3")

ndvi = Float(nir_band - red_band) / Float(nir_band + red_band) # Crop Health
ndwi = Float(green_band - nir_band) / Float(green_band + nir_band) # Water Bodies
ndvi.save("Cascade_NDVI")
ndwi.save("Cascade_NDWI")

print("4. Preparing Data for Deep Learning (Data Management)...")
# Deep learning models require stacked tensors of the same shape and resolution.
# We use Resample to ensure all rasters have the exact same cell size.
print(" - Resampling DEM to match Sentinel-2 resolution (e.g., 10x10 meters)...")
# Note: In practice, use arcpy.GetRasterProperties_management to dynamically get the Sentinel cell size.
# Here we use "10 10" as an example for Sentinel-2 10m bands.
dem_10m_path = "Cascade_DEM_10m"
arcpy.management.Resample(dem_path, dem_10m_path, "10 10", "BILINEAR")

# Extract By Mask: If you have a specific Study Area polygon shapefile, you can clip all rasters to it.
# This prevents processing 'NoData' areas outside your region of interest.
# Example (commented out as we don't have the shapefile defined):
# print(" - Clipping to Study Area...")
# study_area_poly = "Study_Area_Boundary"
# clipped_ndvi = ExtractByMask(ndvi, study_area_poly)
# clipped_ndvi.save("Cascade_NDVI_Clipped")

# Composite Bands: To feed into PyTorch, it's often easier to stack all your features into a single multi-band raster.
# Example:
# print(" - Compositing features into a single stack...")
# arcpy.management.CompositeBands([dem_10m_path, "Cascade_Slope", "Cascade_NDVI"], "Cascade_FeatureStack.tif")

print("\nSpatial feature engineering complete! The data is now ready to be exported for PyTorch.")
