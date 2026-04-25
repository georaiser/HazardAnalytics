import arcpy
from arcpy.sa import *
from arcpy.ia import *
import os

def execute_feature_engineering():
    """
    Automates the calculation of spatial indices and hydrological routing.
    Designed to be run as an ArcGIS Pro Script Tool.
    
    Parameters:
    0: Target Geodatabase Workspace (Data Type: Workspace)
    1: DEM Raster (Data Type: Raster Layer)
    2: Sentinel-2 Multispectral Raster (Data Type: Raster Layer)
    3: Geological Fault Lines (Data Type: Feature Layer)
    4: Output Feature Stack - Optional (Data Type: Raster Layer, Direction: Output)
    """
    workspace_gdb = arcpy.GetParameterAsText(0)
    dem_path = arcpy.GetParameterAsText(1)
    sentinel_path = arcpy.GetParameterAsText(2)
    faults_path = arcpy.GetParameterAsText(3)

    if not workspace_gdb or not dem_path or not sentinel_path:
        arcpy.AddError("Workspace, DEM, and Sentinel paths are required.")
        return

    arcpy.env.workspace = workspace_gdb
    arcpy.env.overwriteOutput = True
    
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("Image")

    arcpy.AddMessage("1. Calculating Topography & Terrain Analysis...")
    slope_raster = Slope(dem_path, "DEGREE")
    slope_raster.save("Hazard_Slope")

    arcpy.AddMessage(" - Calculating Aspect...")
    aspect_raster = Aspect(dem_path)
    aspect_raster.save("Hazard_Aspect")

    arcpy.AddMessage(" - Calculating Curvature...")
    out_curv = Curvature(dem_path, z_factor=1, out_profile_curve="Hazard_ProfCurve", out_plan_curve="Hazard_PlanCurve")
    out_curv.save("Hazard_Curvature")

    arcpy.AddMessage(" - Calculating Terrain Roughness (Focal Statistics)...")
    neighborhood = NbrRectangle(3, 3, "CELL")
    roughness_raster = FocalStatistics(dem_path, neighborhood, "STD")
    roughness_raster.save("Hazard_Roughness")

    arcpy.AddMessage(" - Reclassifying Slope into Risk Zones...")
    remap = RemapRange([[0, 15, 1], [15, 30, 2], [30, 90, 3]])
    slope_risk = Reclassify(slope_raster, "Value", remap)
    slope_risk.save("Hazard_SlopeRisk")

    arcpy.AddMessage("2. Calculating Hydrology & Routing...")
    filled_dem = Fill(dem_path)
    flow_dir = FlowDirection(filled_dem)
    flow_acc = FlowAccumulation(flow_dir)
    flow_acc.save("Hazard_FlowAcc")

    if faults_path:
        arcpy.AddMessage(" - Calculating distance to geological faults...")
        out_dist = DistanceAccumulation(faults_path, dem_path)
        out_dist.save("Hazard_FaultDist")
    else:
        arcpy.AddWarning("No Fault Lines provided. Skipping Fault Distance calculation.")

    arcpy.AddMessage("3. Calculating Spectral Indices (Crop Health)...")
    red_band = arcpy.Raster(f"{sentinel_path}/Band_4")
    nir_band = arcpy.Raster(f"{sentinel_path}/Band_8")
    green_band = arcpy.Raster(f"{sentinel_path}/Band_3")

    ndvi = Float(nir_band - red_band) / Float(nir_band + red_band) 
    ndwi = Float(green_band - nir_band) / Float(green_band + nir_band) 
    ndvi.save("Hazard_NDVI")
    ndwi.save("Hazard_NDWI")

    arcpy.AddMessage("4. Preparing Data for Deep Learning (Data Management)...")
    dem_10m_path = "Hazard_DEM_10m"
    arcpy.management.Resample(dem_path, dem_10m_path, "10 10", "BILINEAR")

    arcpy.AddMessage("Spatial feature engineering complete! Rasters are ready for PyTorch.")

if __name__ == '__main__':
    execute_feature_engineering()
