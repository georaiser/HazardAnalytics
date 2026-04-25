import arcpy
from arcpy.sa import *
from arcpy.ia import *
import os
import sys

def execute_feature_engineering():
    """
    Automates the calculation of spatial indices and hydrological routing.
    Supports both ArcGIS Pro Script Tool inputs and Terminal inputs.
    """
    workspace_gdb = arcpy.GetParameterAsText(0)
    dem_path = arcpy.GetParameterAsText(1)
    sentinel_path = arcpy.GetParameterAsText(2)
    faults_path = arcpy.GetParameterAsText(3)

    if not workspace_gdb and len(sys.argv) > 1: workspace_gdb = sys.argv[1]
    if not dem_path and len(sys.argv) > 2: dem_path = sys.argv[2]
    if not sentinel_path and len(sys.argv) > 3: sentinel_path = sys.argv[3]
    if not faults_path and len(sys.argv) > 4: faults_path = sys.argv[4]

    def log(msg):
        print(msg)
        arcpy.AddMessage(msg)
        
    def log_err(msg):
        print(f"ERROR: {msg}")
        arcpy.AddError(msg)
        
    def log_warn(msg):
        print(f"WARNING: {msg}")
        arcpy.AddWarning(msg)

    if not workspace_gdb or not dem_path or not sentinel_path:
        log_err("Workspace, DEM, and Sentinel paths are required.")
        print("Usage: python 02_arcpy_feature_engineering.py <workspace_gdb> <dem_path> <sentinel_path> [faults_path]")
        return

    arcpy.env.workspace = workspace_gdb
    arcpy.env.overwriteOutput = True
    
    arcpy.CheckOutExtension("Spatial")
    arcpy.CheckOutExtension("Image")

    log("1. Calculating Topography & Terrain Analysis...")
    slope_raster = Slope(dem_path, "DEGREE")
    slope_raster.save("Hazard_Slope")

    log(" - Calculating Aspect...")
    aspect_raster = Aspect(dem_path)
    aspect_raster.save("Hazard_Aspect")

    log(" - Calculating Curvature...")
    out_curv = Curvature(dem_path, z_factor=1, out_profile_curve="Hazard_ProfCurve", out_plan_curve="Hazard_PlanCurve")
    out_curv.save("Hazard_Curvature")

    log(" - Calculating Terrain Roughness (Focal Statistics)...")
    neighborhood = NbrRectangle(3, 3, "CELL")
    roughness_raster = FocalStatistics(dem_path, neighborhood, "STD")
    roughness_raster.save("Hazard_Roughness")

    log(" - Reclassifying Slope into Risk Zones...")
    remap = RemapRange([[0, 15, 1], [15, 30, 2], [30, 90, 3]])
    slope_risk = Reclassify(slope_raster, "Value", remap)
    slope_risk.save("Hazard_SlopeRisk")

    log("2. Calculating Hydrology & Routing...")
    filled_dem = Fill(dem_path)
    flow_dir = FlowDirection(filled_dem)
    flow_acc = FlowAccumulation(flow_dir)
    flow_acc.save("Hazard_FlowAcc")

    if faults_path and faults_path != "#" and faults_path != "":
        log(" - Calculating distance to geological faults...")
        out_dist = DistanceAccumulation(faults_path, dem_path)
        out_dist.save("Hazard_FaultDist")
    else:
        log_warn("No Fault Lines provided. Skipping Fault Distance calculation.")

    log("3. Calculating Spectral Indices (Crop Health)...")
    red_band = arcpy.Raster(f"{sentinel_path}/Band_4")
    nir_band = arcpy.Raster(f"{sentinel_path}/Band_8")
    green_band = arcpy.Raster(f"{sentinel_path}/Band_3")

    ndvi = Float(nir_band - red_band) / Float(nir_band + red_band) 
    ndwi = Float(green_band - nir_band) / Float(green_band + nir_band) 
    ndvi.save("Hazard_NDVI")
    ndwi.save("Hazard_NDWI")

    log("4. Preparing Data for Deep Learning (Data Management)...")
    dem_10m_path = "Hazard_DEM_10m"
    arcpy.management.Resample(dem_path, dem_10m_path, "10 10", "BILINEAR")

    log("Spatial feature engineering complete! Rasters are ready for PyTorch.")

if __name__ == '__main__':
    execute_feature_engineering()
