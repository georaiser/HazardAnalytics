import arcpy
import torch

def map_tensors_to_raster():
    """
    Converts PyTorch output tensors back into Georeferenced ArcGIS Rasters.
    Designed to be run as an ArcGIS Pro Script Tool.
    
    Parameters:
    0: Input Tensor File .pt (Data Type: File)
    1: Reference Raster (Data Type: Raster Layer)
    2: Output Geodatabase (Data Type: Workspace)
    3: Disaster Risk Name (Data Type: String)
    4: Crop Risk Name (Data Type: String)
    5: Toxicity Risk Name (Data Type: String)
    """
    arcpy.AddMessage("1. Loading Parameters...")
    input_pt_path = arcpy.GetParameterAsText(0)
    reference_raster_path = arcpy.GetParameterAsText(1)
    output_gdb = arcpy.GetParameterAsText(2)
    name_disaster = arcpy.GetParameterAsText(3) or "Pred_DisasterRisk"
    name_crop = arcpy.GetParameterAsText(4) or "Pred_CropRisk"
    name_tox = arcpy.GetParameterAsText(5) or "Pred_ToxicityRisk"

    arcpy.env.workspace = output_gdb
    arcpy.env.overwriteOutput = True

    arcpy.AddMessage("2. Extracting Spatial Reference from Reference Raster...")
    ref_raster = arcpy.Raster(reference_raster_path)
    lower_left = arcpy.Point(ref_raster.extent.XMin, ref_raster.extent.YMin)
    cell_size = ref_raster.meanCellWidth
    spatial_ref = ref_raster.spatialReference

    arcpy.AddMessage(f"3. Loading Tensors from {input_pt_path}...")
    try:
        tensors = torch.load(input_pt_path)
    except Exception as e:
        arcpy.AddError(f"Failed to load tensor file. Error: {e}")
        return

    def tensor_to_raster(tensor, out_name):
        array = tensor.squeeze().numpy()
        raster = arcpy.NumPyArrayToRaster(array, lower_left, cell_size, value_to_nodata=0)
        arcpy.management.DefineProjection(raster, spatial_ref)
        raster.save(out_name)
        arcpy.AddMessage(f" -> Saved Raster: {out_name}")

    arcpy.AddMessage("4. Converting Tensors to Rasters...")
    tensor_to_raster(tensors['disaster'], name_disaster)
    tensor_to_raster(tensors['agriculture'], name_crop)
    tensor_to_raster(tensors['environment'], name_tox)

    arcpy.AddMessage("Mapping Loop Complete! The Cascade has been mapped. Add the rasters to your map.")

if __name__ == '__main__':
    map_tensors_to_raster()
