import arcpy
import torch
import sys

def map_tensors_to_raster():
    """
    Converts PyTorch output tensors back into Georeferenced ArcGIS Rasters.
    Supports both ArcGIS Pro Script Tool inputs and Terminal inputs.
    """
    input_pt_path = arcpy.GetParameterAsText(0)
    reference_raster_path = arcpy.GetParameterAsText(1)
    output_gdb = arcpy.GetParameterAsText(2)
    name_disaster = arcpy.GetParameterAsText(3) or "Pred_DisasterRisk"
    name_crop = arcpy.GetParameterAsText(4) or "Pred_CropRisk"
    name_tox = arcpy.GetParameterAsText(5) or "Pred_ToxicityRisk"

    if not input_pt_path and len(sys.argv) > 1: input_pt_path = sys.argv[1]
    if not reference_raster_path and len(sys.argv) > 2: reference_raster_path = sys.argv[2]
    if not output_gdb and len(sys.argv) > 3: output_gdb = sys.argv[3]
    if len(sys.argv) > 4: name_disaster = sys.argv[4]
    if len(sys.argv) > 5: name_crop = sys.argv[5]
    if len(sys.argv) > 6: name_tox = sys.argv[6]

    def log(msg):
        print(msg)
        arcpy.AddMessage(msg)
        
    def log_err(msg):
        print(f"ERROR: {msg}")
        arcpy.AddError(msg)

    if not input_pt_path or not output_gdb:
        log_err("Input Tensor, Reference Raster, and Output GDB are required.")
        print("Usage: python 04_arcpy_mapping_loop.py <input.pt> <ref_raster> <out_gdb>")
        return

    log("1. Loading Parameters...")
    arcpy.env.workspace = output_gdb
    arcpy.env.overwriteOutput = True

    log("2. Extracting Spatial Reference from Reference Raster...")
    ref_raster = arcpy.Raster(reference_raster_path)
    lower_left = arcpy.Point(ref_raster.extent.XMin, ref_raster.extent.YMin)
    cell_size = ref_raster.meanCellWidth
    spatial_ref = ref_raster.spatialReference

    log(f"3. Loading Tensors from {input_pt_path}...")
    try:
        tensors = torch.load(input_pt_path)
    except Exception as e:
        log_err(f"Failed to load tensor file. Error: {e}")
        return

    def tensor_to_raster(tensor, out_name):
        array = tensor.squeeze().numpy()
        raster = arcpy.NumPyArrayToRaster(array, lower_left, cell_size, value_to_nodata=0)
        arcpy.management.DefineProjection(raster, spatial_ref)
        raster.save(out_name)
        log(f" -> Saved Raster: {out_name}")

    log("4. Converting Tensors to Rasters...")
    tensor_to_raster(tensors['disaster'], name_disaster)
    tensor_to_raster(tensors['agriculture'], name_crop)
    tensor_to_raster(tensors['environment'], name_tox)

    log("Mapping Loop Complete! The Cascade has been mapped. Add the rasters to your map.")

if __name__ == '__main__':
    map_tensors_to_raster()
