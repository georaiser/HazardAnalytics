import arcpy
import numpy as np
import torch
import torch.nn as nn

print("1. Bridging ArcGIS to PyTorch Tensors...")
# Extract spatial reference data from base raster
# Make sure your workspace is set correctly before running this script
# arcpy.env.workspace = r"C:\Path\To\CascadeProject.gdb"
base_raster = arcpy.Raster("Cascade_Slope")
lower_left = arcpy.Point(base_raster.extent.XMin, base_raster.extent.YMin)
cell_size = base_raster.meanCellWidth
spatial_ref = base_raster.spatialReference

# Convert Rasters to NumPy (ensure identical extents/resolutions first)
arr_slope = arcpy.RasterToNumPyArray("Cascade_Slope", nodata_to_value=0)
arr_flow  = arcpy.RasterToNumPyArray("Cascade_FlowAcc", nodata_to_value=0)
arr_dist  = arcpy.RasterToNumPyArray("Cascade_FaultDist", nodata_to_value=0)
arr_ndvi  = arcpy.RasterToNumPyArray("Cascade_NDVI", nodata_to_value=0)
arr_ndwi  = arcpy.RasterToNumPyArray("Cascade_NDWI", nodata_to_value=0)

# Shape: [Channels(5), Height, Width]
stacked_array = np.stack([arr_slope, arr_flow, arr_dist, arr_ndvi, arr_ndwi], axis=0)
tensor_input = torch.from_numpy(stacked_array).float().unsqueeze(0) # Add batch dimension

print("2. Defining the Multi-Head CascadeNet...")
class MultiHeadCascadeNet(nn.Module):
    def __init__(self):
        super(MultiHeadCascadeNet, self).__init__()
        # Shared Encoder (Feature Extractor)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Head 1: Disaster (Flood/Landslide Risk)
        self.head_disaster = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Head 2: Agriculture (Crop Vulnerability)
        self.head_agriculture = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Head 3: Environment (Toxic Runoff)
        self.head_environment = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared_conv(x)
        out_disaster = self.head_disaster(features)
        out_agriculture = self.head_agriculture(features)
        out_environment = self.head_environment(features)
        return out_disaster, out_agriculture, out_environment

model = MultiHeadCascadeNet()
model.eval()

# NOTE: In reality, you would load pre-trained weights here via model.load_state_dict()
# For this tutorial, we simulate a forward pass with initialized weights.

with torch.no_grad():
    print("Running PyTorch Multi-Task Inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    tensor_input = tensor_input.to(device)
    
    pred_disaster, pred_agriculture, pred_env = model(tensor_input)

print("3. Bridging PyTorch Tensors back to ArcGIS...")
def tensor_to_raster(tensor, out_name):
    array = tensor.cpu().squeeze().numpy()
    raster = arcpy.NumPyArrayToRaster(array, lower_left, cell_size, value_to_nodata=0)
    arcpy.management.DefineProjection(raster, spatial_ref)
    raster.save(out_name)
    print(f"Saved: {out_name}")

tensor_to_raster(pred_disaster, "Pred_DisasterRisk")
tensor_to_raster(pred_agriculture, "Pred_CropRisk")
tensor_to_raster(pred_env, "Pred_ToxicityRisk")

print("Pipeline Complete! The Cascade has been mapped.")
