import arcpy
import numpy as np
import torch
import torch.nn as nn
import os

def run_inference():
    """
    Runs the multi-head PyTorch CNN inference on stacked environmental features.
    Designed to be run as an ArcGIS Pro Script Tool.
    
    Parameters:
    0: Workspace Geodatabase (Data Type: Workspace)
    1: Slope Raster (Data Type: Raster Layer)
    2: Flow Accumulation Raster (Data Type: Raster Layer)
    3: Distance to Faults Raster (Data Type: Raster Layer)
    4: NDVI Raster (Data Type: Raster Layer)
    5: NDWI Raster (Data Type: Raster Layer)
    6: Output Tensor File .pt (Data Type: File)
    """
    arcpy.AddMessage("1. Loading Inputs...")
    
    workspace_gdb = arcpy.GetParameterAsText(0)
    slope_path = arcpy.GetParameterAsText(1)
    flow_path = arcpy.GetParameterAsText(2)
    dist_path = arcpy.GetParameterAsText(3)
    ndvi_path = arcpy.GetParameterAsText(4)
    ndwi_path = arcpy.GetParameterAsText(5)
    output_pt_path = arcpy.GetParameterAsText(6)

    arcpy.env.workspace = workspace_gdb

    arcpy.AddMessage(" - Converting Rasters to NumPy Arrays...")
    arr_slope = arcpy.RasterToNumPyArray(slope_path, nodata_to_value=0)
    arr_flow  = arcpy.RasterToNumPyArray(flow_path, nodata_to_value=0)
    arr_dist  = arcpy.RasterToNumPyArray(dist_path, nodata_to_value=0)
    arr_ndvi  = arcpy.RasterToNumPyArray(ndvi_path, nodata_to_value=0)
    arr_ndwi  = arcpy.RasterToNumPyArray(ndwi_path, nodata_to_value=0)

    # Shape: [Channels(5), Height, Width]
    stacked_array = np.stack([arr_slope, arr_flow, arr_dist, arr_ndvi, arr_ndwi], axis=0)
    tensor_input = torch.from_numpy(stacked_array).float().unsqueeze(0) # Add batch dimension

    arcpy.AddMessage("2. Defining the Multi-Head HazardNet...")
    class MultiHeadHazardNet(nn.Module):
        def __init__(self):
            super(MultiHeadHazardNet, self).__init__()
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

    model = MultiHeadHazardNet()
    model.eval()

    with torch.no_grad():
        arcpy.AddMessage("Running PyTorch Multi-Task Inference...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        tensor_input = tensor_input.to(device)
        
        pred_disaster, pred_agriculture, pred_env = model(tensor_input)

    arcpy.AddMessage(f"3. Saving Output Tensors to {output_pt_path}...")
    torch.save({
        'disaster': pred_disaster.cpu(),
        'agriculture': pred_agriculture.cpu(),
        'environment': pred_env.cpu()
    }, output_pt_path)

    arcpy.AddMessage("PyTorch Inference Complete! Run the Mapping Loop script next to generate Rasters.")

if __name__ == '__main__':
    run_inference()
