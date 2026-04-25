import arcpy
import numpy as np
import torch
import torch.nn as nn
import os
import sys

def run_inference():
    """
    Runs the multi-head PyTorch CNN inference on stacked environmental features.
    Supports both ArcGIS Pro Script Tool inputs and Terminal inputs.
    """
    workspace_gdb = arcpy.GetParameterAsText(0)
    slope_path = arcpy.GetParameterAsText(1)
    flow_path = arcpy.GetParameterAsText(2)
    dist_path = arcpy.GetParameterAsText(3)
    ndvi_path = arcpy.GetParameterAsText(4)
    ndwi_path = arcpy.GetParameterAsText(5)
    output_pt_path = arcpy.GetParameterAsText(6)

    if not workspace_gdb and len(sys.argv) > 1: workspace_gdb = sys.argv[1]
    if not slope_path and len(sys.argv) > 2: slope_path = sys.argv[2]
    if not flow_path and len(sys.argv) > 3: flow_path = sys.argv[3]
    if not dist_path and len(sys.argv) > 4: dist_path = sys.argv[4]
    if not ndvi_path and len(sys.argv) > 5: ndvi_path = sys.argv[5]
    if not ndwi_path and len(sys.argv) > 6: ndwi_path = sys.argv[6]
    if not output_pt_path and len(sys.argv) > 7: output_pt_path = sys.argv[7]

    def log(msg):
        print(msg)
        arcpy.AddMessage(msg)

    if not output_pt_path:
        print("Usage: python 03_pytorch_multitask_model.py <workspace> <slope> <flow> <dist> <ndvi> <ndwi> <output.pt>")
        return

    log("1. Loading Inputs...")
    arcpy.env.workspace = workspace_gdb

    log(" - Converting Rasters to NumPy Arrays...")
    arr_slope = arcpy.RasterToNumPyArray(slope_path, nodata_to_value=0)
    arr_flow  = arcpy.RasterToNumPyArray(flow_path, nodata_to_value=0)
    arr_dist  = arcpy.RasterToNumPyArray(dist_path, nodata_to_value=0)
    arr_ndvi  = arcpy.RasterToNumPyArray(ndvi_path, nodata_to_value=0)
    arr_ndwi  = arcpy.RasterToNumPyArray(ndwi_path, nodata_to_value=0)

    # Shape: [Channels(5), Height, Width]
    stacked_array = np.stack([arr_slope, arr_flow, arr_dist, arr_ndvi, arr_ndwi], axis=0)
    tensor_input = torch.from_numpy(stacked_array).float().unsqueeze(0) # Add batch dimension

    log("2. Defining the Multi-Head HazardNet...")
    class MultiHeadHazardNet(nn.Module):
        def __init__(self):
            super(MultiHeadHazardNet, self).__init__()
            self.shared_conv = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.head_disaster = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
            self.head_agriculture = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
            self.head_environment = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

        def forward(self, x):
            features = self.shared_conv(x)
            return self.head_disaster(features), self.head_agriculture(features), self.head_environment(features)

    model = MultiHeadHazardNet()
    model.eval()

    with torch.no_grad():
        log("Running PyTorch Multi-Task Inference...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        tensor_input = tensor_input.to(device)
        
        pred_disaster, pred_agriculture, pred_env = model(tensor_input)

    log(f"3. Saving Output Tensors to {output_pt_path}...")
    torch.save({
        'disaster': pred_disaster.cpu(),
        'agriculture': pred_agriculture.cpu(),
        'environment': pred_env.cpu()
    }, output_pt_path)

    log("PyTorch Inference Complete! Run the Mapping Loop script next to generate Rasters.")

if __name__ == '__main__':
    run_inference()
