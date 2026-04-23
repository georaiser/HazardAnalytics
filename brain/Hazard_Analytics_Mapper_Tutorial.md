# Hazard Analytics Mapper: The Complete Tutorial
**Building an AI-Enhanced Flood & Agricultural Risk Engine in Chile**

Welcome to the Hazard Analytics Mapper project tutorial! Since you are currently on Linux but will be building the core spatial pipeline in **Windows (ArcGIS Pro)**, this document serves as your complete, step-by-step master guide. 

We will use the **Maule Region (Región del Maule)** in Chile as our target. It has abundant open data, significant agricultural activity, and a history of extreme winter flooding, making it the perfect testbed.

---

## 🛠️ Step 1: Environment Setup (Windows)

Before writing any code, we need to bridge the ESRI ecosystem with deep learning.

1. **Launch ArcGIS Pro** and create a new project called `Hazard Analytics Mapper`.
2. **Clone the Python Environment:**
   * Go to **Settings** > **Package Manager**.
   * By default, the `arcgispro-py3` environment is read-only. Click the "gear" icon and **Clone** it (name it `terracascade-env`).
   * Activate the new cloned environment.
3. **Install Deep Learning Libraries:**
   * Open the **Python Command Prompt** (search for it in your Windows Start Menu under the ArcGIS folder).
   * Run the following command to install PyTorch with CUDA support (ArcGIS Pro officially supports this via `conda`):
     ```bash
     conda install -c esri deep-learning-essentials
     ```
   * *This installs PyTorch, torchvision, and fastai pre-configured to work with `arcpy`.*

---

## 🗺️ Step 2: Downloading Open Data (Chile)

You will need three free datasets for the Maule Region.

1. **Topography (DEM):**
   * Go to the **Alaska Satellite Facility (ASF)** vertex portal (search "ASF Vertex").
   * Draw a box over Talca/Constitución (Maule).
   * Filter for **ALOS PALSAR** and download the `Hi-Res Terrain Corrected (12.5m)` GeoTIFF.
2. **Optical Imagery (Sentinel-2):**
   * Go to the **Copernicus Browser** (browser.dataspace.copernicus.eu).
   * Find a cloud-free summer image (January/February) of Maule.
   * Download the **Level-2A** product (this is already atmospherically corrected, which is crucial for NDVI).
3. **Hydrology & Faults:**
   * Go to the **IDE Minagri** (ide.minagri.gob.cl) for agricultural boundaries.
   * Go to **Sernageomin** for geological fault lines. Download the Shapefiles.

---

## ⚙️ Step 3: The `arcpy` Data Processing Script

Once you have your raw `.tif` and `.shp` files in a folder, open a Jupyter Notebook inside ArcGIS Pro (Insert > New Notebook). This script prepares the data.

```python
import arcpy
from arcpy.sa import *
from arcpy.ia import *
import os

# 1. Setup Workspace
arcpy.env.workspace = r"C:\Path\To\Hazard Analytics Mapper\Hazard Analytics Mapper.gdb"
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("Image")

dem_path = r"C:\Path\To\RawData\Maule_DEM_12m.tif"
sentinel_path = r"C:\Path\To\RawData\Sentinel2_Maule.tif"

print("1. Calculating Topography...")
# Calculate Slope
slope_raster = Slope(dem_path, "DEGREE")
slope_raster.save("Maule_Slope")

print("2. Calculating Hydrology...")
# Fill sinks in the DEM, then calculate Flow Direction
filled_dem = Fill(dem_path)
flow_dir = FlowDirection(filled_dem)
flow_acc = FlowAccumulation(flow_dir)
flow_acc.save("Maule_FlowAcc")

print("3. Calculating Spectral Indices (NDVI)...")
# Assuming Band 8 is NIR and Band 4 is Red for Sentinel-2
red_band = arcpy.Raster(f"{sentinel_path}/Band_4")
nir_band = arcpy.Raster(f"{sentinel_path}/Band_8")

# Map Algebra for NDVI
ndvi = Float(nir_band - red_band) / Float(nir_band + red_band)
ndvi.save("Maule_NDVI")

print("Data Pre-processing Complete!")
```

---

## 🧠 Step 4: The PyTorch & `arcpy` Bridge

Now we convert the spatial rasters into NumPy arrays to feed into PyTorch. This is the hardest, but most rewarding part.

```python
import numpy as np
import torch
import torch.nn as nn

# 1. Convert ArcGIS Rasters to NumPy
print("Converting to Tensors...")
lower_left = arcpy.Point(dem_raster.extent.XMin, dem_raster.extent.YMin)
cell_size = dem_raster.meanCellWidth

# Extract arrays (ensure they all have the same dimensions)
arr_slope = arcpy.RasterToNumPyArray("Maule_Slope", nodata_to_value=0)
arr_ndvi = arcpy.RasterToNumPyArray("Maule_NDVI", nodata_to_value=0)
arr_flow = arcpy.RasterToNumPyArray("Maule_FlowAcc", nodata_to_value=0)

# 2. Stack into a multi-channel PyTorch Tensor
# Shape: [Channels, Height, Width]
stacked_array = np.stack([arr_slope, arr_ndvi, arr_flow], axis=0)
tensor_input = torch.from_numpy(stacked_array).float().unsqueeze(0) # Add batch dimension
print(f"Tensor Shape: {tensor_input.shape}")

# 3. Simple PyTorch Inference (Example CNN)
class CascadeNet(nn.Module):
    def __init__(self):
        super(CascadeNet, self).__init__()
        # 3 Input channels (Slope, NDVI, Flow) -> 1 Output (Risk Score)
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.conv(x))

model = CascadeNet()
model.eval() # Set to inference mode

with torch.no_grad():
    print("Running PyTorch prediction on GPU...")
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    tensor_input = tensor_input.to(device)
    
    output_tensor = model(tensor_input)

# 4. Convert PyTorch Output back to ArcGIS Raster
output_array = output_tensor.cpu().squeeze().numpy()

final_risk_raster = arcpy.NumPyArrayToRaster(
    output_array, 
    lower_left, 
    cell_size, 
    value_to_nodata=0
)

# Apply Chilean coordinate system (SIRGAS Chile UTM 19S)
arcpy.management.DefineProjection(final_risk_raster, arcpy.SpatialReference(31980))
final_risk_raster.save("Maule_CascadeRisk")
print("Successfully written PyTorch prediction to Geodatabase!")
```

---

## 🗺️ Step 5: Final Visualization in ArcGIS Pro
1. Return to the ArcGIS Pro map interface.
2. Drag and drop the `Maule_CascadeRisk` raster from your Geodatabase.
3. Change the Symbology to a **Classified** color ramp (Green to Red).
4. Overlay the Sernageomin Fault lines and IDE Minagri agricultural boundaries.
5. Create a Map Layout with a legend to export as a PDF.

You have now built an end-to-end Geo-Intelligence pipeline!
