# Advanced Hazard Analytics Mapper: The Cascade Effect Tutorial
**Building a Multi-Task AI Engine for Flood, Agriculture, and Environmental Risk**

This master tutorial guides you through building the full "Cascade Effect" spatial pipeline. We will move beyond simple single-output models to build a system where extreme weather triggers a chain reaction: **Flooding (Disaster) → Soil Erosion (Agriculture) → Toxic Runoff (Environment)**.

We will execute this entirely within a Windows environment, bridging **ArcGIS Pro**, **`arcpy`**, and **PyTorch**.

---

## 🛠️ Phase 1: Environment & Workspace Setup (Windows)

To bridge the ESRI ecosystem with deep learning, we must set up a dedicated Python environment.

### 1. Clone the ArcGIS Pro Environment
1. Launch **ArcGIS Pro**. Create a new project called `Cascade Analytics Engine`.
2. Go to **Settings** > **Package Manager**.
3. The default `arcgispro-py3` environment is read-only. Click the "gear" icon, select **Clone**, and name it `cascade-env`.
4. Right-click your new `cascade-env` and **Activate** it.

### 2. Install Deep Learning Libraries
For newer versions of ArcGIS Pro (like 3.0+), the conda solver often struggles to resolve dependencies (causing `LibMambaUnsatisfiableError`). 

The officially recommended and easiest method is to use ESRI's provided installer:
1. Go to the official ESRI GitHub page: **[Esri/deep-learning-frameworks](https://github.com/Esri/deep-learning-frameworks)**
2. Scroll down to the **Download** section and download the installer that matches your ArcGIS Pro version (e.g., ArcGIS Pro 3.5).
3. Close ArcGIS Pro and run the installer (`.msi`). 

This will automatically install PyTorch, torchvision, and all necessary dependencies perfectly synced with `arcpy`, completely bypassing the need to use `conda install`!

---

## 🗺️ Phase 2: Open Data Acquisition (Chile)

We will focus on the **Maule Region (Región del Maule)** or **Biobío Region**. You need to gather data across three domains.

### 1. Disaster Domain (Topography & Geology)
*   **DEM (Elevation):** Go to the [ASF Vertex Portal](https://search.asf.alaska.edu/). Search your target area, filter for **ALOS PALSAR**, and download the `Hi-Res Terrain Corrected (12.5m)` GeoTIFF.
*   **Fault Lines:** Go to **Sernageomin**. Download the national geological fault lines Shapefile.
*   **Historical Floods (Optional):** Download Sentinel-1 (Radar) GRD products from Copernicus to identify past flood extents for training labels.

### 2. Agriculture Domain (Vegetation Health)
*   **Optical Imagery:** Go to the [Copernicus Browser](https://browser.dataspace.copernicus.eu). Download a cloud-free **Sentinel-2 Level-2A** image (summer months like Jan/Feb) for your region. Level-2A is atmospherically corrected.
*   **Crop Boundaries:** Go to **IDE Minagri** (ide.minagri.gob.cl) and download the agricultural land use/cadastre Shapefiles.

### 3. Environment Domain (Protected Areas)
*   **Water Bodies:** Go to the **DGA (Dirección General de Aguas)** or IDE Chile portals to download the official shapefiles for rivers, wetlands, and protected natural reserves.

---

## ⚙️ Phase 3: The `arcpy` Feature Engineering Pipeline

Create a new Python script (`scripts/02_arcpy_feature_engineering.py`) or a Jupyter Notebook within ArcGIS Pro. We will automate the calculation of spatial indices and hydrological routing.

```python
import arcpy
from arcpy.sa import *
from arcpy.ia import *
import os

# 1. Setup Workspace and Projections
arcpy.env.workspace = r"C:\Path\To\CascadeProject.gdb"
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("Image")

dem_path = r"C:\RawData\Maule_DEM.tif"
sentinel_path = r"C:\RawData\Sentinel2_L2A.tif"

print("1. Calculating Topography...")
slope_raster = Slope(dem_path, "DEGREE")
slope_raster.save("Cascade_Slope")

print("2. Calculating Hydrology & Routing...")
filled_dem = Fill(dem_path)
flow_dir = FlowDirection(filled_dem)
flow_acc = FlowAccumulation(flow_dir)
flow_acc.save("Cascade_FlowAcc")

# Calculate distance to geological faults
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

print("Spatial feature engineering complete!")
```

---

## 🔄 Phase 3.5: Bridging Python (`arcpy`) and the ArcGIS Pro Interface

A key part of learning ArcGIS Pro is understanding that **Python (`arcpy`) and the visual user interface are fully connected**. Anything you can click in the UI, you can write in Python, and vice versa!

Here is how you transition between the two:

### 1. From UI to Python ("Copy Python Command")
If you don't know how to write the code for a specific tool:
1. Open ArcGIS Pro and go to the **Analysis** tab -> **Tools** to open the Geoprocessing pane.
2. Find the tool you want (e.g., *Focal Statistics*), fill out the parameters visually, and click **Run**.
3. Go to the **View** tab -> **Geoprocessing History**.
4. Right-click the tool you just ran and select **Copy Python Command**. 
5. Paste it right into your script! This is the best way to learn `arcpy` syntax.

### 2. From Python to UI (Creating a Custom Toolbox)
You can turn our `02_arcpy_feature_engineering.py` script into a visual button that anyone can click, even if they don't know Python:
1. In ArcGIS Pro, open the **Catalog** pane.
2. Right-click **Toolboxes** -> **New Toolbox** (name it `Cascade_Tools.atbx`).
3. Right-click your new toolbox -> **New** -> **Script**.
4. In the **General** tab, name it "FeatureEngineering".
5. In the **Execution** tab, point it to your `02_arcpy_feature_engineering.py` file.
6. In the **Parameters** tab, you can define inputs (like asking the user to select the `dem_path` visually instead of hard-coding it in the script).
7. Now, when you double-click your new tool in ArcGIS Pro, it opens a standard visual window, but runs your Python code in the background!

---

## 🧠 Phase 4: The Multi-Head PyTorch Bridge

This is the core innovation. We convert our ArcGIS Rasters into a multi-dimensional NumPy array, feed it into a PyTorch model with three separate output heads, and map the outputs back.

Create a new script: `scripts/03_pytorch_multitask_model.py`

```python
import arcpy
import numpy as np
import torch
import torch.nn as nn

print("1. Bridging ArcGIS to PyTorch Tensors...")
# Extract spatial reference data from base raster
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
```

---

## 🎨 Phase 5: Final Geoprocessing & Visualization

Now that PyTorch has returned three distinct risk prediction rasters, we complete the pipeline using the ArcGIS Pro UI.

1. **Add to Map:** Drag `Pred_DisasterRisk`, `Pred_CropRisk`, and `Pred_ToxicityRisk` from your `.gdb` into the main map view.
2. **Apply Symbology:** 
   * Open the Symbology pane for each layer.
   * Apply a **Classified** color ramp (e.g., green to dark red for high risk).
   * Utilize the **Swipe Tool** (under the Raster Layer tab) to visually correlate how areas of high flood risk cascade into high crop risk.
3. **Overlay Ground Truths:**
   * Overlay the **IDE Minagri Crop Boundaries** on the `Pred_CropRisk` raster to identify exactly which farms are in danger.
   * Overlay the **DGA River Networks** on the `Pred_ToxicityRisk` raster.
4. **Agentic/Automated Reporting:** 
   * As an optional final step, use `arcpy.sa.ZonalStatisticsAsTable` to calculate the exact hectares of farms at high risk, exporting a CSV that an LLM agent could ingest to write an automated emergency briefing.
