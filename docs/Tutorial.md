# Advanced Hazard Analytics Mapper: The Cascade Effect Tutorial
**Building a Multi-Task AI Engine for Flood, Agriculture, and Environmental Risk**

This master tutorial guides you through building the full "Cascade Effect" spatial pipeline. We will move beyond simple single-output models to build a system where extreme weather triggers a chain reaction: **Flooding (Disaster) → Soil Erosion (Agriculture) → Toxic Runoff (Environment)**.

We will execute this entirely within a Windows environment, bridging **ArcGIS Pro**, **`arcpy`**, and **PyTorch**.

---

## 🛠️ Phase 1: Environment & Workspace Setup (Windows)

To bridge the ESRI ecosystem with deep learning, we must set up a dedicated Python environment.

### 1. Install Deep Learning Libraries
Before cloning the environment, we must install the necessary deep learning frameworks into the base ArcGIS Pro environment. For newer versions of ArcGIS Pro (like 3.0+), the conda solver often struggles to resolve dependencies (causing `LibMambaUnsatisfiableError`).

The officially recommended and easiest method is to use ESRI's provided installer:
1. Go to the official ESRI GitHub page: **[Esri/deep-learning-frameworks](https://github.com/Esri/deep-learning-frameworks)**
2. Scroll down to the **Download** section and download the installer that matches your ArcGIS Pro version (e.g., ArcGIS Pro 3.5).
3. Close ArcGIS Pro and run the installer (`.msi`).

This will automatically install PyTorch, torchvision, and all necessary dependencies perfectly synced with `arcpy`, completely bypassing the need to use `conda install`!

### 2. Clone the ArcGIS Pro Environment
Now that the base environment has the deep learning tools installed, we can clone it to create our custom workspace.
1. Launch **ArcGIS Pro**. Create a new project called `HazardArcGis`.
2. Go to **Settings** > **Package Manager**.
3. The default `arcgispro-py3` environment is read-only. Click the "gear" icon, select **Clone**, and name it `hazard-env`.
4. Right-click your new `hazard-env` and **Activate** it.

### 3. Activating the Environment (Terminal & IDE)
ArcGIS Pro uses a specialized internal conda configuration. To use this environment outside of the ArcGIS Pro UI:

**In a Terminal:**
*   If you have Conda initialized in your PATH, simply run: `conda activate hazard-env`
*   Otherwise, open your Windows Start Menu -> ArcGIS -> **Python Command Prompt**. This terminal automatically starts with your active ArcGIS environment.

**In an IDE (e.g., VS Code):**
*   Press `Ctrl + Shift + P` -> **Python: Select Interpreter**.
*   Select `hazard-env` (Path is usually `C:\Users\<username>\AppData\Local\ESRI\conda\envs\hazard-env\python.exe`).

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

## ⚙️ Phase 3: Workspace Setup & Feature Engineering

Instead of running long, hardcoded Python scripts, we have modularized the spatial pipeline into tools that can be run directly from the ArcGIS Pro interface.

### 1. Workspace Initialization
**Script:** `scripts/01_workspace_setup.py`
This tool automates the creation of a dedicated Geodatabase (e.g., `Hazard_Outputs.gdb`) and sets it as your default workspace. This keeps your project organized and prevents intermediate deep learning files from bloating your main project.

### 2. Feature Engineering Pipeline
**Script:** `scripts/02_arcpy_feature_engineering.py`
This tool ingests your raw DEM and Sentinel-2 imagery and calculates the necessary spatial inputs for the AI: Topography (Slope, Aspect, Curvature), Hydrology (Flow Accumulation), and Vegetation Indices (NDVI, NDWI).

### 💡 Terminal Fallback (Optional)
If you prefer scripting over clicking buttons in the UI, all of the scripts in this project are hybrid. They will automatically read from your terminal arguments if no ArcGIS UI inputs are provided. 
For example, to run Workspace Setup from PowerShell:
`python scripts\01_workspace_setup.py "D:\Path\To\Folder" "HazardOutputs.gdb"`

---

## 🔄 Phase 3.5: Bridging Python and the ArcGIS Pro Interface

You can turn our Python scripts into visual buttons that anyone can click, even if they don't know Python!

### Adding Tools to your Project Toolbox
When you created your `HazardArcGis` project, ArcGIS automatically generated a default toolbox for you.
1. In ArcGIS Pro, open the **Catalog** pane.
2. Expand **Toolboxes** and find the default `HazardArcGis.atbx` file.
3. Right-click `HazardArcGis.atbx` -> **New** -> **Script**.

**Tool 1: Workspace Setup**
*   **General Tab:** Name it `WorkspaceSetup`
*   **Execution Tab:** Point to `scripts/01_workspace_setup.py`
*   **Parameters Tab:**
    1. Label: `Folder Location` | Data Type: `Folder` | Direction: `Input`
    2. Label: `Geodatabase Name` | Data Type: `String` | Direction: `Input`
    3. Label: `Output Workspace` | Data Type: `Workspace` | Direction: `Output` (Derived)

**Tool 2: Feature Engineering**
*   **General Tab:** Name it `FeatureEngineering`
*   **Execution Tab:** Point to `scripts/02_arcpy_feature_engineering.py`
*   **Parameters Tab:**
    1. Label: `Target Geodatabase` | Data Type: `Workspace`
    2. Label: `DEM Raster` | Data Type: `Raster Layer`
    3. Label: `Sentinel-2 Raster` | Data Type: `Raster Layer`
    4. Label: `Geological Fault Lines` | Data Type: `Feature Layer` (Optional)

---

## 🧠 Phase 4: The Multi-Head PyTorch Bridge

We have separated the PyTorch AI logic from the ArcGIS mapping logic to ensure maximum efficiency.

### 1. The PyTorch Inference Engine
**Script:** `scripts/03_pytorch_multitask_model.py`
This tool loads the generated rasters, stacks them into a tensor, and feeds them into our `MultiHeadHazardNet`. Instead of trying to write rasters directly, it saves a raw PyTorch tensor file (`.pt`) containing the three distinct predictions: Disaster, Agriculture, and Environment.

### 2. The GIS Mapping Loop
**Script:** `scripts/04_arcpy_mapping_loop.py`
This tool ingests the `.pt` tensor file from the AI, extracts the spatial reference coordinates from your base DEM, and converts the raw arrays back into georeferenced ArcGIS Rasters.

### Adding AI Tools to the Toolbox
Right-click your `HazardArcGis.atbx` -> **New** -> **Script** for each of these:

**Tool 3: PyTorch Inference**
*   **Execution Tab:** `scripts/03_pytorch_multitask_model.py`
*   **Parameters Tab:**
    1. `Workspace Geodatabase` (Data Type: Workspace)
    2. `Slope Raster` (Data Type: Raster Layer)
    3. `Flow Accumulation Raster` (Data Type: Raster Layer)
    4. `Distance to Faults Raster` (Data Type: Raster Layer)
    5. `NDVI Raster` (Data Type: Raster Layer)
    6. `NDWI Raster` (Data Type: Raster Layer)
    7. `Output Tensor File (.pt)` (Data Type: File, Direction: Output)

**Tool 4: Mapping Loop**
*   **Execution Tab:** `scripts/04_arcpy_mapping_loop.py`
*   **Parameters Tab:**
    1. `Input Tensor File (.pt)` (Data Type: File, Direction: Input)
    2. `Reference Raster (e.g., DEM)` (Data Type: Raster Layer)
    3. `Output Geodatabase` (Data Type: Workspace)
    4. `Disaster Risk Name` (Data Type: String)
    5. `Crop Risk Name` (Data Type: String)
    6. `Toxicity Risk Name` (Data Type: String)
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
