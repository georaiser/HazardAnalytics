# Hazard Analytics Mapper: The Cascade Effect

An integrated Geo-Intelligence Engine predicting a **Cascade Impact Scenario** in Chile. 
This project bridges the gap between **ArcGIS Pro** (Spatial Data Management), **`arcpy`** (Automated Geoprocessing), and **PyTorch** (Multi-Task Deep Learning).

## The "Cascade Effect" Architecture

In the context of this project, **"Cascade"** refers to a **chain reaction of disasters**, where one event directly triggers the next, crossing into entirely different domains.

Instead of predicting a single isolated event (like "where will it flood?"), this system models how that initial hazard *cascades* into secondary and tertiary crises:

1. **🌊 Disaster Domain (The Trigger):** Extreme weather and steep topography cause **Flooding and Landslides**.
2. **🌾 Agriculture Domain (The First Cascade):** The violent flooding destroys farmland, leading to severe topsoil erosion and **Crop Damage**.
3. **☣️ Environment Domain (The Second Cascade):** Because the destroyed farmland was treated with pesticides and fertilizers, that eroded soil becomes **Toxic Runoff** that washes into protected rivers, wetlands, and the water supply.

To achieve this computationally, the project utilizes a **Multi-Head PyTorch CNN**. Instead of building three separate AI models, we have *one* neural network that processes shared spatial features (topography, hydrology, vegetation indices) and outputs three predictions simultaneously to capture this interconnected chain of events.

## Project Structure

```text
HazardAnalyticsMapper/
│
├── data/                  # Local Geodatabases and rasters
│   ├── raw/               # Downloaded DEMs, Sentinel imagery, Shapefiles
│   └── processed/         # arcpy generated indices and intermediate arrays
│
├── docs/                  # Documentation and tutorials
│   └── Tutorial.md        # The complete step-by-step master guide
│
├── models/                # PyTorch saved models and weights
│   └── cascade_net.pth    # Multi-head CNN model file (generated after training)
│
├── scripts/               # Core Python automation scripts
│   ├── 01_workspace_setup.py         # Initializes the Geodatabase
│   ├── 02_arcpy_feature_engineering.py # Generates slope, flow, NDVI, etc.
│   ├── 03_pytorch_multitask_model.py   # Multi-Head CNN inference script
│   └── 04_arcpy_mapping_loop.py        # Converts PyTorch tensors back to Raster
│
├── brain/                 # Ideation, architecture proposals, and AI conversation history
└── README.md              # Project overview
```

## Setup & Prerequisites
To run this project, you must have:
- **Windows OS**
- **ArcGIS Pro 3.x** installed with Spatial Analyst and Image Analyst extensions.
- A cloned ArcGIS Pro Python environment with `deep-learning-essentials` installed.

## Getting Started
Please read the complete tutorial located at `docs/Tutorial.md` for a step-by-step guide on downloading the required open data for Chile (Maule Region) and running the `arcpy` to PyTorch pipeline.
