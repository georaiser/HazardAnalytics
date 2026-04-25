import arcpy
import os
import sys

def setup_workspace():
    """
    Creates a new File Geodatabase and sets it as the default workspace.
    Supports both ArcGIS Pro Script Tool inputs and Terminal inputs.
    """
    # Fetch from ArcGIS Pro UI
    folder_path = arcpy.GetParameterAsText(0)
    gdb_name = arcpy.GetParameterAsText(1)
    
    # Fallback to Terminal Arguments if UI inputs are empty
    if not folder_path and len(sys.argv) > 1:
        folder_path = sys.argv[1]
    if not gdb_name and len(sys.argv) > 2:
        gdb_name = sys.argv[2]
        
    # Helper functions to print to both Terminal and ArcGIS Pro UI
    def log(msg):
        print(msg)
        arcpy.AddMessage(msg)
        
    def log_err(msg):
        print(f"ERROR: {msg}")
        arcpy.AddError(msg)

    log("Initializing Workspace Setup...")
    
    if not folder_path or not gdb_name:
        log_err("Folder path and Geodatabase name are required.")
        print("Usage in terminal: python 01_workspace_setup.py \"D:\\Path\\To\\Folder\" \"ProjectName.gdb\"")
        return
        
    if not gdb_name.endswith(".gdb"):
        gdb_name += ".gdb"
        
    gdb_path = os.path.join(folder_path, gdb_name)
    
    if not arcpy.Exists(gdb_path):
        log(f"Creating Geodatabase: {gdb_path}")
        arcpy.management.CreateFileGDB(folder_path, gdb_name)
    else:
        log(f"Geodatabase already exists at {gdb_path}. Using existing GDB.")
        
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True
    
    # This only affects the UI, will be ignored by terminal
    arcpy.SetParameterAsText(2, gdb_path)
    
    log("Workspace Setup Complete.")

if __name__ == '__main__':
    setup_workspace()
