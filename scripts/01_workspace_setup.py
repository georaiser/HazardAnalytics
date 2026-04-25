import arcpy
import os

def setup_workspace():
    """
    Creates a new File Geodatabase and sets it as the default workspace.
    Designed to be run as an ArcGIS Pro Script Tool.
    
    Parameters:
    0: Folder Location (Data Type: Folder)
    1: Geodatabase Name (Data Type: String)
    2: Output Workspace (Data Type: Workspace, Direction: Output)
    """
    arcpy.AddMessage("Initializing Workspace Setup...")
    
    folder_path = arcpy.GetParameterAsText(0)
    gdb_name = arcpy.GetParameterAsText(1)
    
    if not folder_path or not gdb_name:
        arcpy.AddError("Folder path and Geodatabase name are required.")
        return
        
    if not gdb_name.endswith(".gdb"):
        gdb_name += ".gdb"
        
    gdb_path = os.path.join(folder_path, gdb_name)
    
    if not arcpy.Exists(gdb_path):
        arcpy.AddMessage(f"Creating Geodatabase: {gdb_path}")
        arcpy.management.CreateFileGDB(folder_path, gdb_name)
    else:
        arcpy.AddWarning(f"Geodatabase already exists at {gdb_path}. Using existing GDB.")
        
    arcpy.env.workspace = gdb_path
    arcpy.env.overwriteOutput = True
    
    arcpy.SetParameterAsText(2, gdb_path)
    arcpy.AddMessage("Workspace Setup Complete.")

if __name__ == '__main__':
    setup_workspace()
