import os
import glob

def get_most_recent_file(folder_path):
    # Get list of all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    
    if not files:
        return None
    
    # Find the file with the most recent modification time
    most_recent_file = max(files, key=os.path.getmtime)
    
    # Return just the filename (not the full path)
    return os.path.basename(most_recent_file)