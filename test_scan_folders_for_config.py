import os
import glob
import re

def scan_folders_for_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result = []

    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        required_files = ["steps.py", "processing.py", "__init__.py"]
        if all(os.path.isfile(os.path.join(folder_path, file)) for file in required_files):
            config_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.yaml') and 'config' in file.lower():
                        config_files.append(os.path.join(root, file))
            
            for config_file in config_files:
                relative_path = os.path.relpath(config_file, folder_path)
                result.append({
                    "folder": folder,
                    "config": relative_path
                })
    
    return result

# Example usage:
if __name__ == "__main__":
    config_list = scan_folders_for_config()
    print(config_list)