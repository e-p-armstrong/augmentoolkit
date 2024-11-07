import yaml
import os
import argparse
from typing import Dict, Union, List
import glob

def load_yaml(file_path: str) -> Dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading {file_path}: {e}")
            return {}

def find_missing_keys(base_dict: Dict, compare_dict: Dict, parent_key: str = '') -> List[tuple]:
    """
    Recursively find missing keys between two dictionaries.
    Returns list of tuples containing (full_key_path, value_from_base).
    """
    missing_keys = []
    
    for key, value in base_dict.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        
        if key not in compare_dict:
            missing_keys.append((current_key, value))
        elif isinstance(value, dict) and isinstance(compare_dict[key], dict):
            missing_keys.extend(find_missing_keys(value, compare_dict[key], current_key))
            
    return missing_keys

def update_dict_with_key_path(target_dict: Dict, key_path: str, value: any) -> None:
    """Update a dictionary using a dot-notation key path."""
    keys = key_path.split('.')
    current = target_dict
    
    # Navigate to the correct nested level
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value

def update_yaml_file(target_file: str, missing_keys: List[tuple]) -> None:
    """Update a YAML file with missing keys."""
    if not missing_keys:
        return
    
    # Load existing content
    target_dict = load_yaml(target_file)
    
    # Add missing keys
    for key_path, value in missing_keys:
        update_dict_with_key_path(target_dict, key_path, value)
    
    # Write back to file
    with open(target_file, 'w') as f:
        yaml.dump(target_dict, f, sort_keys=False)

def process_yaml_files(base_file: str, target_path: str) -> None:
    """Process YAML files and update missing keys."""
    # Load base file
    base_dict = load_yaml(base_file)
    if not base_dict:
        print(f"Error: Could not load base file {base_file}")
        return

    # Handle target path (file or directory)
    if os.path.isfile(target_path):
        target_files = [target_path]
    else:
        # Recursively find all yaml files in directory
        target_files = glob.glob(os.path.join(target_path, '**/*.yaml'), recursive=True)
        target_files.extend(glob.glob(os.path.join(target_path, '**/*.yml'), recursive=True))

    # Process each target file
    for target_file in target_files:
        if target_file == base_file:
            continue
            
        print(f"\nProcessing: {target_file}")
        target_dict = load_yaml(target_file)
        
        if not target_dict:
            print(f"Error: Could not load target file {target_file}")
            continue
        
        missing_keys = find_missing_keys(base_dict, target_dict)
        
        if missing_keys:
            print(f"Found missing keys in {target_file}:")
            for key, value in missing_keys:
                print(f"  {key}: {value}")
            
            update_yaml_file(target_file, missing_keys)
            print(f"Updated {target_file} with missing keys")
        else:
            print(f"No missing keys found in {target_file}")

def main():
    parser = argparse.ArgumentParser(description='Update YAML files with missing keys from a base file')
    parser.add_argument('base_file', help='Path to the base YAML file')
    parser.add_argument('target_path', help='Path to target YAML file or directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_file):
        print(f"Error: Base file {args.base_file} does not exist")
        return
    
    if not os.path.exists(args.target_path):
        print(f"Error: Target path {args.target_path} does not exist")
        return
        
    process_yaml_files(args.base_file, args.target_path)

if __name__ == "__main__":
    main()