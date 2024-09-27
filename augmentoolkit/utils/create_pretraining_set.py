import chardet

import json
import os

def create_pretraining_set(content_list, json_file):
    # Initialize a variable to store the combined text of all files
    # Walk through all directories and files in the directory
    # remove previous pretraining set if it exists
    if os.path.exists(json_file):
        os.remove(json_file)
    for file_contents in content_list:
            with open(json_file, "a", encoding='utf-8', errors='ignore') as file:
                data = {"text": file_contents}
                write = json.dumps(data, ensure_ascii=False)
                file.write(write + "\n")

    # Create a dictionary with the combined text

    