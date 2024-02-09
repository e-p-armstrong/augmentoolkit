import os
import json

# Replace 'your_directory_path' with the path to your directory containing the text files
directory_path = '.'

# JSON structure to prepend
json_structure = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""},
    {"role": "assistant", "content": ""}
]

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        # Construct the path to the current file
        file_path = os.path.join(directory_path, filename)
        # Construct the new filename with .json extension
        new_filename = os.path.splitext(filename)[0] + '.json'
        new_file_path = os.path.join(directory_path, new_filename)
        
        # Read the contents of the original text file
        with open(file_path, 'r', encoding='utf-8') as file:
            original_content = file.read()
            
            # Instead of escaping, we directly assign the content
            json_structure.append({"original_content": original_content})
            
            # Write the JSON structure to the new file
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                json.dump(json_structure, new_file, indent=2, ensure_ascii=False)

        # Reset json_structure for the next file by removing the last element (original content)
        json_structure.pop()

print("Conversion completed.")
