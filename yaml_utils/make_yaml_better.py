import os
import yaml

def convert_escaped_newlines(yaml_content):
    # Load the original YAML content
    data = yaml.safe_load(yaml_content)

    # Iterate over each item, replacing escaped newlines in strings
    for item in data:
        if 'content' in item:
            item['content'] = item['content'].replace('\\n', '\n')

    # Convert back to YAML format with correct newlines
    return yaml.dump(data, allow_unicode=True, width=1000)

def process_directory(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            # Construct the full path of the file
            filepath = os.path.join(directory, filename)
            # Read the original YAML file
            with open(filepath, 'r', encoding='utf-8') as file:
                yaml_content = file.read()

            # Convert the YAML content
            converted_yaml = convert_escaped_newlines(yaml_content)

            # Save the converted YAML to a new file
            new_filename = f"{os.path.splitext(filename)[0]}.yaml"
            new_filepath = os.path.join(directory, new_filename)
            with open(new_filepath, 'w', encoding='utf-8') as new_file:
                new_file.write(converted_yaml)
            print(f"Converted file saved as: {new_filepath}")


# Specify the directory to process
directory_path = '../prompts'
process_directory(directory_path)