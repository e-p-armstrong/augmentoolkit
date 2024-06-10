import os
import json
import yaml
def json_to_yaml(json_dir):
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            yaml_path = os.path.join(json_dir, filename[:-5] + ".yaml")
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            yaml_content = []
            for item in json_data:
                yaml_content.append(f"- role: {item['role']}")
                yaml_content.append(f"  content: |")
                content = item['content'].replace('\\n', '\n')
                content = content.replace('\\"', '"')
                for line in content.split('\n'):
                    yaml_content.append(f"    {line}")

            with open(yaml_path, "w") as yaml_file:
                yaml_file.write('\n'.join(yaml_content))

# Example usage
json_directory = "./prompts"
json_to_yaml(json_directory)