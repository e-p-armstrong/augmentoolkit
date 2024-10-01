print("Augmentoolkit is starting to run! If this is your first time running this it might take a few moments to start due to imports and such.")
# Named "run_augmentoolkit" so that you can type "run" and tab instead of typing "augmentoolkit" and having it clash with the folder we import stuff from.
# Orchestrate and execute pipelines according to the super-config in order.
import subprocess
import os
import yaml
import sys

with open("super_config.yaml", "r") as f:
    super_config = yaml.safe_load(f)

def run_processing_script(folder_path, config_path, project_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["CONFIG_PATH"] = config_path
    env["FOLDER_PATH"] = folder_path 
    subprocess.run([sys.executable, "processing.py"], cwd=folder_path, env=env)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    folder_configs = super_config["pipeline_order"]

    for folder_config in folder_configs:
        folder_path = folder_config["folder"]
        config_path = folder_config["config"]
        run_processing_script(folder_path, config_path, project_root)

if __name__ == "__main__":
    main()
