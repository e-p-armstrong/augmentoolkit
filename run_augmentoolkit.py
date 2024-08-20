# Named "run_augmentoolkit" so that you can type "run" and tab instead of typing "augmentoolkit" and having it clash with the folder we import stuff from.
# Orchestrate and execute pipelines according to the super-config in order.

import os
import yaml

with open("super_config.yaml", "r") as f:
    super_config = yaml.safe_load(f)
    
for pipeline in super_config["pipelines"]:
    pipeline_folder_path = pipeline["pipeline_folder_path"]
    