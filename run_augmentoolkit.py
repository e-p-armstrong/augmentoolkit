# Imports the function named by the config file, and runs it with the arguments inside the config specified

import importlib
import asyncio
import sys
import traceback
import yaml
import argparse
from pathlib import Path
import json

from resolve_path import resolve_path


def load_function_from_path(function_path):
    """Dynamically import a function from a module path string"""
    module_path, function_name = function_path.rsplit(".", 1)
    # Ensure module path uses dots for importlib
    module_import_path = module_path.replace("/", ".")
    module = importlib.import_module(module_import_path)
    return getattr(module, function_name)


def flatten_config(config, no_flatten_keys=None):
    """Flatten config while preserving specified nested structures"""
    flattened = {}
    no_flatten = set(no_flatten_keys or [])

    for key, value in config.items():
        if key in no_flatten:
            flattened[key] = value
            continue

        if isinstance(value, dict):
            nested_flat = flatten_config(value, no_flatten_keys)
            for nested_key, nested_value in nested_flat.items():
                if nested_key in flattened:
                    raise ValueError(f"Key conflict: '{nested_key}'")
                flattened[nested_key] = nested_value
        else:
            if key in flattened:
                raise ValueError(f"Key conflict: '{key}'")
            flattened[key] = value
    print("FLATTENED")
    print(flattened)
    return flattened


try:
    with open("super_config.yaml", "r") as f:
        super_config = yaml.safe_load(f)
    path_aliases = super_config.get(
        "path_aliases", {}
    )  # Get aliases, default to empty dict if not present

except FileNotFoundError:
    print(f"Error: Super config file not found at super_config.yaml")
    sys.exit(1)
# Load super config
except yaml.YAMLError as e:
    print(f"Error parsing super config file super_config.yaml: {e}")
    sys.exit(1)


def run_pipeline(node, config, override_fields={}):
    # Resolve node and config paths using aliases
    resolved_node_path = resolve_path(node, path_aliases)
    # print(f"DEBUG: Resolved node path: {resolved_node_path}")

    # Handle optional config key and resolve its path if present
    config_path_str = config
    resolved_config_path_str = (
        resolve_path(config_path_str, path_aliases) if config_path_str else None
    )
    # print(f"DEBUG: Resolved config path: {resolved_config_path_str}")

    resolved_config_path_str = (
        resolved_config_path_str
        if resolved_config_path_str and resolved_config_path_str.endswith(".yaml")
        else (
            resolved_config_path_str + ".yaml"
            if resolved_config_path_str
            else resolved_config_path_str
        )
    )
    # Load pipeline-specific config if a path is provided
    config = {}
    if resolved_config_path_str:
        # Resolve config path relative to the script's directory
        script_dir = Path(__file__).parent
        config_path = (script_dir / resolved_config_path_str).resolve()
        # print(f"DEBUG: Final resolved config path: {config_path}")
        try:
            with open(config_path, "r") as f:
                config = (
                    yaml.safe_load(f) or {}
                )  # Ensure config is at least an empty dict
        except FileNotFoundError:
            print(
                f"Warning: Config file not found at {config_path}. Proceeding without it."
            )
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            # Decide if you want to raise the error or continue
            # raise e
            print("Proceeding with empty configuration for this pipeline.")
    # print("DEBUG: CONFIG")
    # print(config)
    run_pipeline_config(
        config=config,
        resolved_node_path=resolved_node_path,
        override_fields=override_fields,
    )


def run_pipeline_config(
    config, resolved_node_path, override_fields={}
):  # the second half of run_pipeline, extracted so that it is easier to use in isolation as an api.
    # Merge pipeline config from super_config with loaded config file parameters
    # Parameters defined directly in the pipeline entry override those in the loaded config file.
    # Parameters passed in as overrides override those in either.

    # Flatten the nested configuration
    no_flatten_keys = config.get(
        "no_flatten", []
    )  # Get no_flatten from loaded/merged config
    flattened_config = flatten_config(config, no_flatten_keys=no_flatten_keys)
    flattened_config.update(override_fields)

    # Import the target function using the resolved node path
    try:
        function = load_function_from_path(resolved_node_path)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error loading function from node path '{resolved_node_path}': {e}")
        print(f"Skipping pipeline: resolved_node_path")  # Use name if available
        return  # Skip this pipeline if function cannot be loaded

    if asyncio.iscoroutinefunction(function):
        print(f"Running async pipeline: {resolved_node_path}")

        # print("DEBUG: Flattened config")
        # print(flattened_config)

        try:
            asyncio.run(function(**flattened_config))
        except Exception as e:
            print(f"Error running async pipeline {resolved_node_path}: {e}")
            traceback.print_exc()
            raise
            # Optionally re-raise or handle error reporting
    else:
        print(f"Running sync pipeline: {resolved_node_path}")
        try:
            function(**flattened_config)
        except Exception as e:
            print(f"Error running sync pipeline {resolved_node_path}: {e}")
            # Optionally re-raise or handle error reporting
            raise

    print(f"Completed pipeline: {resolved_node_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Augmentoolkit pipelines.")
    parser.add_argument(
        "--node",
        type=str,
        help="Path (potentially aliased) to the pipeline node function (e.g., 'pipelines/my_pipeline.run').",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path (potentially aliased) to the pipeline-specific configuration YAML file.",
    )
    parser.add_argument(
        "--override-json",
        type=str,
        help="JSON string of parameters to override pipeline config.",
    )

    args = parser.parse_args()

    # Always load path aliases
    # path_aliases are already loaded globally before main()

    override_params = {}
    if args.override_json:
        try:
            override_params = json.loads(args.override_json)
            if not isinstance(override_params, dict):
                print(
                    "Error: --override-json must be a valid JSON object (dictionary)."
                )
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing --override-json: {e}")
            sys.exit(1)

    # Decide execution path based on provided arguments
    if args.node:
        print(f"Running single pipeline specified via command line arguments:")
        print(f"  Node: {args.node}")
        print(f"  Config: {args.config}")
        print(f"  Overrides: {override_params}")
        run_pipeline(args.node, args.config, override_params)
    else:
        print("Running pipelines defined in super_config.yaml 'pipeline_order'.")
        # Run pipelines in order specified in super_config
        pipelines_to_run = super_config.get("pipeline_order", [])
        if not pipelines_to_run:
            print(
                "No pipelines specified in 'pipeline_order' and no specific pipeline provided via args. Exiting."
            )
            return

        for pipeline in pipelines_to_run:
            if not isinstance(pipeline, dict) or "node" not in pipeline:
                print(
                    f"Warning: Skipping invalid pipeline entry in super_config: {pipeline}. Must be a dictionary with a 'node' key."
                )
                continue
            # Merge super_config parameters with any CLI overrides (though CLI overrides usually imply single pipeline run)
            pipeline_params = pipeline.get("parameters", {})
            # Note: If running from super_config, CLI overrides are *not* typically used per-pipeline.
            # The logic here prioritizes the CLI override if BOTH --override-json and super_config parameters exist,
            # but this scenario is less common when running the whole sequence.
            # If you intended CLI overrides to *only* apply when --node is used, keep override_params empty here.
            # If CLI overrides should *globally* apply even to super_config runs, update pipeline_params:
            # pipeline_params.update(override_params) # Uncomment this if CLI overrides should apply globally

            print(f"Running pipeline from super_config:")
            print(f"  Node: {pipeline['node']}")
            print(f"  Config: {pipeline.get('config')}")  # Use .get for optional config
            print(f"  Parameters: {pipeline_params}")

            run_pipeline(
                pipeline["node"], pipeline.get("config"), pipeline_params
            )  # Use .get for config key


if __name__ == "__main__":
    main()
