# tasks.py
import os
import sys
import time
import random
import logging
import subprocess  # Import subprocess module
import json  # For parameter serialization
import signal  # To check exit codes against signals
from typing import Optional, Dict, Any, TYPE_CHECKING, IO, Union
import traceback
import io  # For type hinting file handles
import yaml  # For loading config files
from pathlib import Path  # For path manipulation

from huey_config import huey
import json
from redis_config import redis_client, set_progress
from resolve_path import resolve_path
from run_augmentoolkit import flatten_config

if TYPE_CHECKING:
    from huey.api import Task

# Assuming run_augmentoolkit.py can be called directly
# You might need to adjust the path or how it's called
# Ensure run_augmentoolkit.py uses argparse or similar to accept these args
PIPELINE_RUNNER_SCRIPT = "run_augmentoolkit.py"
ATK3_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ATK3_DIRECTORY, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)  # Ensure logs directory exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PID_KEY_TIMEOUT = 24 * 60 * 60  # 24 hours
# Define a separate timeout for the output dir mapping, potentially longer
OUTPUT_DIR_MAPPING_TIMEOUT = PID_KEY_TIMEOUT * 7  # Keep mapping for 7 days
# Define a timeout for parameters, same as output dir seems reasonable
PARAMETERS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT
# Define timeout for the final status key, make it long like the others
FINAL_STATUS_TIMEOUT = OUTPUT_DIR_MAPPING_TIMEOUT


# --- Helper Function for Finding Nested Key ---
def find_first_output_dir(
    data: Union[Dict, list], key_name: str = "output_dir"
) -> Optional[str]:
    """Recursively searches dict/list for the first value associated with key_name."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_name:
                if isinstance(value, str):
                    return value
                else:
                    # Found the key, but the value is not a string, log warning?
                    print(
                        f"Found key '{key_name}' but its value is not a string: {type(value)}. Skipping."
                    )
                    # Continue searching in case there's another valid one deeper
            elif isinstance(value, (dict, list)):
                found = find_first_output_dir(value, key_name)
                if found is not None:
                    return found  # Return the first one found in recursion
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                found = find_first_output_dir(item, key_name)
                if found is not None:
                    return found  # Return the first one found in recursion
    return None  # Key not found


# --- Helper Function for Setting Final Status in Redis ---
def set_final_status(
    task_id: str, status: str, message: str, details: Optional[Dict] = None
):
    """Sets the final status of a task in Redis."""
    redis_key = f"status_for_task:{task_id}"
    status_data = {
        "status": status,
        "message": message,
        "details": details or {},
        "timestamp": time.time(),  # Add timestamp for potential debugging
    }
    try:
        redis_client.set(redis_key, json.dumps(status_data), ex=FINAL_STATUS_TIMEOUT)
        print(
            f"Task {task_id}: Set final status in Redis ({redis_key}) to {status}. Message: {message}"
        )
    except Exception as e:
        print(
            f"Task {task_id}: Failed to set final status '{status}' in Redis ({redis_key}): {e}",
            exc_info=True,
        )


@huey.task(context=True)
def run_pipeline_task(
    task: "Task",
    node_path: str,
    config_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
):
    """
    Executes a pipeline via a subprocess, stores subprocess PID, parameters,
    and waits for completion, capturing combined stdout/stderr to a log file.
    Sets the final status (COMPLETED, FAILED, REVOKED) in Redis.
    """
    task_id = str(task.id)
    redis_pid_key = f"worker_pid_for_task:{task_id}"  # Key now stores *subprocess* PID
    redis_output_dir_key = (
        f"output_dir_for_task:{task_id}"  # Key for output dir mapping
    )
    redis_params_key = f"parameters_for_task:{task_id}"  # Key for parameters
    redis_status_key = f"status_for_task:{task_id}"  # Key for final status
    process = None  # Initialize process variable
    log_file: Optional[io.TextIOWrapper] = None  # Initialize log file handle
    log_file_path = ""  # Store path for logging
    final_status_set = False  # Flag to ensure final status is set only once

    print(
        f"Task {task_id}: Preparing to run pipeline subprocess for node: {node_path}"
    )

    # Initialize progress early to indicate the task is starting
    try:
        set_progress(task_id, 0.0, "Initializing task...")
        print(f"Task {task_id}: Initial progress set to 0.0.")
    except Exception as e:
        print(f"Task {task_id}: Failed to set initial progress: {e}")

    if parameters is None:
        parameters = {}
    
    print(f"Task {task_id}: Original parameters (first 500 chars): {str(parameters)[:500]}")

    # first, flatten parameters
    no_flatten_keys = parameters.get("no_flatten", [])
    print(f"Task {task_id}: About to call flatten_config. No_flatten keys: {no_flatten_keys}")
    try:
        parameters_flat = flatten_config(parameters, no_flatten_keys=no_flatten_keys)
        print(f"Task {task_id}: Successfully called flatten_config. Flat params (first 500 chars): {str(parameters_flat)[:500]}")
    except Exception as fc_e:
        print(f"Task {task_id}: Error during flatten_config: {fc_e}", exc_info=True)
        # Set final status to FAILED
        set_final_status(
            task_id,
            "FAILED",
            f"Task failed during parameter flattening: {fc_e}",
            details={"error": str(fc_e), "traceback": traceback.format_exc()},
        )
        final_status_set = True # Mark as set
        raise # Re-raise to be caught by the main handler

    if "task_id" not in parameters_flat:
        parameters_flat["task_id"] = task_id
        print(f"Task {task_id}: task_id added to parameters_flat.")

    try:
        # --- Determine and Store Output Directory ---
        print(f"Task {task_id}: Determining and storing output directory...")
        output_dir_value = None
        resolved_output_dir = None

        with open("super_config.yaml", "r") as f:
            super_config = yaml.safe_load(f)
        path_aliases = super_config.get("path_aliases", {})

        # 1. Check parameters override
        if "output_dir" in parameters_flat:
            output_dir_value = parameters_flat["output_dir"]
            print(
                f"Task {task_id}: Found 'output_dir' in parameters override: {output_dir_value}"
            )
        # 2. Check config file if not in parameters
        elif config_path:
            try:
                # Resolve config_path relative to ATK3_DIRECTORY
                resolved_config_from_alias = resolve_path(config_path, path_aliases)
                abs_config_path = Path(ATK3_DIRECTORY) / resolved_config_from_alias
                if not abs_config_path.is_file():
                    print(
                        f"Task {task_id}: Config path '{config_path}' (resolved to {abs_config_path}) does not exist or is not a file."
                    )
                else:
                    print(
                        f"Task {task_id}: Loading config file: {abs_config_path}"
                    )
                    with open(abs_config_path, "r", encoding="utf-8") as f:
                        config_data = yaml.safe_load(f)
                    if config_data:
                        output_dir_value = find_first_output_dir(config_data)
                        if output_dir_value:
                            print(
                                f"Task {task_id}: Found 'output_dir' in config file '{config_path}': {output_dir_value}"
                            )
                        else:
                            print(
                                f"Task {task_id}: Key 'output_dir' not found in config file '{config_path}'."
                            )
                    else:
                        print(
                            f"Task {task_id}: Config file '{config_path}' loaded as empty or invalid."
                        )
            except yaml.YAMLError as e:
                print(
                    f"Task {task_id}: Error parsing YAML config file '{config_path}': {e}"
                )
            except Exception as e:
                print(
                    f"Task {task_id}: Error reading or processing config file '{config_path}': {e}"
                )

        # 3. Resolve the path and handle fallback
        if output_dir_value and isinstance(output_dir_value, str):
            output_dir_path = Path(output_dir_value)
            if output_dir_path.is_absolute():
                resolved_output_dir = output_dir_path
            else:
                # Resolve relative to ATK3_DIRECTORY
                resolved_output_dir = (
                    Path(ATK3_DIRECTORY) / output_dir_value
                ).resolve()
            print(
                f"Task {task_id}: Resolved output directory to: {resolved_output_dir}"
            )
        else:
            resolved_output_dir = "NO OUTPUT DIR DEFINED"
            # raise Exception("No output dir! This ought not to have happened. Config or parameters are likely buggered.")
            # If still no output dir, use a default one based on task_id in outputs
            # print(f"Task {task_id}: No 'output_dir' found in parameters or config. Defaulting to './outputs/{task_id}'")
            # resolved_output_dir = (Path(ATK3_DIRECTORY) / "outputs" / task_id).resolve()
            # # Ensure the default output dir exists before the task runs
            # try:
            #     resolved_output_dir.mkdir(parents=True, exist_ok=True)
            #     print(f"Task {task_id}: Created default output directory: {resolved_output_dir}")
            #     # Add this default path back to parameters_flat so the script knows about it
            #     parameters_flat["output_dir"] = str(resolved_output_dir)
            # except Exception as mkdir_e:
            #     print(f"Task {task_id}: Failed to create default output directory {resolved_output_dir}: {mkdir_e}")
            #     raise Exception(f"Failed to create default output directory: {mkdir_e}")
        print(f"Task {task_id}: Finished determining output directory. Resolved to: {resolved_output_dir}")

        # 4. Store in Redis
        if resolved_output_dir:
            try:
                redis_client.set(
                    redis_output_dir_key,
                    str(resolved_output_dir),
                    ex=OUTPUT_DIR_MAPPING_TIMEOUT,
                )
                print(
                    f"Task {task_id}: Stored output directory mapping in Redis key {redis_output_dir_key}"
                )
            except Exception as e:
                print(
                    f"Task {task_id}: Failed to store output directory mapping in Redis: {e}"
                )
        else:
            # Should not happen if fallback logic is correct, but log defensively
            print(
                f"Task {task_id}: Could not determine a resolved output directory to store in Redis."
            )

        # --- Store Parameters in Redis ---
        print(f"Task {task_id}: Storing parameters in Redis...")
        try:
            # Use the flattened 'parameters' dictionary
            params_json = json.dumps(parameters_flat)
            redis_client.set(redis_params_key, params_json, ex=PARAMETERS_TIMEOUT)
            print(
                f"Task {task_id}: Stored parameters in Redis key {redis_params_key}"
            )
        except TypeError as json_e:
            print(
                f"Task {task_id}: Failed to serialize parameters to JSON: {json_e}. Parameters will not be stored in Redis.",
                exc_info=True,
            )
            # Consider if this should be a fatal error or just a warning
        except Exception as redis_e:
            print(
                f"Task {task_id}: Failed to store parameters in Redis key {redis_params_key}: {redis_e}",
                exc_info=True,
            )
            # Consider if this should be a fatal error or just a warning

        # --- Prepare Subprocess Command ---
        print(f"Task {task_id}: Preparing subprocess command...")
        command = [ # This is actually ideal. We get the same code as with the cli, with the API of the API mode, and the task queue management of huey. Everything fits. All we need to do is modify run_augmentoolkit to take those args. if provided and not rely on super_config only.
            sys.executable,  # Use the current Python interpreter
            PIPELINE_RUNNER_SCRIPT,
            "--node",
            node_path,
            # Add other necessary arguments for your script
        ]
        if config_path:
            command.extend(["--config", config_path])
            print(f"Task {task_id}: Added config_path '{config_path}' to command.")

        if parameters_flat:
            print(f"Task {task_id}: About to call json.dumps for parameters_flat in command extend.")
            try:
                params_json_for_command = json.dumps(parameters_flat)
                command.extend(["--override-json", params_json_for_command])
                print(f"Task {task_id}: Added override-json to command.")
            except Exception as jd_e:
                print(f"Task {task_id}: Error during json.dumps for command parameters: {jd_e}", exc_info=True)
                set_final_status(
                    task_id,
                    "FAILED",
                    f"Task failed during command parameter JSON serialization: {jd_e}",
                    details={"error": str(jd_e), "traceback": traceback.format_exc()},
                )
                final_status_set = True # Mark as set
                raise

        print(
            f"Task {task_id}: Executing command: {' '.join(command)} in CWD: {ATK3_DIRECTORY}"
        )

        # --- Open Log File ---
        log_file_path = os.path.join(LOGS_DIR, f"{task_id}.log")
        print(
            f"Task {task_id}: Determined log file path: {log_file_path}. About to open."
        )
        # Open in text mode with utf-8 encoding, handle potential errors
        try:
            log_file = open(log_file_path, "w", encoding="utf-8")
            print(f"Task {task_id}: Successfully opened log file: {log_file_path}")
        except IOError as open_e:
            print(
                f"Task {task_id}: Failed to open log file {log_file_path}: {open_e}. Output will not be saved to file."
            )
            log_file = None  # Ensure log_file is None if opening failed

        # --- Start Subprocess ---
        print(f"Task {task_id}: Preparing to start subprocess. Log file object is: {log_file}")
        # If log file opened successfully, pipe both stdout and stderr to it
        # Otherwise, let output go to devnull to avoid blocking
        if log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=log_file,  # Both stdout and stderr go to the same file
                cwd=ATK3_DIRECTORY,  # Set working directory if script needs it
                # Note: when passing file objects, subprocess handles them in text mode based on the file's mode
            )
        else:
            # Fallback: send output to devnull if log file couldn't be opened
            print(f"Task {task_id}: Log file was not opened. Subprocess output will go to DEVNULL.")
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=ATK3_DIRECTORY,
            )
        print(f"Task {task_id}: Subprocess Popen call completed. Process object: {process}")

        # --- Store Subprocess PID ---
        # Add a check here:
        if process is None:
            print(f"Task {task_id}: Subprocess.Popen appears to have failed, process object is None.")
            # This state should ideally not be reached if Popen raises an error,
            # but as a safeguard:
            if not final_status_set:
                 set_final_status(
                    task_id,
                    "FAILED",
                    "Task failed: Subprocess.Popen resulted in a None process object.",
                    details={"error": "Popen returned None"},
                )
                 final_status_set = True
            raise RuntimeError("Subprocess.Popen failed, process object is None.")
        
        subprocess_pid = process.pid
        redis_client.set(redis_pid_key, subprocess_pid, ex=PID_KEY_TIMEOUT)
        print(
            f"Task {task_id}: Subprocess started (PID: {subprocess_pid}). Stored PID in Redis key {redis_pid_key}"
        )

        # --- Wait for Subprocess Completion ---
        print(
            f"Task {task_id}: Waiting for subprocess (PID: {subprocess_pid}) to complete..."
        )
        exit_code = process.wait()  # This blocks until the subprocess finishes

        print(
            f"Task {task_id}: Subprocess (PID: {subprocess_pid}) finished with exit code {exit_code}."
        )

        # --- Determine Task Status from Exit Code and Set Final Status ---
        if exit_code == 0:
            print(
                f"Pipeline task {task_id} completed successfully via subprocess."
            )
            set_final_status(
                task_id, "COMPLETED", f"Pipeline task {task_id} completed successfully."
            )
            final_status_set = True
            # Return success status for Huey's internal tracking (though we primarily rely on Redis now)
            return {
                "status": "success",
                "message": f"Pipeline {node_path} completed successfully.",
            }
        # Check if the exit code corresponds to SIGINT (common interrupt signal)
        # On Unix, exit code for signal N is often -N or 128+N. Check your OS specifics if needed.
        # Python subprocess killed by SIGINT usually results in exit code -2 (-signal.SIGINT)
        else:
            # Double-check if the status was already set to REVOKED by the API
            redis_status_key = f"status_for_task:{task_id}"
            final_status_json = redis_client.get(redis_status_key)
            if final_status_json:
                try:
                    status_data = json.loads(final_status_json)
                    if status_data.get("status") == "REVOKED":
                        print(
                            f"Task {task_id}: Subprocess exited with code {exit_code}, but status was already set to REVOKED by API. Ignoring exit code."
                        )
                        # Do not set status to FAILED if it was already REVOKED
                        # We still need to raise an exception for Huey if the exit code was non-zero,
                        # unless the only reason it was non-zero was the interrupt itself.
                        # However, differentiating the cause of non-zero exit code here is complex.
                        # Simplest approach: if API set REVOKED, trust that and don't raise internal Huey error.
                        # This means Huey might not internally record the task as failed/revoked in this edge case,
                        # but our Redis status (the source of truth for the API) is correct.
                        return  # Exit gracefully without setting FAILED or raising error
                except Exception as e:
                    print(
                        f"Task {task_id}: Error checking existing Redis status before setting FAILED: {e}"
                    )

            # If status was not REVOKED, proceed with FAILED logic
            error_message = f"Task {task_id}: Subprocess (PID: {subprocess_pid}) failed with non-zero exit code {exit_code}."
            print(error_message)
            # Avoid overwriting if status was somehow set between the check above and here
            if not final_status_set:
                set_final_status(
                    task_id,
                    "FAILED",
                    f"Pipeline subprocess failed with exit code {exit_code}.",
                    details={"exit_code": exit_code},
                )
                final_status_set = True
            # Raise an exception so Huey marks the task as failed internally
            raise RuntimeError(f"Pipeline subprocess failed with exit code {exit_code}")

    except Exception as e:
        # This catches errors in the task logic itself (before/after subprocess) OR the RuntimeError raised above
        error_message = f"Exception during pipeline task {task_id}: {e}"
        print(error_message, exc_info=True)
        detailed_error = traceback.format_exc()
        print(detailed_error)  # Log detailed traceback

        # Set final status to FAILED if not already set
        if not final_status_set:
            set_final_status(
                task_id,
                "FAILED",
                f"Task failed due to an internal error: {e}",
                details={"error": str(e), "traceback": detailed_error},
            )
            final_status_set = True

        # Attempt to terminate the subprocess if it's still running
        if process and process.poll() is None:
            print(
                f"Task {task_id}: Attempting to terminate lingering subprocess (PID: {process.pid}) due to task error."
            )
            try:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(
                        f"Task {task_id}: Subprocess {process.pid} did not respond to SIGTERM, sending SIGKILL."
                    )
                    process.kill()
                    process.wait()
            except Exception as term_err:
                print(
                    f"Task {task_id}: Error during cleanup termination of subprocess {process.pid}: {term_err}"
                )

        # Re-raise the original exception for Huey
        raise

    finally:
        # Close the log file if it was opened successfully
        if log_file:
            try:
                log_file.close()
                print(f"Task {task_id}: Closed log file {log_file_path}")
            except Exception as close_e:
                print(
                    f"Task {task_id}: Error closing log file {log_file_path}: {close_e}"
                )

        # --- Cleanup Redis Keys ---
        # Clean up PID key
        deleted_pid_count = redis_client.delete(redis_pid_key)
        if deleted_pid_count > 0:
            print(
                f"Task {task_id}: Removed subprocess PID key {redis_pid_key} from Redis."
            )
        else:
            # This is possible if the task finished extremely quickly, or if interrupt happened *after* wait() but before finally
            print(
                f"Task {task_id}: Attempted to remove PID key {redis_pid_key}, but it was not found (task likely finished/terminated already)."
            )

        # Note: We don't automatically delete the output_dir_for_task key or parameters_for_task key here.
        # They persist based on their timeout, allowing API access even after the task finishes.
