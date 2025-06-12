from enum import Enum
import json
import logging
from pathlib import Path as PyPath
import traceback
import yaml
import sys
import shutil
import zipfile
import os  # Import os module
import tempfile  # Import the tempfile module
import time  # Import time module
import subprocess  # Import subprocess module

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    Path as FastApiPath,
    Query,
    Body,
)
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# Remove unused make_id, replaced by Huey task ID
# from augmentoolkit.utils.make_id import make_id
from huey_config import huey  # Import the Huey instance
from tasks import (
    run_pipeline_task,
    set_final_status,
)  # Import the Huey task and set_final_status
from huey.exceptions import (
    HueyException,
    TaskException,
)  # Import Huey exceptions for status check
from redis_config import (
    get_progress,
    redis_client,
    set_progress,
)  # Import redis_client and potentially set_progress
import signal  # Import signal module

# Import path resolution logic from run_augmentoolkit
from resolve_path import resolve_path

# Import helpers
from file_operation_helpers import (
    get_safe_path,
    zip_directory,
    get_dir_structure,
    FileStructure,
    MoveItemRequest,
    handle_get_structure,
    handle_download_item,
    handle_delete_item,
    handle_move_item,
    handle_create_directory,  # <-- Add new helper
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration Loading ---
SUPER_CONFIG_PATH = PyPath("super_config.yaml")
PATH_ALIASES = {}

try:
    with open(SUPER_CONFIG_PATH, "r") as f:
        super_config = yaml.safe_load(f)
    PATH_ALIASES = super_config.get("path_aliases", {})
except FileNotFoundError:
    print(
        f"ERROR: Super config file not found at {SUPER_CONFIG_PATH}. Path aliases will not work."
    )
    # Consider if the API should fail to start if super_config is essential
    # sys.exit(1)
except yaml.YAMLError as e:
    print(f"ERROR: Error parsing super config file {SUPER_CONFIG_PATH}: {e}")
    # sys.exit(1)

INPUTS_DIR = PyPath("./inputs").resolve()
OUTPUTS_DIR = PyPath("./outputs").resolve()
GENERATION_DIR = PyPath("./generation").resolve()
CONFIGS_DIR = PyPath("./external_configs").resolve()
MAX_FILE_SIZE = 1024 * 1024 * 1000  # 1 GB Limit for uploads/downloads, adjust as needed

# Define LOGS_DIR consistent with tasks.py
LOGS_DIR = PyPath("./logs").resolve()

# Ensure base directories exist
INPUTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
GENERATION_DIR.mkdir(exist_ok=True)  # Assuming generation dir needs to exist
CONFIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)  # Ensure logs directory exists


# --- Pydantic Models ---
class PipelineRunRequest(BaseModel):
    node_path: str
    config_path: Optional[str] = None  # Relative to CONFIGS_DIR or absolute
    parameters: Optional[Dict[str, Any]] = None


class PipelineRunResponse(BaseModel):
    pipeline_id: str  # This will now be the Huey Task ID
    message: str


class PipelineStatus(str, Enum):
    PENDING = "PENDING"  # Task is waiting in the queue
    RUNNING = "RUNNING"  # Task is being executed by a worker
    COMPLETED = "COMPLETED"  # Task finished successfully
    FAILED = "FAILED"  # Task failed with an exception
    # CANCELLED / REVOKED might be needed later if task cancellation is implemented
    REVOKED = "REVOKED"  # Task was explicitly revoked


class PipelineStatusResponse(BaseModel):
    task_id: str
    status: PipelineStatus
    message: Optional[str] = None
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Progress is harder to track generically with Huey tasks unless the task itself reports it.
    # Details can contain the return value of the task or error information
    details: Optional[Dict[str, Any]] = None


# Removed unused models
# class FileContent(BaseModel):
#     content: str
#
# class QueueStateResponse(BaseModel):
#     pending_tasks: List[Dict[str, Any]]
#     running_tasks: List[Dict[str, Any]]

# Removed in-memory queue variables
# pipeline_executions: Dict[str, Dict[str, Any]] = {}
# pipeline_queue: List[Dict[str, Any]] = []


# --- Pydantic Models (New Response Model) ---
class QueueStatusResponse(BaseModel):
    pending_tasks: List[str] = Field(
        ..., description="List of task IDs currently pending execution."
    )
    scheduled_tasks: List[str] = Field(
        ..., description="List of task IDs scheduled for future execution."
    )
    message: str


class CreateDirectoryRequest(BaseModel):
    relative_path: str = Field(
        ...,
        description="The relative path within the base directory where the new directory should be created.",
    )


# --- New Pydantic model for config duplication ---
class DuplicateConfigRequest(BaseModel):
    source_alias: str = Field(
        ...,
        description="The alias from super_config.yaml pointing to the source config file.",
    )
    destination_relative_path: str = Field(
        ...,
        description="The desired relative path (including filename) for the duplicated config within the external_configs directory.",
    )


# --- Pydantic model for task parameters response ---
class TaskParametersResponse(BaseModel):
    task_id: str
    parameters: Dict[str, Any]


# --- FastAPI App ---
app = FastAPI(
    title="Augmentoolkit API",
    description="API for managing and running Augmentoolkit dataset generation pipelines using Huey.",
    version="1.0",
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",  # Frontend origin
    "http://127.0.0.1:5173",  # Also allow this variant
    "http://localhost:5174",  # Frontend origin
    "http://127.0.0.1:5174",  # Also allow this variant
    "http://localhost:3000",  # Frontend origin
    # Add other origins if needed, e.g., production URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all standard methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# TODO make sure taht pipeline settings in the config makes it run the proper pipeline
# You know what to do -- get this shippable
# edit the two videos, and record new one with the new running process
# configs polish and checking
# documentation update to match new features and utility pipelines
# also will have to rerecord interface thing so be it, because of new chat window
# and censor the history


@app.get("/", summary="Health Check")
async def read_root():
    """Basic health check endpoint."""
    return {"message": "Augmentoolkit API is running."}


@app.post(
    "/pipelines/run",
    response_model=PipelineRunResponse,
    status_code=202,
    summary="Queue a dataset generation pipeline for execution.",
)
def queue_pipeline_run(request: PipelineRunRequest):
    """
    Resolves paths using super_config aliases and enqueues a pipeline run task using Huey.
    Returns the task ID which can be used to check the status.
    """
    try:
        # as little logic for the queue and paths, in the post, as possible
        print(f"DEBUG: Queuing task with:")
        print(f"  Resolved Node Path: {request.node_path}")
        print(f"  Resolved Config Path: {request.config_path}")
        print(f"  Parameters: {request.parameters}")
        # request.parameters.update(task_id=task.id) # TODO is this how it's done?

        # Enqueue the task with resolved paths
        task = run_pipeline_task(
            node_path=request.node_path,
            config_path=request.config_path,
            parameters=request.parameters,
        )
        return PipelineRunResponse(
            pipeline_id=task.id,
            message="Pipeline run queued successfully.",
        )
    except Exception as e:
        print(f"ERROR during path resolution or task enqueueing: {e}")
        traceback.print_exc()  # Log traceback for debugging
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve paths or enqueue pipeline task: {e}",
        )


@app.get(
    "/pipelines/available",
    response_model=List[str],
    summary="Get available pipeline aliases from super_config.yaml.",
)
def get_available_pipelines():
    """
    Reads the super_config.yaml and returns a list of path aliases
    that do not point to configuration (.yaml) files.
    These typically represent runnable pipeline entry points.
    """
    available_pipelines = []
    if not SUPER_CONFIG_PATH.exists():
        logger.error(
            f"Super config file not found at {SUPER_CONFIG_PATH} for '/pipelines/available' endpoint."
        )
        # Return empty list or raise 500? Returning empty list might be more graceful for UI.
        return []
        # Alternatively: raise HTTPException(status_code=500, detail="Super configuration file not found.")

    try:
        with open(SUPER_CONFIG_PATH, "r") as f:
            super_config = yaml.safe_load(f) or {}  # Handle empty file case

        aliases = super_config.get("path_aliases", {})
        if not isinstance(aliases, dict):
            logger.warning(
                f"path_aliases in {SUPER_CONFIG_PATH} is not a dictionary. Returning empty list."
            )
            return []

        for alias, path_value in aliases.items():
            # Check if the path value is a string and does not end with .yaml
            if isinstance(path_value, str) and not path_value.strip().lower().endswith(
                ".yaml"
            ):
                available_pipelines.append(alias)

        logger.info(f"Found {len(available_pipelines)} available pipeline aliases.")
        return sorted(available_pipelines)  # Return sorted list

    except yaml.YAMLError as e:
        logger.error(f"Error parsing super config file {SUPER_CONFIG_PATH}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error parsing super configuration file: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error reading super config for available pipelines: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while retrieving available pipelines.",
        )


# and here I was about to say, oh but how do we get the return value? Butwe dont need it the main output of the pipelines are files.
# so


@app.get(
    "/configs/aliases",
    response_model=List[str],
    summary="Get available config file aliases from super_config.yaml.",
)
def get_available_config_aliases():
    """
    Reads the super_config.yaml and returns a list of path aliases
    that point to configuration (.yaml) files.
    """
    config_aliases = []
    if not SUPER_CONFIG_PATH.exists():
        logger.error(
            f"Super config file not found at {SUPER_CONFIG_PATH} for '/configs/aliases' endpoint."
        )
        return []  # Graceful empty list if file is missing

    try:
        with open(SUPER_CONFIG_PATH, "r") as f:
            super_config = yaml.safe_load(f) or {}  # Handle empty file case

        aliases = super_config.get("path_aliases", {})
        if not isinstance(aliases, dict):
            logger.warning(
                f"path_aliases in {SUPER_CONFIG_PATH} is not a dictionary. Returning empty list."
            )
            return []

        for alias, path_value in aliases.items():
            # Check if the path value is a string and *does* end with .yaml
            if isinstance(path_value, str) and path_value.strip().lower().endswith(
                ".yaml"
            ):
                config_aliases.append(alias)

        logger.info(f"Found {len(config_aliases)} available config aliases.")
        logger.info(config_aliases)
        return sorted(config_aliases)  # Return sorted list

    except yaml.YAMLError as e:
        logger.error(f"Error parsing super config file {SUPER_CONFIG_PATH}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error parsing super configuration file: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error reading super config for available config aliases: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while retrieving available config aliases.",
        )


@app.get(
    "/tasks/queue",
    response_model=QueueStatusResponse,
    summary="Get lists of pending and scheduled tasks.",
)
def get_queue_status():
    """
    Retrieves the IDs of tasks currently pending or scheduled in the Huey queue.
    Note: This does not reliably show tasks that are *actively running*.
    """
    try:
        pending_tasks = [task.id for task in huey.pending()]
        scheduled_tasks = [task.id for task in huey.scheduled()]
        count_pending = len(pending_tasks)
        count_scheduled = len(scheduled_tasks)
        logger.info(
            f"Retrieved queue status: {count_pending} pending, {count_scheduled} scheduled."
        )
        return QueueStatusResponse(
            pending_tasks=pending_tasks,
            scheduled_tasks=scheduled_tasks,
            message=f"Found {count_pending} pending and {count_scheduled} scheduled tasks.",
        )
    except Exception as e:
        logger.error(f"Error retrieving Huey queue status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve queue status: {e}"
        )


@app.get(
    "/tasks/{task_id}/status",
    response_model=PipelineStatusResponse,
    summary="Get the status of a pipeline run.",
)
def get_pipeline_status(task_id: str):
    """
    Retrieves the status of a specific pipeline run task, primarily using Redis.
    Checks Redis for final status, then progress, then checks Huey for pending/revoked state.
    """
    redis_status_key = f"status_for_task:{task_id}"
    redis_progress_key = f"progress_for_task:{task_id}"  # Assuming this is the key used by redis_config.get_progress

    try:
        # 1. Check Redis for Final Status
        final_status_json = redis_client.get(redis_status_key)
        if final_status_json:
            try:
                status_data = json.loads(final_status_json)
                logger.info(
                    f"Status check for {task_id}: Found final status in Redis: {status_data['status']}"
                )
                # Map stored status string to PipelineStatus enum
                final_status_enum = PipelineStatus(
                    status_data["status"]
                )  # Raises ValueError if invalid

                # Progress should be 1.0 for completed/failed/revoked unless explicitly stored otherwise
                progress_value = 1.0
                if (
                    final_status_enum == PipelineStatus.RUNNING
                ):  # Should not happen if final status is set correctly
                    progress_value = 0.0  # Or query progress key? Defaulting to 0.0 if wrongly marked RUNNING as final.

                # If progress info exists alongside final status, use it (e.g., last known progress before failure)
                # This part might be redundant if tasks.py sets progress correctly on final status.
                # Let's keep it simple for now: assume 1.0 for terminal states.

                return PipelineStatusResponse(
                    task_id=task_id,
                    status=final_status_enum,
                    message=status_data.get(
                        "message", "Status retrieved from final record."
                    ),
                    progress=progress_value,  # Assume 1.0 for terminal states stored here
                    details=status_data.get("details"),
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Status check for {task_id}: Failed to parse final status JSON from Redis ({redis_status_key}): {e}. Content: {final_status_json[:100]}..."
                )
                # Fall through to check other methods, but log the error
            except (
                ValueError
            ) as e:  # Handle case where status string is not in PipelineStatus enum
                logger.error(
                    f"Status check for {task_id}: Invalid status value '{status_data.get('status')}' found in Redis ({redis_status_key}): {e}"
                )
                # Fall through
            except Exception as e:  # Catch other unexpected errors during processing
                logger.error(
                    f"Status check for {task_id}: Error processing final status from Redis ({redis_status_key}): {e}",
                    exc_info=True,
                )
                # Fall through

        # 2. Check Redis for Progress (indicates RUNNING)
        pipeline_progress = get_progress(task_id)  # Use the existing helper
        if pipeline_progress:
            logger.info(
                f"Status check for {task_id}: Found progress info in Redis. Assuming RUNNING."
            )
            progress_value = min(
                pipeline_progress.get("progress", 0.0), 1.0
            )  # Cap at 1.0
            return PipelineStatusResponse(
                task_id=task_id,
                status=PipelineStatus.RUNNING,
                message=pipeline_progress.get("message", "Task is running."),
                progress=progress_value,
                details={"source": "progress_tracker"},
            )

        # 3. Check Huey for Pending/Scheduled (if no final status or progress found)
        # This requires iterating through potentially large lists, might be less efficient.
        try:
            pending_ids = {task.id for task in huey.pending()}
            scheduled_ids = {task.id for task in huey.scheduled()}

            if task_id in pending_ids or task_id in scheduled_ids:
                logger.info(
                    f"Status check for {task_id}: Found in Huey pending/scheduled list."
                )
                return PipelineStatusResponse(
                    task_id=task_id,
                    status=PipelineStatus.PENDING,
                    message="Task is pending in the queue.",
                    progress=0.0,
                    details={"source": "huey_queue"},
                )
        except Exception as e:
            logger.error(
                f"Status check for {task_id}: Error checking Huey pending/scheduled queues: {e}",
                exc_info=True,
            )
            # Continue to next check

        # 4. Check Huey for Revoked status (as a fallback)
        try:
            if huey.is_revoked(task_id):
                logger.info(
                    f"Status check for {task_id}: Found revoked status via huey.is_revoked()."
                )
                # Check if a final status was set just now or very recently
                final_status_json = redis_client.get(redis_status_key)
                if final_status_json:
                    logger.warning(
                        f"Status check for {task_id}: Task is revoked in Huey, but also has a final status record in Redis. Preferring Redis record."
                    )
                    # Re-process the Redis record found above (code duplication, consider refactor)
                    try:
                        status_data = json.loads(final_status_json)
                        final_status_enum = PipelineStatus(status_data["status"])
                        return PipelineStatusResponse(
                            task_id=task_id,
                            status=final_status_enum,
                            message=status_data.get("message"),
                            progress=1.0,
                            details=status_data.get("details"),
                        )
                    except Exception:
                        pass  # Ignore errors here, proceed with REVOKED

                # If no Redis status, return REVOKED based on Huey state
                return PipelineStatusResponse(
                    task_id=task_id,
                    status=PipelineStatus.REVOKED,
                    message="Task was revoked.",
                    progress=0.0,  # Progress is likely 0 if revoked before running
                    details={"source": "huey_revoked"},
                )
        except HueyException:
            # This likely means the task ID isn't known to Huey *at all* anymore.
            logger.warning(
                f"Status check for {task_id}: HueyException checking revoked status. Task ID likely invalid or expired from Huey storage."
            )
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline task with ID '{task_id}' not found or status expired.",
            )
        except Exception as e:
            logger.error(
                f"Status check for {task_id}: Error checking Huey revoked status: {e}",
                exc_info=True,
            )
            # Fall through to final check/404

        # 5. If none of the above, Task ID is likely invalid or expired
        logger.warning(
            f"Status check for {task_id}: No status found in Redis (final/progress) or Huey (pending/scheduled/revoked). Task may be invalid or expired."
        )
        # Check Redis one last time - maybe it appeared?
        final_status_json = redis_client.get(redis_status_key)
        if final_status_json:
            logger.info(
                f"Status check for {task_id}: Final status appeared in Redis on last check."
            )
            try:
                status_data = json.loads(final_status_json)
                final_status_enum = PipelineStatus(status_data["status"])
                return PipelineStatusResponse(
                    task_id=task_id,
                    status=final_status_enum,
                    message=status_data.get("message"),
                    progress=1.0,
                    details=status_data.get("details"),
                )
            except Exception:
                pass  # Ignore errors, raise 404

        raise HTTPException(
            status_code=404,
            detail=f"Pipeline task with ID '{task_id}' not found or status could not be determined.",
        )

    except HTTPException:  # Re-raise 404
        raise
    except Exception as e:
        logger.error(
            f"Unexpected ERROR during status check for task {task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Unexpected error retrieving task status: {e}"
        )


@app.get(
    "/tasks/{task_id}/parameters",
    response_model=TaskParametersResponse,
    summary="Get the parameters a task was executed with.",
)
def get_task_parameters(task_id: str):
    """
    Retrieves the parameters (overrides and config) used to start a specific task run.
    Parameters are stored in Redis when the task begins execution.
    """
    redis_key = f"parameters_for_task:{task_id}"
    logger.info(
        f"Request for parameters for task {task_id} using Redis key: {redis_key}"
    )

    try:
        params_json = redis_client.get(redis_key)

        if params_json is None:
            logger.warning(
                f"Parameters not found in Redis for task {task_id} (key: {redis_key}). Checking task existence."
            )
            # Check if the task ID itself is known to Huey to give a better error
            try:
                # Peek at the result without blocking to see if the task ID is valid
                huey.result(task_id, blocking=False)
                # If the above line doesn't raise HueyException, the task exists or existed.
                # Parameters might have expired or failed to store.
                raise HTTPException(
                    status_code=404,
                    detail=f"Parameters for task {task_id} not found. They may have expired or were not stored.",
                )
            except HueyException:
                # Task ID is not known to Huey
                raise HTTPException(
                    status_code=404, detail=f"Task with ID '{task_id}' not found."
                )
            except Exception as huey_check_e:
                # Catch other potential errors during the check
                logger.error(
                    f"Error checking Huey task status for {task_id} while getting parameters: {huey_check_e}"
                )
                # Fallback to generic parameter not found error
                raise HTTPException(
                    status_code=404, detail=f"Parameters for task {task_id} not found."
                )

        # Attempt to parse the JSON string
        try:
            parameters = json.loads(params_json)
            logger.info(
                f"Successfully retrieved and parsed parameters for task {task_id}"
            )
            return TaskParametersResponse(task_id=task_id, parameters=parameters)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse parameters JSON retrieved from Redis for task {task_id}. Key: {redis_key}. Content starts with: {params_json[:100]}... Error: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error decoding parameters stored for task {task_id}. Data may be corrupted.",
            )

    except HTTPException:  # Re-raise the 404/500 from inner blocks
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error retrieving parameters for task {task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving parameters for task {task_id}: {e}",
        )


def _find_and_kill_run_augmentoolkit_process() -> Optional[Dict[str, Any]]:
    """
    Fallback mechanism to find and kill run_augmentoolkit.py process directly.
    Returns dict with 'pid' and 'killed' status if found, None if not found.
    """
    try:
        # Use ps -ef to find run_augmentoolkit.py process
        result = subprocess.run(
            ["ps", "-ef"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Look for lines containing "run_augmentoolkit.py"
        for line in result.stdout.splitlines():
            if "run_augmentoolkit.py" in line and "grep" not in line:
                # Parse the PID (second field in ps -ef output)
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        pid = int(fields[1])
                        logger.warning(f"Found run_augmentoolkit.py process via ps -ef fallback: PID {pid}")
                        
                        # Try SIGINT first for graceful shutdown
                        os.kill(pid, signal.SIGINT)
                        logger.info(f"Sent SIGINT to run_augmentoolkit.py process (PID {pid}) via fallback")
                        
                        # Wait for graceful termination
                        grace_period = 8  # seconds to wait for SIGINT
                        check_interval = 0.3  # seconds between checks
                        start_time = time.time()
                        
                        # Poll to see if process terminates gracefully
                        while time.time() - start_time < grace_period:
                            try:
                                os.kill(pid, 0)  # Check if process still exists
                                time.sleep(check_interval)
                            except ProcessLookupError:
                                # Process has terminated gracefully
                                elapsed = time.time() - start_time
                                logger.info(f"Process {pid} terminated gracefully after {elapsed:.1f}s via fallback")
                                return {"pid": pid, "killed": True, "method": "SIGINT"}
                        
                        # Process didn't respond to SIGINT, escalate to SIGKILL
                        logger.warning(f"Process {pid} did not respond to SIGINT after {grace_period}s. Escalating to SIGKILL via fallback.")
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"Successfully killed run_augmentoolkit.py process (PID {pid}) with SIGKILL via fallback")
                        return {"pid": pid, "killed": True, "method": "SIGKILL"}
                        
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        logger.error(f"Failed to kill run_augmentoolkit.py process found via fallback: {e}")
                        continue
        
        logger.info("No run_augmentoolkit.py process found via ps -ef fallback")
        return None
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run ps -ef command in fallback: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in run_augmentoolkit.py fallback kill mechanism: {e}")
        return None


@app.post(
    "/tasks/{task_id}/interrupt",
    status_code=200,
    summary="Interrupt a running pipeline task subprocess or revoke a pending task.",
)
def interrupt_or_revoke_task(task_id: str):
    """
    Attempts to interrupt a RUNNING task's subprocess via SIGINT,
    or revokes a PENDING task (setting final status in Redis).
    Reports status if already finished/revoked based on Redis final status.
    """
    redis_pid_key = f"worker_pid_for_task:{task_id}"
    redis_status_key = f"status_for_task:{task_id}"
    logger.info(f"Received interrupt request for task {task_id}")

    try:
        # === Step 1: Check Final Status in Redis ===
        final_status_json = redis_client.get(redis_status_key)
        if final_status_json:
            try:
                status_data = json.loads(final_status_json)
                existing_status = status_data.get("status", "UNKNOWN").upper()
                logger.warning(
                    f"Interrupt request for {task_id}: Task already has a final status in Redis: {existing_status}"
                )
                # Return 409 Conflict if already completed, failed, or revoked
                if existing_status in ["COMPLETED", "FAILED", "REVOKED"]:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Task {task_id} has already finished with status: {existing_status}.",
                    )
                # Otherwise, log warning but proceed (e.g., if status was somehow inconsistent)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(
                    f"Interrupt request for {task_id}: Error reading existing final status from Redis ({redis_status_key}): {e}. Proceeding with interrupt/revoke attempt."
                )

        # === Step 2: Check if Running (via Subprocess PID in Redis) ===
        pid_bytes = redis_client.get(redis_pid_key)
        if pid_bytes:
            pid = None
            try:
                # --- Get PID ---
                try:
                    pid = int(
                        pid_bytes
                    )  # Assumes redis client returns bytes/string convertible to int
                except ValueError:
                    logger.error(
                        f"Invalid non-integer PID string '{pid_bytes}' found in Redis for key {redis_pid_key}. Cleaning up."
                    )
                    redis_client.delete(redis_pid_key)  # Clean up invalid data
                    # Proceed to check if it's pending, maybe the PID was wrong but task exists
                    # Try fallback mechanism to find and kill run_augmentoolkit.py
                    logger.info(f"Attempting fallback mechanism to find run_augmentoolkit.py process for task {task_id}")
                    fallback_result = _find_and_kill_run_augmentoolkit_process()
                    if fallback_result and fallback_result.get("killed"):
                        set_final_status(
                            task_id,
                            "REVOKED",
                            f"Task {task_id} terminated via fallback mechanism (found PID {fallback_result['pid']}).",
                            details={"source": "api_interrupt_fallback", "pid": fallback_result['pid'], "termination_method": f"{fallback_result.get('method', 'SIGKILL')}_FALLBACK"}
                        )
                        return {
                            "message": f"Task {task_id} terminated via fallback mechanism after invalid Redis PID."
                        }
                    pass  # Fall through to pending check

                if pid is not None:  # Only proceed if PID was valid
                    # --- Send SIGINT and wait for graceful termination ---
                    logger.info(
                        f"Task {task_id} appears to be running (Subprocess PID {pid} found). Sending SIGINT."
                    )
                    os.kill(pid, signal.SIGINT)
                    
                    # Wait for graceful termination
                    grace_period = 8  # seconds to wait for SIGINT
                    check_interval = 0.3  # seconds between checks
                    start_time = time.time()
                    
                    # Poll to see if process terminates gracefully
                    process_terminated_gracefully = False
                    while time.time() - start_time < grace_period:
                        try:
                            os.kill(pid, 0)  # Check if process still exists (signal 0 = no-op)
                            time.sleep(check_interval)
                        except ProcessLookupError:
                            # Process has terminated
                            elapsed = time.time() - start_time
                            process_terminated_gracefully = True
                            logger.info(f"Process {pid} terminated gracefully after {elapsed:.1f}s")
                            break
                        except PermissionError:
                            # Process exists but we can't check it - treat as still running
                            time.sleep(check_interval)
                    
                    if process_terminated_gracefully:
                        # Graceful termination succeeded
                        elapsed = time.time() - start_time
                        set_final_status(
                            task_id,
                            "REVOKED",
                            f"Task {task_id} gracefully terminated with SIGINT after {elapsed:.1f}s.",
                            details={"source": "api_interrupt", "pid": pid, "termination_method": "SIGINT"}
                        )
                        return {
                            "message": f"Task {task_id} gracefully terminated with SIGINT after {elapsed:.1f}s."
                        }
                    
                    # --- Escalate to SIGKILL ---
                    logger.warning(
                        f"Process {pid} did not respond to SIGINT after {grace_period}s. Escalating to SIGKILL."
                    )
                    
                    try:
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"Sent SIGKILL to subprocess {pid} for task {task_id}")
                        
                        # Brief wait for SIGKILL to take effect
                        time.sleep(2)
                        
                        # Verify SIGKILL worked
                        try:
                            os.kill(pid, 0)
                            # If we get here, process is STILL running after SIGKILL
                            logger.error(f"Process {pid} still exists after SIGKILL! This should not happen.")
                            message = f"Task {task_id} may still be running despite SIGKILL (PID {pid}). Manual intervention may be required."
                            termination_method = "SIGKILL_FAILED"
                        except ProcessLookupError:
                            # SIGKILL succeeded
                            logger.info(f"Process {pid} successfully terminated with SIGKILL")
                            message = f"Task {task_id} forcefully terminated with SIGKILL after SIGINT timeout."
                            termination_method = "SIGKILL"
                            
                    except ProcessLookupError:
                        # Process terminated between our grace period check and SIGKILL attempt
                        logger.info(f"Process {pid} terminated just before SIGKILL was sent")
                        message = f"Task {task_id} terminated during escalation (process ended just after SIGINT timeout)."
                        termination_method = "SIGINT_DELAYED"
                    except PermissionError:
                        logger.error(f"Permission denied sending SIGKILL to PID {pid} for task {task_id}")
                        message = f"Permission error escalating to SIGKILL for task {task_id} (PID {pid})."
                        termination_method = "PERMISSION_ERROR"
                    except Exception as kill_e:
                        logger.error(f"Unexpected error sending SIGKILL to PID {pid}: {kill_e}")
                        message = f"Error escalating to SIGKILL for task {task_id}: {kill_e}"
                        termination_method = "SIGKILL_ERROR"
                    
                    # Set final status with escalation details
                    set_final_status(
                        task_id,
                        "REVOKED",
                        message,
                        details={
                            "source": "api_interrupt", 
                            "pid": pid, 
                            "termination_method": termination_method,
                            "escalated_to_sigkill": True
                        }
                    )
                    
                    return {"message": message}

            except ProcessLookupError:
                logger.warning(
                    f"Subprocess with PID {pid} for task {task_id} not found when sending SIGINT. It likely finished/crashed recently."
                )
                # Don't immediately raise 404, check final status / pending status below.
                # Try fallback mechanism before falling through
                logger.info(f"Attempting fallback mechanism to find run_augmentoolkit.py process for task {task_id}")
                fallback_result = _find_and_kill_run_augmentoolkit_process()
                if fallback_result and fallback_result.get("killed"):
                    set_final_status(
                        task_id,
                        "REVOKED",
                        f"Task {task_id} terminated via fallback mechanism (found PID {fallback_result['pid']}).",
                        details={"source": "api_interrupt_fallback", "pid": fallback_result['pid'], "termination_method": f"{fallback_result.get('method', 'SIGKILL')}_FALLBACK", "original_pid": pid}
                    )
                    return {
                        "message": f"Task {task_id} terminated via fallback mechanism after PID {pid} was not found."
                    }
                pass  # Fall through
            except PermissionError:
                logger.error(
                    f"Permission denied trying to send SIGINT to PID {pid} for task {task_id}."
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Permission error signaling subprocess {pid}.",
                )
            except Exception as e:
                pid_str = str(pid) if pid is not None else "<conversion failed>"
                logger.error(
                    f"Unexpected error processing interrupt for PID {pid_str} / task {task_id}: {e}",
                    exc_info=True,
                )
                # Fall through to check pending/final status
                pass

        # === Fallback: Try to find and kill run_augmentoolkit.py directly ===
        # This handles cases where Redis PID was missing or other failures occurred
        if not redis_client.exists(redis_pid_key):
            logger.info(f"No PID found in Redis for task {task_id}. Attempting fallback mechanism.")
            fallback_result = _find_and_kill_run_augmentoolkit_process()
            if fallback_result and fallback_result.get("killed"):
                set_final_status(
                    task_id,
                    "REVOKED",
                    f"Task {task_id} terminated via fallback mechanism (found PID {fallback_result['pid']}).",
                    details={"source": "api_interrupt_fallback", "pid": fallback_result['pid'], "termination_method": f"{fallback_result.get('method', 'SIGKILL')}_FALLBACK", "reason": "no_redis_pid"}
                )
                return {
                    "message": f"Task {task_id} terminated via fallback mechanism (no Redis PID available)."
                }

        # === Step 3: Attempt to Revoke (if not running and no final status yet) ===
        # Double-check final status in case it finished between checks
        final_status_json = redis_client.get(redis_status_key)
        if final_status_json:
            try:
                status_data = json.loads(final_status_json)
                existing_status = status_data.get("status", "UNKNOWN").upper()
                logger.warning(
                    f"Interrupt request for {task_id}: Task finished with status {existing_status} just before revocation attempt."
                )
                raise HTTPException(
                    status_code=409,
                    detail=f"Task {task_id} finished with status: {existing_status} just before revocation.",
                )
            except Exception:
                pass  # Ignore parsing errors, proceed with revoke attempt

        logger.info(
            f"Task {task_id} is not running (no valid PID found) or signal failed. Attempting to revoke (PENDING)."
        )
        try:
            # Use revoke_once=True: if the task started between checks, this won't revoke it.
            was_revoked = huey.revoke(task_id, revoke_once=True)

            if was_revoked:
                logger.info(
                    f"Task {task_id} was pending and successfully revoked via Huey."
                )
                # Set final status in Redis to REVOKED
                set_final_status(
                    task_id,
                    "REVOKED",
                    f"Task {task_id} was revoked while pending.",
                    details={"source": "api_revoke"},
                )
                return {
                    "message": f"Task {task_id} was pending and has been successfully revoked."
                }
            else:
                logger.warning(
                    f"Attempted to revoke task {task_id}, but revoke command returned False. Task might have just started, finished, already revoked, or ID is invalid."
                )
                # Check final status *again*
                final_status_json = redis_client.get(redis_status_key)
                if final_status_json:
                    try:
                        status_data = json.loads(final_status_json)
                        existing_status = status_data.get("status", "UNKNOWN").upper()
                        raise HTTPException(
                            status_code=409,
                            detail=f"Task {task_id} could not be revoked; final status is already {existing_status}.",
                        )
                    except Exception:
                        pass  # Ignore parsing errors

                # If still no final status, maybe it just started.
                # Check PID key again
                if redis_client.exists(redis_pid_key):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Failed to revoke task {task_id}. It likely started running just now.",
                    )
                else:
                    # Maybe already revoked via Huey but Redis status not set? Check Huey again.
                    if huey.is_revoked(task_id):
                        logger.info(
                            f"Task {task_id} was already revoked in Huey. Setting Redis status."
                        )
                        set_final_status(
                            task_id,
                            "REVOKED",
                            f"Task {task_id} was already revoked.",
                            details={"source": "api_revoke_already_revoked"},
                        )
                        raise HTTPException(
                            status_code=409,
                            detail=f"Task {task_id} was already revoked.",
                        )
                    else:
                        # Task ID might be invalid
                        raise HTTPException(
                            status_code=404,
                            detail=f"Failed to revoke task {task_id}. Task not found or state is inconsistent.",
                        )

        except HueyException as e:
            logger.error(f"HueyException during revoke/check for task {task_id}: {e}")
            # Check if a final status exists in Redis despite Huey error
            final_status_json = redis_client.get(redis_status_key)
            if final_status_json:
                logger.warning(
                    f"Task {task_id} not found in Huey for revoke, but has Redis status. Assuming Redis is correct."
                )
                try:
                    status_data = json.loads(final_status_json)
                    existing_status = status_data.get("status", "UNKNOWN").upper()
                    raise HTTPException(
                        status_code=409,
                        detail=f"Task {task_id} has final status: {existing_status} (not found in Huey).",
                    )
                except Exception:
                    pass  # Ignore parsing errors
            raise HTTPException(
                status_code=404,
                detail=f"Task with ID '{task_id}' not found in Huey for revocation.",
            )

    except HTTPException:
        raise  # Re-raise intentional HTTP exceptions
    except Exception as e:
        logger.error(
            f"Unexpected error during interrupt/revoke top-level for task {task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected server error occurred: {e}"
        )


# === Log Management ===


@app.get("/tasks/{task_id}/logs", summary="Get logs for a specific task.")
def get_task_logs(
    task_id: str,
    tail: Optional[int] = Query(
        None, description="Return only the last N lines of the log.", ge=1
    ),
):
    """
    Retrieves the log file content for a given task ID.
    Optionally returns only the last N lines using the `tail` query parameter.
    """
    log_file_path = LOGS_DIR / f"{task_id}.log"
    logger.info(f"Attempting to retrieve logs for task {task_id} from {log_file_path}")

    if not log_file_path.exists() or not log_file_path.is_file():
        logger.warning(f"Log file not found for task {task_id} at {log_file_path}")
        # Check Huey status to provide more context if the task itself doesn't exist
        try:
            if huey.is_revoked(task_id):
                status_msg = "Task was revoked."
            elif huey.result(task_id, blocking=False) is not None:
                status_msg = "Task has finished."
            else:
                # It might be pending/running but hasn't created the log file yet
                status_msg = (
                    "Task is pending or running, but logs are not yet available."
                )
        except HueyException:
            status_msg = "Task ID not found in task queue."
        except Exception:
            status_msg = "Could not determine task status."

        raise HTTPException(
            status_code=404,
            detail=f"Log file for task {task_id} not found. {status_msg}",
        )

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if tail is not None:
            log_content = "".join(lines[-tail:])
            message = f"Successfully retrieved last {len(lines[-tail:])} lines for task {task_id}."
        else:
            log_content = "".join(lines)
            message = f"Successfully retrieved all logs for task {task_id}."

        logger.info(f"Successfully retrieved logs for task {task_id}")
        return JSONResponse(
            content={"task_id": task_id, "message": message, "logs": log_content}
        )

    except Exception as e:
        logger.error(
            f"Error reading log file {log_file_path} for task {task_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Failed to read log file: {e}")


@app.delete("/logs", status_code=200, summary="Clear all task log files.")
def clear_all_logs():
    """
    Deletes all *.log files within the logs directory.
    """
    logger.info(f"Attempting to clear all logs in directory: {LOGS_DIR}")
    deleted_count = 0
    errors = []

    try:
        for item in LOGS_DIR.iterdir():
            if item.is_file() and item.suffix == ".log":
                try:
                    item.unlink()  # Delete the file
                    deleted_count += 1
                    logger.debug(f"Deleted log file: {item.name}")
                except Exception as e:
                    logger.error(f"Failed to delete log file {item.name}: {e}")
                    errors.append(f"Failed to delete {item.name}: {e}")

        message = f"Successfully deleted {deleted_count} log file(s) from {LOGS_DIR}."
        if errors:
            message += f" Errors occurred for {len(errors)} file(s)."
            logger.warning(f"Errors occurred during log deletion: {errors}")
            # Return 207 Multi-Status if some deletions failed?
            # For simplicity, returning 200 but noting errors in the message.
            return JSONResponse(
                status_code=200, content={"message": message, "errors": errors}
            )
        else:
            logger.info(message)
            return {"message": message}

    except Exception as e:
        logger.error(
            f"Error accessing logs directory {LOGS_DIR} for deletion: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to access or clear logs directory: {e}"
        )


@app.delete(
    "/tasks/{task_id}/logs",
    status_code=200,
    summary="Delete log file for a specific task.",
)
def delete_task_log(task_id: str):
    """
    Deletes the log file associated with a specific task ID.
    """
    log_file_path = LOGS_DIR / f"{task_id}.log"
    logger.info(f"Attempting to delete log file for task {task_id} at {log_file_path}")

    if not log_file_path.exists() or not log_file_path.is_file():
        logger.warning(
            f"Log file not found for task {task_id} at {log_file_path} during delete request."
        )
        raise HTTPException(
            status_code=404, detail=f"Log file for task {task_id} not found."
        )

    try:
        log_file_path.unlink()  # Delete the file
        logger.info(f"Successfully deleted log file: {log_file_path.name}")
        return {"message": f"Successfully deleted log file for task {task_id}."}
    except Exception as e:
        logger.error(
            f"Failed to delete log file {log_file_path.name}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to delete log file: {e}")


@app.get("/logs", summary="List all available log files.")
def list_log_files():
    """
    Returns a list of all *.log files currently in the logs directory.
    """
    logger.info(f"Listing log files in directory: {LOGS_DIR}")
    log_files = []
    try:
        for item in LOGS_DIR.iterdir():
            if item.is_file() and item.suffix == ".log":
                log_files.append(item.name)
        logger.info(f"Found {len(log_files)} log files.")
        return {"log_files": sorted(log_files)}  # Return sorted list
    except Exception as e:
        logger.error(f"Error listing logs directory {LOGS_DIR}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to list logs directory: {e}"
        )


### OUTPUT MANAGEMENT ###
# download the output dir of a specific task execution
# view the output dir structure (basically, a `tree .` sort of thing, except accessible/traversable in a way that enables the next route--)
# download something in the output dir structure. Takes a route relative to the output dir and downloads the thing that points to if it exists. A way of getting specific output dirs, or even single files within those, without needing a task ID that you may no longer have.
# Delete an output dir. Stuff can get cluttered. Takes the same sort of path input as the download thing


# Add other endpoints (file upload/download, etc.) as needed.
# Remember to adjust file paths (INPUTS_DIR, etc.) if necessary,
# especially if tasks running in the Huey worker need to access them.

# --- Output Management --- #


@app.get(
    "/tasks/{task_id}/outputs/download", summary="Download output directory for a task."
)
def download_task_output(task_id: str):
    """Downloads the complete output directory for a specific task run as a zip file."""
    # 1. Get output directory path from Redis
    redis_key = f"output_dir_for_task:{task_id}"
    output_dir_str = redis_client.get(redis_key)

    if not output_dir_str:
        logger.warning(
            f"Output directory mapping not found in Redis for task {task_id} (key: {redis_key})"
        )
        # Optionally check Huey status here too for better error message
        raise HTTPException(
            status_code=404,
            detail=f"Output directory mapping not found for task {task_id}. Task may not have run, failed early, or mapping expired.",
        )

    # 2. Validate the retrieved path
    try:
        # Convert back to Path and resolve (resolve might not be strictly needed if stored absolute, but good practice)
        task_output_dir = PyPath(output_dir_str).resolve()
        logger.info(
            f"Request to download output for task {task_id}. Resolved path from Redis: {task_output_dir}"
        )
        if not task_output_dir.is_dir():
            logger.error(
                f"Path retrieved from Redis for task {task_id} is not a valid directory: {task_output_dir}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Output directory for task {task_id} exists in mapping but is not a valid directory on disk.",
            )
    except Exception as path_e:
        logger.error(
            f"Error processing path '{output_dir_str}' from Redis for task {task_id}: {path_e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Error validating output directory path retrieved from storage.",
        )

    # Create a temporary zip file
    # Use task_id for zip name for consistency, even if dir name differs
    zip_filename = f"task_{task_id}_output.zip"
    # Use tempfile.gettempdir() for OS-independent temporary directory
    temp_dir = tempfile.gettempdir()
    temp_zip_path = PyPath(temp_dir) / zip_filename

    try:
        logger.info(f"Zipping directory {task_output_dir} to {temp_zip_path}")
        zip_directory(task_output_dir, temp_zip_path)
        logger.info(f"Zip file created for task {task_id}")
        # Return FileResponse without the background task
        return FileResponse(
            temp_zip_path, media_type="application/zip", filename=zip_filename
        )
    except Exception as e:
        logger.error(
            f"Failed to create or send zip file for task {task_id}: {e}", exc_info=True
        )
        # Clean up temp file in case of error before sending response
        if temp_zip_path.exists():
            try:
                os.remove(temp_zip_path)
            except Exception as cleanup_e:
                logger.error(
                    f"Error cleaning up temp zip file {temp_zip_path}: {cleanup_e}"
                )
        raise HTTPException(status_code=500, detail=f"Failed to create zip file: {e}")


@app.get(
    "/tasks/{task_id}/outputs/structure",
    response_model=List[FileStructure],
    summary="Get structure of a task's output directory.",
)
def get_task_output_structure(task_id: str):
    """Retrieves the file and folder structure within a specific task's output directory."""
    # 1. Get output directory path from Redis
    redis_key = f"output_dir_for_task:{task_id}"
    output_dir_str = redis_client.get(redis_key)

    if not output_dir_str:
        logger.warning(
            f"Output directory mapping not found in Redis for task {task_id} (key: {redis_key})"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Output directory mapping not found for task {task_id}. Task may not have run, failed early, or mapping expired.",
        )

    # 2. Validate the retrieved path
    try:
        task_output_dir = PyPath(output_dir_str).resolve()
        logger.info(
            f"Request for output structure for task {task_id}. Resolved path from Redis: {task_output_dir}"
        )
        if not task_output_dir.is_dir():
            logger.error(
                f"Path retrieved from Redis for task {task_id} is not a valid directory: {task_output_dir}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Output directory for task {task_id} exists in mapping but is not a valid directory on disk.",
            )
    except Exception as path_e:
        logger.error(
            f"Error processing path '{output_dir_str}' from Redis for task {task_id}: {path_e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Error validating output directory path retrieved from storage.",
        )

    # Pass the directory itself as the base for relative paths
    structure = get_dir_structure(
        task_output_dir, base_path_for_relative=task_output_dir
    )
    return structure


@app.get(
    "/outputs/structure/{relative_path:path}",
    response_model=List[FileStructure],
    summary="Get structure of a path within the outputs directory.",
)
def get_output_structure(relative_path: str = "."):
    """Retrieves the file/folder structure for a given path relative to the main outputs directory."""
    print(
        f"DEBUG [/outputs/structure]: Received request with relative_path='{relative_path}'"
    )
    print(
        f"DEBUG [/outputs/structure]: Calling handle_get_structure with base_dir='{OUTPUTS_DIR}', relative_path='{relative_path}'"
    )
    return handle_get_structure(OUTPUTS_DIR, relative_path)


@app.get(
    "/outputs/download/{relative_path:path}",
    summary="Download a specific file or folder from outputs.",
)
def download_output_item(relative_path: str):
    """Downloads a specific file or folder (as zip) relative to the main outputs directory."""
    return handle_download_item(OUTPUTS_DIR, relative_path)


@app.delete(
    "/outputs/{relative_path:path}",
    status_code=200,
    summary="Delete a file or folder from outputs.",
)
def delete_output_item(relative_path: str):
    """Deletes a specific file or folder relative to the main outputs directory."""
    return handle_delete_item(OUTPUTS_DIR, relative_path)


@app.post(
    "/outputs/move",
    status_code=200,
    summary="Move/Rename a file or folder within outputs.",
)
def move_output_item(request: MoveItemRequest):
    """Moves or renames a file/folder within the main outputs directory."""
    return handle_move_item(
        OUTPUTS_DIR, request.source_relative_path, request.destination_relative_path
    )


### Input File Management ###


@app.get(
    "/inputs/structure/{relative_path:path}",
    response_model=List[FileStructure],
    summary="Get structure of a path within the inputs directory.",
)
def get_input_structure(relative_path: str = "."):
    """Retrieves the file/folder structure for a given path relative to the main inputs directory."""
    print(
        f"DEBUG [/inputs/structure]: Received request with relative_path='{relative_path}'"
    )
    print(
        f"DEBUG [/inputs/structure]: Calling handle_get_structure with base_dir='{INPUTS_DIR}', relative_path='{relative_path}'"
    )
    return handle_get_structure(INPUTS_DIR, relative_path)


@app.get(
    "/inputs/download/{relative_path:path}",
    summary="Download a specific file or folder from inputs.",
)
def download_input_item(relative_path: str):
    """Downloads a specific file or folder (as zip) relative to the main inputs directory."""
    return handle_download_item(INPUTS_DIR, relative_path)


@app.delete(
    "/inputs/{relative_path:path}",
    status_code=200,
    summary="Delete a file or folder from inputs.",
)
def delete_input_item(relative_path: str):
    """Deletes a specific file or folder relative to the main inputs directory."""
    return handle_delete_item(INPUTS_DIR, relative_path)


@app.post(
    "/inputs/move",
    status_code=200,
    summary="Move/Rename a file or folder within inputs.",
)
def move_input_item(request: MoveItemRequest):
    """Moves or renames a file/folder within the main inputs directory."""
    return handle_move_item(
        INPUTS_DIR, request.source_relative_path, request.destination_relative_path
    )


@app.post(
    "/inputs/directory",
    status_code=201,
    summary="Create a new directory within the inputs folder.",
)
def create_input_directory(request: CreateDirectoryRequest):
    """
    Creates a new directory at the specified relative path within the main inputs directory.
    Uses the shared handle_create_directory helper.
    """
    return handle_create_directory(INPUTS_DIR, request.relative_path)


# --- Helper Function for Uploading --- #
async def _save_uploaded_files(
    target_dir: PyPath, files: List[UploadFile]
) -> JSONResponse:
    """Helper function to save uploaded files to a target directory."""
    try:
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured target input directory exists: {target_dir}")
    except Exception as mkdir_e:
        logger.error(
            f"Failed to create target input directory {target_dir}: {mkdir_e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Could not create target directory '{target_dir.relative_to(INPUTS_DIR)}'. Check permissions and path validity.",
        )

    uploaded_files_info = []
    extracted_zips_info = []  # Keep track of successfully extracted zips
    errors = []
    for file in files:
        if not file.filename:
            logger.warning("Skipping upload for file without filename.")
            continue
        safe_filename = os.path.basename(file.filename)
        file_path = target_dir / safe_filename
        is_zip = safe_filename.lower().endswith(".zip")

        try:
            # Save the file first
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Temporarily saved uploaded file: {file_path}")

            if is_zip:
                logger.info(
                    f"Detected zip file: {safe_filename}. Attempting extraction..."
                )
                # Create a subdirectory named after the zip file (without extension)
                extraction_dirname = file_path.stem  # e.g., "loras" from "loras.zip"
                extraction_path = target_dir / extraction_dirname
                extraction_path.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure the subdir exists
                logger.info(
                    f"Extracting zip content into subdirectory: {extraction_path}"
                )

                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        # Extract into the subdirectory
                        zip_ref.extractall(extraction_path)
                    logger.info(
                        f"Successfully extracted {safe_filename} to {extraction_path}"
                    )

                    # Report success using the subdirectory path
                    extracted_zips_info.append(
                        {
                            "filename": safe_filename,
                            "extracted_to": str(
                                extraction_path.relative_to(INPUTS_DIR)
                            ),
                        }
                    )

                    # Clean up the original zip file *after* successful extraction
                    try:
                        file_path.unlink()
                        logger.info(
                            f"Successfully cleaned up original zip file: {file_path}"
                        )
                    except OSError as unlink_e:
                        logger.error(
                            f"Error deleting zip file {file_path} after extraction: {unlink_e}"
                        )
                        # Append error about cleanup failure, but extraction was successful
                        errors.append(
                            f"Error deleting zip {safe_filename} after extraction: {unlink_e}"
                        )

                    # Do not add the zip file itself to uploaded_files_info since it's deleted
                except zipfile.BadZipFile:
                    logger.error(
                        f"Failed to extract zip file {safe_filename}: Invalid zip format."
                    )
                    errors.append(
                        f"Failed to extract {safe_filename}: Invalid zip file."
                    )
                    # Keep the invalid zip file for inspection, add it to regular uploads with error
                    uploaded_files_info.append(
                        {
                            "filename": safe_filename,
                            "size": file_path.stat().st_size,
                            "path": str(file_path.relative_to(INPUTS_DIR)),
                            "error": "Invalid zip file",
                        }
                    )
                except Exception as extract_e:
                    logger.error(
                        f"Failed to extract zip file {safe_filename}: {extract_e}",
                        exc_info=True,
                    )
                    errors.append(f"Failed to extract {safe_filename}: {extract_e}")
                    # Keep the zip file if extraction failed, add it to regular uploads with error
                    uploaded_files_info.append(
                        {
                            "filename": safe_filename,
                            "size": file_path.stat().st_size,
                            "path": str(file_path.relative_to(INPUTS_DIR)),
                            "error": f"Extraction failed: {extract_e}",
                        }
                    )
            else:
                # It's not a zip file, add to regular uploaded files list
                uploaded_files_info.append(
                    {
                        "filename": safe_filename,
                        "size": file_path.stat().st_size,
                        "path": str(file_path.relative_to(INPUTS_DIR)),
                    }
                )
                logger.info(f"Successfully saved non-zip file: {file_path}")

        except Exception as e:
            logger.error(
                f"Failed to process or save file {safe_filename} to {file_path}: {e}",
                exc_info=True,
            )
            errors.append(f"Failed to save/process {safe_filename}: {e}")
            # Ensure temp file is cleaned up if saving failed before zip check
            if (
                file_path.exists()
                and not is_zip
                and not any(
                    info["filename"] == safe_filename for info in uploaded_files_info
                )
            ):
                try:
                    file_path.unlink()  # Clean up partial/failed non-zip saves
                except OSError:
                    pass
        finally:
            # Ensure the underlying temporary file is closed
            try:
                file.file.close()
            except Exception:
                pass  # Ignore errors closing potentially already closed/broken file handle

    response_message = (
        f"Upload process completed for path '{target_dir.relative_to(INPUTS_DIR)}'."
    )
    status_code = 201  # Default to Created/OK
    if errors:
        response_message += f" Errors occurred for {len(errors)} file(s)."
        if not uploaded_files_info and not extracted_zips_info:
            status_code = 500  # All failed
        else:
            status_code = (
                207  # Partial success (some uploads/extractions worked, some failed)
            )

    # Add extraction info to the response content
    response_content = {
        "message": response_message,
        "uploaded_files": uploaded_files_info,
        "extracted_zips": extracted_zips_info,  # Add info about extracted zips
        "errors": errors,
    }

    return JSONResponse(status_code=status_code, content=response_content)


@app.post(
    "/inputs/upload/{relative_path:path}",
    status_code=201,
    summary="Upload files to a specific path within inputs.",
)
async def upload_input_files(
    relative_path: str = FastApiPath(
        ...,
        description="Target path within inputs directory. If uploading a zip, it will be extracted here. Path created if needed.",
    ),
    files: List[UploadFile] = File(
        ..., description="Files to upload. Zip files will be automatically extracted."
    ),
):
    """
    Uploads one or more files to a specified path within the main inputs directory.
    If a '.zip' file is uploaded, it will be automatically extracted into the target path,
    and the original '.zip' file will be deleted upon successful extraction.
    Non-zip files are saved normally. Allows uploading to the root directory.
    """
    logger.info(f"Request to upload files to input path: {relative_path}")
    target_dir = get_safe_path(INPUTS_DIR, relative_path)

    # Removed the check that prevents uploading to the root directory.
    # get_safe_path ensures we stay within INPUTS_DIR.

    return await _save_uploaded_files(target_dir, files)


# Add other endpoints (file upload/download, etc.) as needed.
# Remember to adjust file paths (INPUTS_DIR, etc.) if necessary,
# especially if tasks running in the Huey worker need to access them.

# besides input dir management, the only thing left is config file management which is even simpler

# NOTE to self -- do we need a file output dir upload route? IF someone has a partially-completed thing that they want another augmentoolkit server to continue running, there would be a use for it... sure but not in this version of the project that is for later. Perhaps.

### Config File Management ###


@app.get(
    "/configs/structure/{relative_path:path}",
    response_model=List[FileStructure],
    summary="Get structure of a path within the configs directory.",
)
def get_config_structure(relative_path: str = "."):
    """Retrieves the file/folder structure for a given path relative to the main external_configs directory."""
    return handle_get_structure(CONFIGS_DIR, relative_path)


@app.get(
    "/configs/content/{relative_path:path}",
    summary="Get content of a specific config file.",
)
def get_config_content(relative_path: str):
    """Retrieves the content of a specific file within the external_configs directory."""
    logger.info(f"Request to get content for config: {relative_path}")
    target_path = get_safe_path(CONFIGS_DIR, relative_path)

    if not target_path.exists():
        logger.warning(f"Config file not found: {target_path}")
        raise HTTPException(
            status_code=404, detail=f"Config file '{relative_path}' not found."
        )

    if not target_path.is_file():
        logger.warning(f"Path is not a file: {target_path}")
        raise HTTPException(
            status_code=400, detail=f"Path '{relative_path}' is not a file."
        )

    # Return as plain text, could enhance with media type detection later
    return FileResponse(target_path, media_type="text/plain", filename=target_path.name)


@app.post(
    "/configs/content/{relative_path:path}",
    status_code=200,
    summary="Create or update a config file.",
)
def save_config_content(
    relative_path: str, content: str = Body(..., media_type="text/plain")
):
    """Creates a new config file or overwrites an existing one at the specified path within external_configs."""
    logger.info(f"Request to save content for config: {relative_path}")
    target_path = get_safe_path(CONFIGS_DIR, relative_path)

    # Prevent writing to the root directory itself
    if target_path == CONFIGS_DIR.resolve():
        logger.warning(
            f"Attempt to write directly to config root directory blocked for path: {relative_path}"
        )
        raise HTTPException(
            status_code=400,
            detail="Cannot write directly to the root configs directory.",
        )

    try:
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Write content (overwrite if exists)
        target_path.write_text(content, encoding="utf-8")
        logger.info(f"Successfully saved config file: {target_path}")
        return {"message": f"Successfully saved config '{relative_path}'."}
    except Exception as e:
        logger.error(f"Failed to save config file {target_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save config file '{relative_path}': {e}"
        )


@app.delete(
    "/configs/{relative_path:path}",
    status_code=200,
    summary="Delete a specific config file or directory.",
)
def delete_config_file(relative_path: str):
    """Deletes a specific file or directory within the external_configs directory."""
    logger.info(f"Request to delete config item: {relative_path}")
    target_path = get_safe_path(CONFIGS_DIR, relative_path)

    # Prevent deleting the root directory itself
    if target_path == CONFIGS_DIR.resolve():
        logger.warning(
            f"Attempt to delete config root directory blocked for path: {relative_path}"
        )
        raise HTTPException(
            status_code=400, detail="Cannot delete the root configs directory."
        )

    if not target_path.exists():
        logger.warning(f"Config item not found for deletion: {target_path}")
        raise HTTPException(
            status_code=404, detail=f"Config item '{relative_path}' not found."
        )

    try:
        if target_path.is_file():
            item_type = "file"
            target_path.unlink()
            logger.info(f"Successfully deleted config file: {target_path}")
        elif target_path.is_dir():
            item_type = "directory"
            shutil.rmtree(target_path)  # Use shutil.rmtree for directories
            logger.info(f"Successfully deleted config directory: {target_path}")
        else:
            # Should not happen if exists() is true, but handle just in case
            logger.error(
                f"Target path exists but is neither a file nor a directory: {target_path}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Target '{relative_path}' is neither a file nor a directory.",
            )

        return {
            "message": f"Successfully deleted config {item_type} '{relative_path}'."
        }
    except Exception as e:
        logger.error(f"Failed to delete config item {target_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete config item '{relative_path}': {e}",
        )


@app.post(
    "/configs/move",
    status_code=200,
    summary="Move/Rename a file or folder within configs.",
)
def move_config_item(request: MoveItemRequest):
    """Moves or renames a file/folder within the main external_configs directory."""
    return handle_move_item(
        CONFIGS_DIR, request.source_relative_path, request.destination_relative_path
    )


@app.post(
    "/configs/directory",
    status_code=201,
    summary="Create a new directory within the configs folder.",
)
def create_config_directory(request: CreateDirectoryRequest):
    """
    Creates a new directory at the specified relative path within the main external_configs directory.
    Uses the shared handle_create_directory helper.
    """
    return handle_create_directory(CONFIGS_DIR, request.relative_path)


@app.post(
    "/configs/duplicate",
    status_code=201,
    summary="Duplicate a pipeline config into external_configs.",
)
def duplicate_config_file(request: DuplicateConfigRequest):
    """
    Duplicates a configuration file identified by a super_config.yaml alias
    into the external_configs directory.
    """
    source_alias = request.source_alias
    dest_rel_path = request.destination_relative_path.strip()

    logger.info(
        f"Request to duplicate config from alias '{source_alias}' to external_configs path '{dest_rel_path}'"
    )

    if not dest_rel_path or dest_rel_path == ".":
        raise HTTPException(
            status_code=400, detail="Invalid destination relative path provided."
        )

    # 1. Load super_config aliases
    # Use the PATH_ALIASES loaded at startup
    if source_alias not in PATH_ALIASES:
        logger.warning(
            f"Source alias '{source_alias}' not found in super_config path_aliases."
        )
        raise HTTPException(
            status_code=404,
            detail=f"Source alias '{source_alias}' not found in configuration.",
        )

    original_path_value = PATH_ALIASES[source_alias]
    if not isinstance(
        original_path_value, str
    ) or not original_path_value.lower().endswith(".yaml"):
        logger.warning(
            f"Source alias '{source_alias}' maps to '{original_path_value}', which is not a YAML file path."
        )
        raise HTTPException(
            status_code=400,
            detail=f"Source alias '{source_alias}' does not point to a configuration file (.yaml).",
        )

    # 2. Resolve the source config path using resolve_path
    try:
        # Need to pass the aliases to resolve_path
        resolved_path_str = resolve_path(source_alias, PATH_ALIASES)
        logger.info(
            f"Resolved source alias '{source_alias}' to string path: {resolved_path_str}"
        )

        # --- FIX: Convert the resolved string path to a Path object ---
        try:
            source_config_path = PyPath(
                resolved_path_str
            ).resolve()  # Resolve to make absolute and clean
            logger.info(
                f"Converted and resolved string path to Path object: {source_config_path}"
            )
        except Exception as path_conversion_e:
            logger.error(
                f"Could not convert resolved path string '{resolved_path_str}' to a Path object: {path_conversion_e}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error processing resolved source path for alias '{source_alias}'.",
            )

        if not source_config_path.is_file():
            logger.error(f"Resolved path is not a file: {source_config_path}")
            raise FileNotFoundError(
                f"Resolved path is not a file: {source_config_path}"
            )
    except FileNotFoundError as e:
        logger.error(f"Source config file not found for alias '{source_alias}': {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Source config file for alias '{source_alias}' not found at expected path: {e}",
        )
    except (
        ValueError
    ) as e:  # Catch errors from resolve_path (e.g., alias loop, non-existent intermediate alias)
        logger.error(f"Error resolving source alias '{source_alias}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error resolving source alias '{source_alias}': {e}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error resolving source path for alias '{source_alias}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Unexpected error resolving source path: {e}"
        )

    # 3. Resolve the destination path safely within external_configs
    try:
        dest_path = get_safe_path(CONFIGS_DIR, dest_rel_path)
    except HTTPException:  # Re-raise безпеки exceptions from get_safe_path
        raise
    except Exception as e:
        logger.error(
            f"Error resolving destination path '{dest_rel_path}' in {CONFIGS_DIR}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=400, detail=f"Invalid destination path: {e}")

    # Prevent writing directly to the root configs dir
    if dest_path == CONFIGS_DIR.resolve():
        raise HTTPException(
            status_code=400, detail="Destination cannot be the root configs directory."
        )

    # 4. Check if destination exists
    if dest_path.exists():
        logger.warning(
            f"Destination path '{dest_path}' already exists. Duplicate operation aborted."
        )
        raise HTTPException(
            status_code=409,
            detail=f"Destination path '{dest_rel_path}' already exists.",
        )

    # 5. Check if destination parent directory exists
    dest_parent = dest_path.parent
    if not dest_parent.exists() or not dest_parent.is_dir():
        parent_rel_path = dest_parent.relative_to(CONFIGS_DIR.resolve())
        logger.warning(
            f"Parent directory '{parent_rel_path}' for destination does not exist."
        )
        raise HTTPException(
            status_code=400,
            detail=f"Parent directory '{parent_rel_path}' for destination does not exist.",
        )

    # 6. Read source and write to destination
    try:
        logger.info(f"Copying config from '{source_config_path}' to '{dest_path}'")
        content = source_config_path.read_text(encoding="utf-8")
        dest_path.write_text(content, encoding="utf-8")
        logger.info(f"Successfully duplicated config to '{dest_path}'")
        return {
            "message": f"Successfully duplicated config from alias '{source_alias}' to '{dest_rel_path}' in external_configs."
        }
    except Exception as e:
        logger.error(
            f"Failed to read source or write destination config during duplication: {e}",
            exc_info=True,
        )
        # Clean up destination file if partially created?
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=500, detail=f"Failed to duplicate config file: {e}"
        )


# Add other endpoints (file upload/download, etc.) as needed.

# I SHOULD ADD AN EXTERNAL_PROMPTS/ FOLDER. You pick a pipeline to take the prompts of, it duplicates those prompts and puts a copy of them in external prompts, to serve as a basis for modification... wait no but, prompts are relative to the pipeline code, that would not work as stated...
# hmm...


# OK so there is a question surrounding configs.
# We want to co-locate but that does not really work with the API?
# Simple. Co-located configs are a treat for people using the CLI. There will e all those configs in the externalk configs folder for access too, and one will be able to use them, DUPLICATE THEM IN THE API, CREATE NEW ONES FOR A PIPELINE GIVEN THE TEMPLATE (template = config.yaml in the folder of that pipeline alias by default if it exists) and create folders and move things around input or output management style.
# We need a new route for creating a config file with a given name, for a given pipeline alias. We can reuse existing POST to create a new empty config in the configs folder. And config delete etc. will be... just added to the interface like norma. Making a folder in the config folder for organization may have to be a new route as well.
# So two new routes, a use for the configs inside the generation folder, and a solution that makes both API *AND* CLI happy.
# we also need a route to get the config that a pipeline is beingrun with. And for that we need to store the pipeline in redis. Run_pipeline_task specifically needs to store the parameters in redis associated with the task ID so that we can make a route to get the parameters based on task ID
