from typing import Union
import redis
import os
import json  # For potentially storing structured data

# Configuration (adjust host, port, db if needed)
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))

# Create the Redis client instance
# decode_responses=True is convenient for getting strings directly
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,  # Automatically decode responses from bytes to strings (usually UTF-8)
    )
    # Test connection
    redis_client.ping()
    print(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
except redis.exceptions.ConnectionError as e:
    print(
        f"ERROR: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB} - {e}"
    )
    print("Progress tracking will not work.")
    # Decide how to handle this - maybe raise an error or provide a dummy client?
    # For now, let it proceed, but progress features will fail.
    redis_client = None  # Or a dummy object that silently fails

# Helper constants/functions (optional)
PROGRESS_KEY_PREFIX = "pipeline_progress:"
DEFAULT_EXPIRY_SECONDS = 86400 * 4  # 2 days


def get_progress_key(task_id: str) -> str:
    return f"{PROGRESS_KEY_PREFIX}{task_id}"


def set_progress(task_id: str, progress: float, message: Union[str, None] = None):
    if not redis_client:
        return  # Skip if connection failed
    if not task_id:
        return  # skip if not run as API
    key = get_progress_key(task_id)

    # Retrieve the existing status if message is None
    if message is None:
        try:
            raw_data = redis_client.get(key)
            if raw_data:
                existing_status = json.loads(raw_data)
                message = existing_status.get("message", "initial message")
            else:
                message = "initial message"
        except Exception as e:
            print(
                f"ERROR: Failed to retrieve existing message from Redis for {task_id}: {e}"
            )
            message = "initial message"

    status = {"progress": progress, "message": message}
    try:
        redis_client.set(key, json.dumps(status), ex=DEFAULT_EXPIRY_SECONDS)
        print(f"Task {task_id}: {progress*100:.2f}% complete - {message}")
    except Exception as e:
        print(f"ERROR: Failed to set progress in Redis for {task_id}: {e}")


def get_progress(
    task_id: str,
) -> (
    dict | None
):  # NOTE we don't need to specify which of the enums the task currently is. That is tracked via huey itself. If this thing HAS a thing in redis it is running, else it is pending. The rest are derivable.
    if not redis_client:
        return None  # Skip if connection failed
    key = get_progress_key(task_id)
    try:
        raw_data = redis_client.get(key)
        if raw_data:
            return json.loads(raw_data)
    except Exception as e:
        print(f"ERROR: Failed to get progress from Redis for {task_id}: {e}")
    return None


# TODO ensure that the start command uses a good-licensed redis
