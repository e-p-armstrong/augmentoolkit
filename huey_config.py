import os

# Import RedisHuey and the necessary connection details from redis_config
from huey import RedisHuey
from redis_config import REDIS_HOST, REDIS_PORT, REDIS_DB

# Define the Huey instance using Redis
# The 'name' parameter is optional but good practice, often set to the app name.
# It prefixes keys used by Huey in Redis to avoid collisions if Redis is shared.
huey = RedisHuey(
    name="augmentoolkit_tasks",  # Optional, helps namespace Huey keys in Redis
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    # Add other RedisHuey parameters if needed (e.g., password, connection_pool)
)

# Remove SQLite specific code
# os.makedirs("huey")
# _db_path = os.path.join(os.path.dirname(__file__), 'huey', 'huey_queue.db')
# huey = SqliteHuey(filename=_db_path)
