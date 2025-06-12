import os


def cleanup_dir(output_dir):
    # Walk through directory and remove all .lock files
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".lock"):
                lock_file = os.path.join(root, file)
                try:
                    os.remove(lock_file)
                except OSError:
                    pass  # Ignore errors if file can't be removed
