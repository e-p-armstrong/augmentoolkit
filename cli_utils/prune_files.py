#!/usr/bin/env python3

import os
import random
import argparse
from pathlib import Path
import sys


def get_size_readable(size_bytes):
    """Convert size in bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0 or unit == "GB":
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0


def prune_folder(folder_path, file_ext=".jsonl", fraction=1 / 3):
    """Randomly delete a fraction of files with given extension in folder"""
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Get all files with the specified extension
    files = list(folder.glob(f"*{file_ext}"))

    if not files:
        print(f"No {file_ext} files found in {folder_path}")
        return

    # Calculate how many files to delete
    num_to_delete = int(len(files) * fraction)

    if num_to_delete == 0:
        print(f"No files to delete (too few files in folder)")
        return

    # Randomly select files to delete
    files_to_delete = random.sample(files, num_to_delete)

    # Calculate total size to be freed
    total_size = sum(os.path.getsize(f) for f in files_to_delete)
    readable_size = get_size_readable(total_size)

    # Show preview
    print(f"\nFound {len(files)} {file_ext} files in {folder_path}")
    print(f"Will delete {num_to_delete} files ({readable_size})")
    print("\nFiles to be deleted:")
    for i, file in enumerate(files_to_delete[:10], 1):
        file_size = get_size_readable(os.path.getsize(file))
        print(f"  {i}. {file.name} ({file_size})")

    if len(files_to_delete) > 10:
        print(f"  ... and {len(files_to_delete) - 10} more files")

    # Ask for confirmation
    print("\n⚠️  WARNING: This operation cannot be undone! ⚠️")
    confirmation = input(f"\nDelete {num_to_delete} files? (yes/no): ")

    if confirmation.lower() != "yes":
        print("Operation cancelled")
        return

    # Delete files
    deleted_count = 0
    deleted_size = 0

    for file in files_to_delete:
        try:
            file_size = os.path.getsize(file)
            os.remove(file)
            deleted_count += 1
            deleted_size += file_size
            sys.stdout.write(f"\rDeleted {deleted_count}/{num_to_delete} files...")
            sys.stdout.flush()
        except Exception as e:
            print(f"\nError deleting {file.name}: {e}")

    print(
        f"\n\nSuccessfully deleted {deleted_count} files ({get_size_readable(deleted_size)})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly delete a fraction of files in a folder"
    )
    parser.add_argument("folder", help="Path to the folder containing files to prune")
    parser.add_argument(
        "--ext", default=".jsonl", help="File extension to target (default: .jsonl)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1 / 3,
        help="Fraction of files to delete (default: 1/3)",
    )

    args = parser.parse_args()
    prune_folder(args.folder, args.ext, args.fraction)
