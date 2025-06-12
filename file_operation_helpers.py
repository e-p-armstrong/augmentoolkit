# tentative helpers file, to be imported by api.py
import os
import shutil
import zipfile
from pathlib import Path as PyPath
from typing import Optional, List
import logging

from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Models (for get_dir_structure) ---
class FileStructure(BaseModel):
    path: str
    is_dir: bool
    children: Optional[List["FileStructure"]] = None


# --- Pydantic Model (for move operation) ---
class MoveItemRequest(BaseModel):
    source_relative_path: str = Field(
        ..., description="The current relative path of the item to move."
    )
    destination_relative_path: str = Field(
        ..., description="The desired new relative path for the item."
    )


# --- Helper Functions ---


def get_safe_path(base_dir: PyPath, unsafe_path: str) -> PyPath:
    """Safely join a base directory with a user-provided path, preventing traversal."""
    # print(f"DEBUG [get_safe_path]: Received base_dir='{base_dir}', unsafe_path='{unsafe_path}'")
    try:
        # Normalize the path to resolve '..' etc.
        joined_path = base_dir / unsafe_path
        # print(f"DEBUG [get_safe_path]: Joined path='{joined_path}'")
        resolved_path = joined_path.resolve()
        # print(f"DEBUG [get_safe_path]: Resolved path='{resolved_path}'")
        base_dir_resolved = base_dir.resolve()
        # print(f"DEBUG [get_safe_path]: Resolved base_dir='{base_dir_resolved}'")

        # Ensure the final path is within the base_dir
        # Check if resolved_path starts with base_dir resolved path
        is_safe = str(resolved_path).startswith(str(base_dir_resolved))
        # print(f"DEBUG [get_safe_path]: Checking safety: '{resolved_path}' startswith '{base_dir_resolved}' -> {is_safe}")
        if not is_safe:
            # print(f"ERROR [get_safe_path]: Path traversal attempt detected!")
            raise HTTPException(status_code=400, detail="Invalid path: Access denied.")
        # print(f"DEBUG [get_safe_path]: Path deemed safe. Returning '{resolved_path}'")
        return resolved_path
    except Exception as e:
        # Catch potential resolution errors or permission issues during resolve()
        # print(f"ERROR [get_safe_path]: Exception during path resolution/validation for unsafe_path='{unsafe_path}' in base_dir='{base_dir}': {e}")
        logger.error(
            f"Path resolution/validation error for {unsafe_path} in {base_dir}: {e}"
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid or inaccessible path: {unsafe_path}"
        )


def zip_directory(folder_path: PyPath, zip_path: PyPath):
    """Compresses a directory into a zip file."""
    # print(f"DEBUG [zip_directory]: Zipping '{folder_path}' to '{zip_path}'") # Added basic debug print
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Calculate the archive name relative to the folder being zipped
            current_dir_path = PyPath(root)
            relative_dir = current_dir_path.relative_to(folder_path)
            # print(f"DEBUG [zip_directory]: Processing dir '{relative_dir}', files: {len(files)}") # Optional verbose print
            for file in files:
                file_path = current_dir_path / file
                archive_name = relative_dir / file
                zipf.write(file_path, archive_name)
            # Add empty directories too?
            # for dir_name in dirs:
            #     if not os.listdir(os.path.join(root, dir_name)): # Check if dir is empty
            #         dir_path = current_dir_path / dir_name
            #         archive_name = relative_dir / dir_name
            #         # Add an entry for the empty directory
            #         zipf.write(dir_path, archive_name)
    # print(f"DEBUG [zip_directory]: Finished zipping '{folder_path}'") # Added basic debug print


def get_dir_structure(
    dir_path: PyPath, base_path_for_relative: Optional[PyPath] = None
) -> List[FileStructure]:
    """Recursively gets the directory structure relative to a base path."""
    # print(f"DEBUG [get_dir_structure]: Called with dir_path='{dir_path}', base_path_for_relative='{base_path_for_relative}'")
    structure = []
    if base_path_for_relative is None:
        # print(f"DEBUG [get_dir_structure]: base_path_for_relative was None, setting to dir_path='{dir_path}'")
        base_path_for_relative = dir_path  # Make paths relative to the starting dir

    try:
        items_iterator = list(dir_path.iterdir())  # Convert to list to see count
        # print(f"DEBUG [get_dir_structure]: Found {len(items_iterator)} items in '{dir_path}'")
        for item in sorted(items_iterator):  # Sort for consistent output
            # print(f"DEBUG [get_dir_structure]: Processing item: '{item}'")
            is_dir = item.is_dir()
            # print(f"DEBUG [get_dir_structure]:   Is directory? {is_dir}")
            # Get path relative to the initial base directory we started from
            try:
                relative_item_path = str(item.relative_to(base_path_for_relative))
                # print(f"DEBUG [get_dir_structure]:   Relative path: '{relative_item_path}' (relative to '{base_path_for_relative}')")
            except ValueError as rel_e:
                # print(f"ERROR [get_dir_structure]: Could not make '{item}' relative to '{base_path_for_relative}': {rel_e}")
                # Log this error instead of printing, it might be important
                logger.error(
                    f"Could not make path '{item}' relative to base '{base_path_for_relative}': {rel_e}"
                )
                # Skip this item or use a placeholder if needed, depends on desired behavior
                continue  # Skip this item
                # relative_item_path = f"ERROR_RELATIVE_PATH_{item.name}" # Fallback?

            node = FileStructure(path=relative_item_path, is_dir=is_dir)
            if is_dir:
                # Pass down the original base path for consistent relativity
                # print(f"DEBUG [get_dir_structure]:   Recursing into directory '{item}'...")
                node.children = get_dir_structure(item, base_path_for_relative)
            structure.append(node)
        # print(f"DEBUG [get_dir_structure]: Finished processing '{dir_path}'. Returning structure with {len(structure)} items.")
        return structure
    except FileNotFoundError:
        # This shouldn't happen if called on an existing dir, but handle defensively
        # print(f"ERROR [get_dir_structure]: FileNotFoundError encountered for dir_path='{dir_path}'")
        return []
    except PermissionError as pe:
        # print(f"ERROR [get_dir_structure]: PermissionError accessing dir_path='{dir_path}': {pe}")
        logger.error(
            f"Permission error listing directory {dir_path}: {pe}", exc_info=True
        )
        return []
    except Exception as e:
        # print(f"ERROR [get_dir_structure]: Generic exception listing directory dir_path='{dir_path}': {e}")
        logger.error(f"Error listing directory {dir_path}: {e}", exc_info=True)
        # Depending on desired behaviour, could raise or return empty
        return []


# --- Refactored Generic Handlers ---


def handle_get_structure(base_dir: PyPath, relative_path: str) -> List[FileStructure]:
    """Handles the logic for getting directory structure safely."""
    # --- FIX: Resolve base_dir to absolute path early ---
    try:
        abs_base_dir = base_dir.resolve()
    except Exception as resolve_e:
        # print(f"ERROR [handle_get_structure]: Could not resolve base_dir '{base_dir}': {resolve_e}")
        logger.error(f"Could not resolve base directory '{base_dir}': {resolve_e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve base directory '{base_dir.name}'",
        )
    # print(f"DEBUG [handle_get_structure]: Received base_dir='{base_dir}' (resolved to '{abs_base_dir}'), relative_path='{relative_path}'")
    # --- End Fix ---

    logger.info(
        f"Request for structure in '{abs_base_dir.name}' at relative path: {relative_path}"
    )
    target_path_str = relative_path if relative_path else "."
    # print(f"DEBUG [handle_get_structure]: Calculated target_path_str='{target_path_str}'")

    # --- FIX: Use resolved absolute base dir ---
    target_dir = get_safe_path(abs_base_dir, target_path_str)
    # print(f"DEBUG [handle_get_structure]: Safe path result: '{target_dir}'")
    # --- End Fix ---

    exists = target_dir.exists()
    # print(f"DEBUG [handle_get_structure]: Does target_dir exist? {exists}")
    if not exists:
        # print(f"WARN [handle_get_structure]: Target path not found: '{target_dir}'")
        logger.warning(f"Target path not found in '{abs_base_dir.name}': {target_dir}")
        raise HTTPException(
            status_code=404,
            detail=f"Path '{relative_path}' not found within {abs_base_dir.name} directory.",
        )

    is_dir = target_dir.is_dir()
    # print(f"DEBUG [handle_get_structure]: Is target_dir a directory? {is_dir}")
    if not is_dir:
        # print(f"WARN [handle_get_structure]: Target path is not a directory: '{target_dir}'")
        logger.warning(
            f"Target path in '{abs_base_dir.name}' is not a directory: {target_dir}"
        )
        raise HTTPException(
            status_code=400, detail=f"Path '{relative_path}' is not a directory."
        )

    # Pass resolved abs_base_dir as the base for consistent relative paths in the response
    # --- FIX: Pass resolved absolute base dir ---
    # print(f"DEBUG [handle_get_structure]: Calling get_dir_structure with target_dir='{target_dir}', base_path_for_relative='{abs_base_dir}'")
    structure = get_dir_structure(target_dir, base_path_for_relative=abs_base_dir)
    # --- End Fix ---
    # print(f"DEBUG [handle_get_structure]: get_dir_structure returned {len(structure)} items. Returning result.")
    # print(structure)
    return structure


def handle_download_item(base_dir: PyPath, relative_path: str) -> FileResponse:
    """Handles the logic for downloading a file or zipped directory safely."""
    # --- Apply similar fix for consistency ---
    try:
        abs_base_dir = base_dir.resolve()
    except Exception as resolve_e:
        # print(f"ERROR [handle_download_item]: Could not resolve base_dir '{base_dir}': {resolve_e}")
        logger.error(f"Could not resolve base directory '{base_dir}': {resolve_e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve base directory '{base_dir.name}'",
        )
    # print(f"DEBUG [handle_download_item]: Received base_dir='{base_dir}' (resolved to '{abs_base_dir}'), relative_path='{relative_path}'")
    # --- End Fix ---

    logger.info(
        f"Request to download item from '{abs_base_dir.name}' at relative path: {relative_path}"
    )
    target_path = get_safe_path(abs_base_dir, relative_path)
    # print(f"DEBUG [handle_download_item]: Safe path result: '{target_path}'")

    exists = target_path.exists()
    # print(f"DEBUG [handle_download_item]: Target path exists? {exists}")
    if not exists:
        logger.warning(
            f"Target path not found for download in '{abs_base_dir.name}': {target_path}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Path '{relative_path}' not found within {abs_base_dir.name} directory.",
        )

    download_filename = target_path.name

    is_file = target_path.is_file()
    is_dir = target_path.is_dir()
    # print(f"DEBUG [handle_download_item]: Is file? {is_file}, Is dir? {is_dir}")

    if is_file:
        logger.info(
            f"Path in '{abs_base_dir.name}' is a file, preparing FileResponse for {target_path}"
        )
        # TODO: Consider adding media_type detection based on file extension
        return FileResponse(target_path, filename=download_filename)
    elif is_dir:
        logger.info(
            f"Path in '{abs_base_dir.name}' is a directory, preparing zip archive for {target_path}"
        )
        zip_filename = f"{download_filename}.zip"
        # Use tempfile module for more robust temp file creation if needed
        temp_zip_path = PyPath(f"/tmp/{zip_filename}")
        # print(f"DEBUG [handle_download_item]: Temp zip path: '{temp_zip_path}'")
        try:
            zip_directory(target_path, temp_zip_path)
            logger.info(
                f"Zip file created for directory {relative_path} in '{abs_base_dir.name}'"
            )
            background = BackgroundTasks()
            background.add_task(os.remove, temp_zip_path)
            return FileResponse(
                temp_zip_path,
                media_type="application/zip",
                filename=zip_filename,
                background=background,
            )
        except Exception as e:
            logger.error(
                f"Failed to create or send zip file for directory {relative_path} in '{abs_base_dir.name}': {e}",
                exc_info=True,
            )
            if temp_zip_path.exists():
                try:
                    os.remove(temp_zip_path)
                except Exception as cleanup_e:
                    logger.error(
                        f"Error cleaning temp zip {temp_zip_path}: {cleanup_e}"
                    )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create zip archive for directory: {e}",
            )
    else:
        logger.error(
            f"Target path exists in '{abs_base_dir.name}' but is neither file nor directory: {target_path}"
        )
        raise HTTPException(status_code=500, detail="Target path type is unsupported.")


def handle_delete_item(base_dir: PyPath, relative_path: str) -> dict:
    """Handles the logic for deleting a file or folder safely."""
    # --- Apply similar fix for consistency ---
    try:
        abs_base_dir = base_dir.resolve()
    except Exception as resolve_e:
        # print(f"ERROR [handle_delete_item]: Could not resolve base_dir '{base_dir}': {resolve_e}")
        logger.error(f"Could not resolve base directory '{base_dir}': {resolve_e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve base directory '{base_dir.name}'",
        )
    # print(f"DEBUG [handle_delete_item]: Received base_dir='{base_dir}' (resolved to '{abs_base_dir}'), relative_path='{relative_path}'")
    # --- End Fix ---

    logger.info(
        f"Request to delete item from '{abs_base_dir.name}' at relative path: {relative_path}"
    )
    target_path = get_safe_path(abs_base_dir, relative_path)
    # print(f"DEBUG [handle_delete_item]: Safe path result: '{target_path}'")

    # Basic check: Do not allow deleting the root dir itself via this route
    resolved_base = abs_base_dir  # Already resolved
    # print(f"DEBUG [handle_delete_item]: Checking if target '{target_path}' == resolved base '{resolved_base}'")
    if target_path == resolved_base:
        logger.warning(
            f"Attempt to delete root '{abs_base_dir.name}' directory blocked for path: {relative_path}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete the root {abs_base_dir.name} directory via this endpoint.",
        )

    exists = target_path.exists()
    # print(f"DEBUG [handle_delete_item]: Target path exists? {exists}")
    if not exists:
        logger.warning(
            f"Target path not found for deletion in '{abs_base_dir.name}': {target_path}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Path '{relative_path}' not found within {abs_base_dir.name} directory.",
        )

    try:
        item_type = "Unknown"
        is_file = target_path.is_file()
        is_dir = target_path.is_dir()
        # print(f"DEBUG [handle_delete_item]: Is file? {is_file}, Is dir? {is_dir}")
        if is_file:
            item_type = "file"
            # print(f"DEBUG [handle_delete_item]: Deleting file: '{target_path}'")
            target_path.unlink()
        elif is_dir:
            item_type = "directory"
            # print(f"DEBUG [handle_delete_item]: Deleting directory recursively: '{target_path}'")
            shutil.rmtree(target_path)
        else:
            # This case should be rare if exists() is true, but handle defensively
            raise ValueError(
                f"Target path '{target_path}' exists but is neither a file nor a directory."
            )

        logger.info(
            f"Successfully deleted {item_type} from '{abs_base_dir.name}': {target_path}"
        )
        return {
            "message": f"Successfully deleted {item_type} '{relative_path}' from {abs_base_dir.name}."
        }

    except Exception as e:
        # print(f"ERROR [handle_delete_item]: Exception during deletion of '{target_path}': {e}")
        logger.error(
            f"Failed to delete item {target_path} from '{abs_base_dir.name}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete '{relative_path}' from {abs_base_dir.name}: {e}",
        )


def handle_move_item(
    base_dir: PyPath, source_rel_path: str, dest_rel_path: str
) -> dict:
    """Handles the logic for moving/renaming a file or folder safely within a base directory."""
    try:
        abs_base_dir = base_dir.resolve()
    except Exception as resolve_e:
        logger.error(f"Could not resolve base directory '{base_dir}': {resolve_e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve base directory '{base_dir.name}'",
        )

    logger.info(
        f"Request to move item in '{abs_base_dir.name}' from '{source_rel_path}' to '{dest_rel_path}'"
    )

    # Sanitize inputs
    source_rel_path = source_rel_path.strip()
    dest_rel_path = dest_rel_path.strip()

    if not source_rel_path or source_rel_path == ".":
        raise HTTPException(status_code=400, detail="Invalid source path.")
    if not dest_rel_path or dest_rel_path == ".":
        raise HTTPException(status_code=400, detail="Invalid destination path.")

    # Resolve source and destination paths safely
    source_path = get_safe_path(abs_base_dir, source_rel_path)
    dest_path = get_safe_path(abs_base_dir, dest_rel_path)

    # --- Pre-move Checks ---
    # 1. Source must exist
    if not source_path.exists():
        logger.warning(f"Source path not found for move: {source_path}")
        raise HTTPException(
            status_code=404, detail=f"Source path '{source_rel_path}' not found."
        )

    # 2. Destination must NOT exist (prevent overwrite)
    if dest_path.exists():
        logger.warning(f"Destination path already exists, move blocked: {dest_path}")
        raise HTTPException(
            status_code=409,
            detail=f"Destination path '{dest_rel_path}' already exists.",
        )

    # 3. Parent directory of destination must exist
    dest_parent = dest_path.parent
    if not dest_parent.exists() or not dest_parent.is_dir():
        logger.warning(
            f"Parent directory for destination does not exist or is not a directory: {dest_parent}"
        )
        # Construct a relative path for the parent for the error message
        try:
            dest_parent_rel = dest_parent.relative_to(abs_base_dir)
        except ValueError:
            dest_parent_rel = dest_parent  # Fallback to absolute if somehow outside
        raise HTTPException(
            status_code=400,
            detail=f"Parent directory for destination '{dest_parent_rel}' does not exist.",
        )

    # 4. Cannot move the root directory itself (source or destination)
    if source_path == abs_base_dir:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot move the root {abs_base_dir.name} directory.",
        )
    if dest_path == abs_base_dir:
        # Should be caught by dest_path.exists() check already, but double-check
        raise HTTPException(
            status_code=400,
            detail=f"Destination cannot be the root {abs_base_dir.name} directory.",
        )

    # --- Perform Move --- #
    try:
        logger.info(f"Moving '{source_path}' to '{dest_path}'")
        shutil.move(str(source_path), str(dest_path))
        item_type = (
            "directory" if dest_path.is_dir() else "file"
        )  # Check type *after* move
        logger.info(
            f"Successfully moved {item_type} from '{source_rel_path}' to '{dest_rel_path}' in '{abs_base_dir.name}'"
        )
        return {
            "message": f"Successfully moved {item_type} from '{source_rel_path}' to '{dest_rel_path}'."
        }
    except Exception as e:
        logger.error(
            f"Failed to move '{source_path}' to '{dest_path}': {e}", exc_info=True
        )
        # Try to determine if source still exists to give better context
        status_info = "Source may or may not exist at original location."
        if source_path.exists():
            status_info = "Source still exists at original location."
        elif dest_path.exists():
            status_info = "Destination might exist (partially moved?)."  # Less likely with shutil.move

        raise HTTPException(
            status_code=500,
            detail=f"Failed to move '{source_rel_path}' to '{dest_rel_path}': {e}. {status_info}",
        )


# --- New Helper Function for Creating Directories ---
def handle_create_directory(base_dir: PyPath, relative_path: str) -> dict:
    """Handles the logic for creating a directory safely within a base directory."""
    try:
        abs_base_dir = base_dir.resolve()
    except Exception as resolve_e:
        logger.error(f"Could not resolve base directory '{base_dir}': {resolve_e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not resolve base directory '{base_dir.name}'",
        )

    relative_path = relative_path.strip()
    if not relative_path or relative_path == ".":
        logger.warning(
            f"Attempt to create directory in '{abs_base_dir.name}' with invalid path: '{relative_path}'"
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid relative path provided. Cannot be empty or '.'",
        )

    logger.info(
        f"Request to create directory in '{abs_base_dir.name}' at: {relative_path}"
    )
    try:
        target_path = get_safe_path(abs_base_dir, relative_path)

        # Prevent creating the root directory itself or paths outside the base
        if target_path == abs_base_dir:
            logger.warning(
                f"Attempt to create root {abs_base_dir.name} directory blocked for path: {relative_path}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Cannot create the root {abs_base_dir.name} directory itself.",
            )

        # Check if parent directory exists
        parent_path = target_path.parent
        if not parent_path.exists() or not parent_path.is_dir():
            parent_rel_path = parent_path.relative_to(abs_base_dir)
            logger.warning(
                f"Parent directory does not exist for requested path: {relative_path} (Parent: {parent_path}) in {abs_base_dir.name}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Parent directory does not exist: '{parent_rel_path}'",
            )

        # Check if target already exists
        if target_path.exists():
            logger.warning(
                f"Attempt to create directory in {abs_base_dir.name} that already exists: {target_path}"
            )
            detail_msg = f"Path already exists: '{relative_path}'"
            if target_path.is_file():
                detail_msg += " (and it's a file)"
            raise HTTPException(status_code=409, detail=detail_msg)

        # Create the directory
        target_path.mkdir()
        logger.info(
            f"Successfully created directory in {abs_base_dir.name}: {target_path}"
        )
        return {
            "message": f"Successfully created directory '{relative_path}' in {abs_base_dir.name}."
        }

    except HTTPException:  # Re-raise specific HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Failed to create directory '{relative_path}' in {abs_base_dir.name}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create directory '{relative_path}': {e}"
        )
