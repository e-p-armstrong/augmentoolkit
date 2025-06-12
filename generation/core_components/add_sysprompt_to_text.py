import json
import copy
import os
from typing import List, Dict, Any, Union
import glob

# Function requirements:
# version that takes a list of lists of dicts as well as a string (the instruction) as arguments;
# each sublsit of dicts is in the following structure:
# {"conversations": [{"from": "system", "value": "system prompt"}, {"from": "human", "value": "message"}, {"from": "gpt", "value": "message"}, {"from": ...}
# the goal is to: if the first message is from system, we want to add the instruction to the start of it.
# if there is no system message, create it and add the instruction to it.
# Do this for each sublist of dicts. Return the new dicts.


def _add_sysprompt_to_conversation(
    conversation: List[Dict[str, str]], instruction: str
) -> List[Dict[str, str]]:
    """Adds or prepends a system instruction to a single conversation."""
    # Create a deep copy to avoid modifying the original list elements
    mod_conversation = copy.deepcopy(conversation)

    if not instruction:  # Avoid adding empty instructions
        return mod_conversation

    if not mod_conversation:  # Handle empty conversation list
        mod_conversation.append({"from": "system", "value": instruction})
    elif mod_conversation[0].get("from") == "system":
        # Prepend instruction to existing system message
        current_value = mod_conversation[0].get("value", "")
        mod_conversation[0]["value"] = f"{instruction}\n{current_value}"
    else:
        # Insert new system message at the beginning
        mod_conversation.insert(0, {"from": "system", "value": instruction})
    return mod_conversation


def add_sysprompt_to_conversations_list(
    conversations_list: List[List[Dict[str, str]]], instruction: str
) -> List[List[Dict[str, str]]]:
    """
    Adds a system prompt instruction to each conversation in a list of conversations.

    Args:
        conversations_list: A list where each element is a conversation,
                            represented as a list of message dictionaries
                            (e.g., [{"from": "human", "value": "..."}, ...]).
        instruction: The system prompt instruction string to add.

    Returns:
        A new list containing the modified conversations.
    """
    if not instruction:
        return copy.deepcopy(conversations_list)  # Return a copy even if no changes

    modified_conversations = []
    for conversation in conversations_list:
        modified_conv = _add_sysprompt_to_conversation(conversation, instruction)
        modified_conversations.append(modified_conv)
    return modified_conversations


def add_sysprompt_to_files(
    file_paths: List[str], instruction: str
) -> List[List[Dict[str, str]]]:
    """
    Reads conversations from JSON or JSONL files, adds a system prompt instruction,
    and returns the modified conversations.

    Assumes each file (or line in JSONL) contains data that yields one or more
    conversations, where each conversation is a list of message dictionaries.
    Specifically looks for a "conversations" key within JSON objects.

    Args:
        file_paths: A list of paths to JSON or JSONL files.
        instruction: The system prompt instruction string to add.

    Returns:
        A list containing all modified conversations read from the files.

    Raises:
        FileNotFoundError: If a specified file path does not exist.
        ValueError: If a file is not valid JSON/JSONL or lacks the expected structure.
    """
    all_extracted_conversations: List[List[Dict[str, str]]] = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File not found at {file_path}")

        try:
            if file_path.lower().endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                conversation = data.get("conversations")
                                if isinstance(conversation, list):
                                    # Basic validation of conversation structure
                                    if all(
                                        isinstance(msg, dict)
                                        and "from" in msg
                                        and "value" in msg
                                        for msg in conversation
                                    ):
                                        all_extracted_conversations.append(conversation)
                                    else:
                                        print(
                                            f"Warning: Skipping invalid conversation structure in {file_path}, line {i+1}"
                                        )
                                else:
                                    print(
                                        f"Warning: 'conversations' key in {file_path}, line {i+1} is not a list. Skipping."
                                    )
                            else:
                                print(
                                    f"Warning: JSONL line {i+1} in {file_path} is not a dictionary. Skipping."
                                )
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Error decoding JSONL line {i+1} in {file_path}: {e}"
                            ) from e

            elif file_path.lower().endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        content = json.load(f)
                        # Handle different possible JSON structures
                        conversations_to_process = []
                        if isinstance(content, list):
                            # Assume list of records, each containing "conversations"
                            for item in content:
                                if isinstance(item, dict):
                                    conversation = item.get("conversations")
                                    if isinstance(conversation, list):
                                        conversations_to_process.append(conversation)
                                    else:
                                        print(
                                            f"Warning: 'conversations' key within list item in {file_path} is not a list. Skipping item."
                                        )
                                else:
                                    print(
                                        f"Warning: Item in list in {file_path} is not a dictionary. Skipping item."
                                    )
                        elif isinstance(content, dict):
                            # Assume dict containing "conversations" key which holds ONE conversation list
                            conversation = content.get("conversations")
                            if isinstance(conversation, list):
                                conversations_to_process.append(conversation)
                            else:
                                # OR assume dict containing a key (e.g., "data") which holds a LIST of conversations
                                found_list = False
                                for key, value in content.items():
                                    if isinstance(value, list):
                                        # Check if this list looks like a list of conversations or list of records
                                        if all(
                                            isinstance(item, list)
                                            and len(item) > 0
                                            and isinstance(item[0], dict)
                                            and "from" in item[0]
                                            for item in value
                                        ):  # List of conversations?
                                            conversations_to_process.extend(value)
                                            found_list = True
                                            print(
                                                f"Info: Found list of conversations under key '{key}' in {file_path}"
                                            )
                                            break
                                        elif all(
                                            isinstance(item, dict) for item in value
                                        ):  # List of records?
                                            processed_records = False
                                            for item_dict in value:
                                                conv = item_dict.get("conversations")
                                                if isinstance(conv, list):
                                                    conversations_to_process.append(
                                                        conv
                                                    )
                                                    processed_records = True
                                            if processed_records:
                                                found_list = True
                                                print(
                                                    f"Info: Found list of records containing conversations under key '{key}' in {file_path}"
                                                )
                                                break
                                if not found_list:
                                    print(
                                        f"Warning: Could not find a recognizable conversation list structure in dictionary in {file_path}. Skipping file."
                                    )
                        else:
                            print(
                                f"Warning: Unexpected JSON structure (not list or dict) in {file_path}. Skipping file."
                            )

                        # Validate and add extracted conversations
                        for conv in conversations_to_process:
                            # Check if conv is a non-empty list where all elements are dictionaries with 'from' and 'value' keys
                            if isinstance(conv, list) and all(
                                isinstance(msg, dict)
                                and "from" in msg
                                and "value" in msg
                                for msg in conv
                            ):
                                all_extracted_conversations.append(conv)
                            elif (
                                isinstance(conv, list) and not conv
                            ):  # Allow empty conversations
                                all_extracted_conversations.append(conv)
                            else:
                                print(
                                    f"Warning: Skipping invalid or empty conversation structure found in {file_path}"
                                )

                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Error decoding JSON file {file_path}: {e}"
                        ) from e
            else:
                print(
                    f"Warning: Skipping file {file_path} - extension not .json or .jsonl"
                )

        except Exception as e:
            # Catch other potential errors during file processing
            print(f"Error processing file {file_path}: {e}")
            # Optionally re-raise or continue if specific errors should be ignored
            raise  # Re-raise by default

    # Now apply the instruction to all collected conversations
    modified_conversations = add_sysprompt_to_conversations_list(
        all_extracted_conversations, instruction
    )
    return modified_conversations


def add_sysprompt_to_directory_files(directory_path: str, instruction: str) -> None:
    """
    Adds a system prompt instruction to all conversations in .json and .jsonl
    files within a specified directory, overwriting the original files.

    Args:
        directory_path: The path to the directory containing the files.
        instruction: The system prompt instruction string to add.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If errors occur during file processing or writing.
    """
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(f"Error: Directory not found at {directory_path}")

    if not instruction:  # Don't modify files if instruction is empty
        print("Warning: No instruction provided. Skipping sysprompt addition.")
        return

    file_paths = glob.glob(os.path.join(directory_path, "*.jsonl")) + glob.glob(
        os.path.join(directory_path, "*.json")
    )

    if not file_paths:
        print(f"Warning: No .json or .jsonl files found in {directory_path}")
        return

    print(f"Adding instruction to {len(file_paths)} files in {directory_path}...")
    for file_path in file_paths:
        try:
            # Use existing function to read/process conversations from this single file
            # Note: add_sysprompt_to_files already handles internal errors like JSONDecodeError
            # and returns potentially partial results if some lines/structures fail.
            # It raises FileNotFoundError if the specific file isn't found (handled by glob already).
            modified_conversations = add_sysprompt_to_files([file_path], instruction)

            # Overwrite the original file
            if file_path.lower().endswith(".jsonl"):
                with open(file_path, "w", encoding="utf-8") as f:
                    for conv in modified_conversations:
                        # Write each conversation as a JSON object on its own line
                        json.dump({"conversations": conv}, f)
                        f.write("\n")
            elif file_path.lower().endswith(".json"):
                with open(file_path, "w", encoding="utf-8") as f:
                    # Write as a list of records, where each record has a "conversations" key
                    output_data = [
                        {"conversations": conv} for conv in modified_conversations
                    ]
                    json.dump(output_data, f, indent=2)  # Use indent for readability

        except ValueError as e:
            # Catch errors specifically from add_sysprompt_to_files or json writing
            print(
                f"Error processing or writing file {file_path}: {e}. Skipping this file."
            )
            # Continue to the next file
        except Exception as e:
            # Catch any other unexpected errors
            print(
                f"Unexpected error processing file {file_path}: {e}. Skipping this file."
            )
            # Continue to the next file
    print(f"Finished adding instruction to files in {directory_path}.")
