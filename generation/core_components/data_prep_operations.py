import datasets  # Add this with other importsimport json
import argparse
import random
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
import os
import logging

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ")


def count_tokens(message):
    return len(tokenizer.encode(message))


def count_item_tokens(item, count_all_turns=False):
    if "conversations" in item:
        total = 0
        for conv in item["conversations"]:
            if (
                count_all_turns
                or conv.get("from") == "gpt"
                or conv.get("from") == "assistant"
            ):
                total += count_tokens(conv["value"])
        return total
    else:
        text = item.get("text", "")
        return count_tokens(text)


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df.to_dict("records")


def write_jsonl(file_path, data):
    # Create directories in path if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            converted_item = convert_numpy_types(item)
            json.dump(converted_item, file)
            file.write("\n")


def create_subset(
    input_file,
    output_file,
    target_tokens,
    count_all_turns=False,
    context_to_add: str = None,
    context_to_add_type: str = None,
):
    logging.info(f"Creating subset from {input_file} to {output_file}")
    logging.info(f"Target tokens: {target_tokens}, count_all_turns: {count_all_turns}")

    # Read all data
    if input_file.endswith(".jsonl"):
        logging.info("Reading JSONL file")
        all_data = read_jsonl(input_file)
    elif input_file.endswith(".json"):
        logging.info("Reading JSON file")
        all_data = read_json(input_file)
    elif input_file.endswith(".parquet"):
        logging.info("Reading Parquet file")
        all_data = read_parquet(input_file)
    else:
        msg = "Unsupported file format. Please use .jsonl, .json, or .parquet"
        logging.error(msg)
        raise ValueError(msg)

    logging.info(f"Read {len(all_data)} items from input file")

    # Ensure all_data is a list
    if not isinstance(all_data, list):
        msg = "Input data must be a list of items"
        logging.error(msg)
        raise ValueError(msg)

    subset = []
    current_tokens = 0
    used_indices = set()

    logging.info("Starting random selection of items")
    while current_tokens < target_tokens and len(used_indices) < len(all_data):
        # Find a random unused index
        index = random.randint(0, len(all_data) - 1)
        logging.debug(f"Selected random index {index}")

        if index in used_indices:
            logging.debug(f"Index {index} already used, skipping")
            continue

        item = all_data[index]
        tokens = count_item_tokens(item, count_all_turns)
        logging.debug(f"Item at index {index} has {tokens} tokens")

        subset.append(item)
        current_tokens += tokens
        used_indices.add(index)

        logging.debug(
            f"Current progress: {current_tokens}/{target_tokens} tokens, {len(subset)} items"
        )

    if context_to_add and context_to_add_type:
        for item in subset:
            if context_to_add_type == "system":
                # see if the first message is a system message
                if item["conversations"][0]["from"] == "system":
                    item["conversations"][0]["value"] = (
                        context_to_add + " " + item["conversations"][0]["value"]
                    )
                # otherwise create a new system message
                else:
                    item["conversations"].insert(
                        0, {"from": "system", "value": context_to_add}
                    )
            elif context_to_add_type == "human":
                # see if the first message is a human message
                if item["conversations"][0]["from"] == "human":
                    item["conversations"][0]["value"] = (
                        context_to_add + " " + item["conversations"][0]["value"]
                    )
                # otherwise find the first human message and add the context to it
                else:
                    for conv in item["conversations"]:
                        if conv["from"] == "human":
                            conv["value"] = context_to_add + " " + conv["value"]
                            break

    logging.info(f"Selected {len(subset)} items with total {current_tokens} tokens")

    # Write the subset to the output file
    logging.info(f"Writing subset to {output_file}")
    write_jsonl(output_file, subset)

    logging.info(
        f"Subset creation complete. Final tokens: {current_tokens}, items: {len(subset)}"
    )
    return current_tokens, len(subset)


# TODO make handling of errors in the engine wrapper friendlier


def count_tokens_glob(paths, count_all_turns: bool = False) -> int:
    """
    Count total tokens across multiple files matching glob patterns.

    Args:
        input_dir: Directory path or list of glob patterns to match files
        count_all_turns: Whether to count all conversation turns (False = only assistant turns)

    Returns:
        Total token count across all matched files
    """
    total_tokens = 0
    seen_files = set()

    # Convert single string to list for consistent handling
    patterns = paths if isinstance(paths, list) else [paths]

    for pattern in patterns:
        # If pattern is a directory, create patterns for supported file types
        if os.path.isdir(pattern):
            file_patterns = [
                os.path.join(pattern, "*.json"),
                os.path.join(pattern, "*.jsonl"),
                os.path.join(pattern, "*.parquet"),
            ]
        else:
            file_patterns = [pattern]

        # Process each pattern
        for file_pattern in file_patterns:
            for file_path in glob.glob(file_pattern):
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)

                try:
                    # Read data using existing helpers
                    if file_path.endswith(".jsonl"):
                        data = read_jsonl(file_path)
                    elif file_path.endswith(".json"):
                        data = read_json(file_path)
                    elif file_path.endswith(".parquet"):
                        data = read_parquet(file_path)
                    else:
                        logging.warning(
                            f"Skipping unsupported file format: {file_path}"
                        )
                        continue

                    # Count tokens using existing count_item_tokens
                    file_tokens = sum(
                        count_item_tokens(item, count_all_turns) for item in data
                    )
                    total_tokens += file_tokens
                    logging.info(
                        f"Counted {file_tokens:,} tokens in {os.path.basename(file_path)}"
                    )

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue

    logging.info(f"Total tokens across {len(seen_files)} files: {total_tokens:,}")
    return total_tokens


#!/usr/bin/env python3
import argparse
import json
import glob
import os
from typing import List, Dict, Any
from jinja2 import Template

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def apply_template(messages: List[Dict[str, str]], template_str: str, **kwargs) -> str:
    """
    Format messages using a Jinja2 template string.

    Args:
        messages: List of message dictionaries with 'from' and 'value'
        template_str: Jinja2 template as a string
        **kwargs: Additional variables to pass to template rendering

    Returns:
        Rendered template string
    """
    template = Template(template_str)
    return template.render(messages=messages, **kwargs).strip()


def process_file(input_path: str, template_str: str, **kwargs) -> List[Dict[str, str]]:
    """
    Process a single input file (JSON or JSONL) and convert its conversations
    into a formatted text using the specified template.

    Args:
        input_path: The path to the input file.
        template: The template to use ('chatml' or 'custom').

    Returns:
        A list of dictionaries, each with a 'text' key containing the processed conversation.
    """
    logging.info(f"Processing file: {input_path}")
    data = []
    try:
        # Load the entire file if it ends with .json (assuming it's a JSON array)
        if input_path.lower().endswith(".json"):
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Otherwise treat as JSONL (one JSON object per line)
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    except Exception as e:
        logging.error(f"Error reading file {input_path}: {e}")
        return []

    processed = []
    for item in data:
        messages = []
        # Retrieve conversations safely
        conversations = item.get("conversations", [])
        for conv in conversations:
            conv_from = conv.get("from", "").lower()
            # Map conversation sender to a role
            if conv_from == "system":
                role = "system"
            elif conv_from == "gpt":
                role = "assistant"
            elif conv_from in ("human", "user"):
                role = "user"
            else:
                # Default to 'assistant' if the role is ambiguous
                role = "assistant"
            content = conv.get("value", "")
            messages.append({"from": role, "value": content})

        text = apply_template(messages, template_str, **kwargs)

        processed.append({"text": text})
    return processed


def write_output(output_path: str, processed: List[Dict[str, str]]) -> None:
    """
    Write the processed conversation data to an output file in JSONL format.

    Args:
        output_path: The path where the output file should be written.
        processed: A list of processed conversation entries.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for entry in processed:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logging.info(f"Wrote processed data to {output_path}")
    except Exception as e:
        logging.error(f"Error writing file {output_path}: {e}")


def completionify_sharegpt(
    input_files: str, output_dir: str, template_str: str, **kwargs
) -> None:
    """
    Convert chat dataset to completion format using a specified template string.

    Args:
        input_files: Path to JSONL or JSON files, can be a directory or a globbable pattern
        output_dir: Directory to save processed files
        template_str: Template string to use for formatting (e.g., 'chatml', 'custom')
        **kwargs: Additional variables to pass to template rendering
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle input as directory or glob pattern
    if os.path.isdir(input_files):
        # If input_files is a directory, search for .json and .jsonl files
        file_paths = glob.glob(os.path.join(input_files, "*.json")) + glob.glob(
            os.path.join(input_files, "*.jsonl")
        )
    else:
        # Otherwise treat as a glob pattern
        file_paths = glob.glob(input_files)

    # Process each file
    for input_path in file_paths:
        processed_data = process_file(input_path, template_str, **kwargs)
        if not processed_data:
            logging.warning(f"No data processed for file: {input_path}")
            continue

        output_filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, output_filename)
        write_output(output_path, processed_data)


def route_template_to_preset(template_input: str, template_kwargs: dict):
    # if one of "atk" or "chatml" or potentially other options in the future, then we set the template to a specific preset Template
    # Otherwise we just assume it is a jinja2 string that acts as a custom template and we make an object from it directly
    # return the template object
    """
    Route template input to either a preset template or create a custom template.

    Args:
        template_input: String indicating either a preset template name ('atk', 'chatml')
                        or a custom Jinja2 template string

    Returns:
        A template string that can be used to create a Jinja2 Template
    """
    # Define preset templates
    CHATML_TEMPLATE_STR = (
        "{% for message in messages %}"
        "{% if message['from'] == 'system' %}"
        "<|im_start|>system\n{{ message['value'] }}<|im_end|>\n"
        "{% elif message['from'] == 'user' %}"
        "<|im_start|>user\n{{ message['value'] }}<|im_end|>\n"
        "{% elif message['from'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['value'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
    )

    ATK_TEMPLATE_STR = (
        "{{ bos_token if bos_token is defined else '' }}"
        "{% for message in messages %}"
        "{% if message['from'] == 'system' %}"
        "Instruction: {{ message['value'] }} **Finished.**\n"
        "{% elif message['from'] == 'user' %}"
        "Human: {{ message['value'] }} **Finished.**\n"
        "{% elif message['from'] == 'assistant' %}"
        "AI: {{ message['value'] }} **Finished.**\n"
        "{% endif %}"
        "{% endfor %}"
    )

    # Check if the input is a preset name
    if template_input.lower() == "chatml":
        return CHATML_TEMPLATE_STR
    elif template_input.lower() == "atk":
        return ATK_TEMPLATE_STR
    else:
        # Assume it's a custom template string
        return template_input


# def main():
#     parser = argparse.ArgumentParser(
#         description='Convert chat dataset to a completion format using a specified template.'
#     )
#     parser.add_argument('input_files', type=str, nargs='?',
#                         help='Globbable path to JSONL or JSON files')
#     parser.add_argument('output_dir', type=str, nargs='?',
#                         help='Directory to save processed files')
#     parser.add_argument('template', type=str, nargs='?',
#                         choices=['chatml', 'custom'],
#                         help='Template to use: chatml or custom')

#     # Optional flags (for backwards compatibility)
#     parser.add_argument('--input_files', type=str, dest='input_files_flag',
#                         help='Globbable path to JSONL or JSON files')
#     parser.add_argument('--output_dir', type=str, dest='output_dir_flag',
#                         help='Directory to save processed files')
#     parser.add_argument('--template', type=str, dest='template_flag',
#                         choices=['chatml', 'custom'],
#                         help='Template to use: chatml or custom')

#     args = parser.parse_args()

#     # Use flag values if provided; otherwise, use positional arguments.
#     input_files = args.input_files_flag if args.input_files_flag else args.input_files
#     output_dir = args.output_dir_flag if args.output_dir_flag else args.output_dir
#     template = args.template_flag if args.template_flag else args.template

#     # Check that all required arguments are provided.
#     if not all([input_files, output_dir, template]):
#         parser.error("All arguments (input_files, output_dir, template) are required either as positional arguments or flags.")

#     # Ensure the output directory exists.
#     os.makedirs(output_dir, exist_ok=True)

#     # Process each file matching the input pattern.
#     for input_path in glob.glob(input_files):
#         processed_data = process_file(input_path, template)
#         if not processed_data:
#             logging.warning(f"No data processed for file: {input_path}")
#             continue

#         output_filename = os.path.basename(input_path)
#         output_path = os.path.join(output_dir, output_filename)
#         write_output(output_path, processed_data)


# if __name__ == '__main__':
#     main()

from datasets import load_dataset, DatasetDict


def save_hf_dataset(
    hf_dataset_path: str, local_path: str, split: str = None, max_samples: int = None
) -> None:
    """
    Save a Hugging Face dataset to local JSONL format if it doesn't exist.

    Args:
        hf_dataset_path: Path/name of the Hugging Face dataset (e.g. 'username/dataset_name')
        local_path: Local file path to save the JSONL file
        split: Optional dataset split to save (default: None for single-split datasets)
        max_samples: Optional maximum number of samples to save (default: None for full dataset)
    """
    print(
        f"Checking if dataset already exists at {local_path} and max_samples is None..."
    )
    if os.path.exists(local_path) and max_samples is None:
        logging.info(f"Dataset already exists at {local_path}, skipping download")
        print(f"Dataset already exists at {local_path}, skipping download")
        return

    try:
        print("Importing datasets module...")
        print("Datasets module imported successfully.")

        # Load dataset with split handling
        if split:
            print(f"Loading dataset {hf_dataset_path} with split {split}...")
            dataset = load_dataset(hf_dataset_path, split=split)
            print(f"Dataset loaded with split {split}.")
        else:
            print(f"Loading dataset {hf_dataset_path} without specifying split...")
            dataset_dict = load_dataset(hf_dataset_path)
            print("Dataset loaded. Checking if it is a DatasetDict...")
            if isinstance(dataset_dict, DatasetDict):
                print("Dataset is a DatasetDict.")
                if len(dataset_dict) == 1:
                    print(
                        "DatasetDict contains a single split. Using the single split."
                    )
                    dataset = next(iter(dataset_dict.values()))
                else:
                    error_message = f"Dataset contains multiple splits: {list(dataset_dict.keys())}. Please specify one."
                    print(error_message)
                    raise ValueError(error_message)
            else:
                print(
                    "Dataset is not a DatasetDict. Using the loaded dataset directly."
                )
                dataset = dataset_dict

        # Apply max_samples if specified
        if max_samples and max_samples > 0:
            print(f"Applying max_samples: {max_samples}")
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            print(f"Dataset truncated to {len(dataset)} samples.")

        # Convert to list of dictionaries
        print("Converting dataset to list of dictionaries...")
        data = [item for item in dataset]
        print(f"Dataset converted. Total samples: {len(data)}")

        # Create directory if it doesn't exist
        print(f"Ensuring directory exists for path: {os.path.dirname(local_path)}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Use existing write_jsonl function
        print(f"Writing dataset to JSONL at {local_path}...")
        write_jsonl(local_path, data)
        logging.info(
            f"Successfully saved {hf_dataset_path} to {local_path} ({len(data)} samples)"
        )
        print(
            f"Successfully saved {hf_dataset_path} to {local_path} ({len(data)} samples)"
        )

    except Exception as e:
        logging.error(f"Failed to save dataset: {e}")
        print(f"Failed to save dataset: {e}")
        raise
