import os
import yaml
import uuid
import json
import asyncio
from typing import Callable, Dict, List, Any, Union


def create_log_observer(
    log_dir: str,
) -> (
    Callable
):  # TODO maybe put these in a class so that I can have type checking and warn when I am putting an input observer in an output observer list?
    """
    Creates an output observer that logs all inputs and outputs to files.

    Args:
        log_dir: Directory path where logs will be saved

    Returns:
        A callback function that logs inputs and outputs
    """
    full_output_path = os.path.join(log_dir, "debug_outputs")

    def log_observer(
        input_data: Union[str, List[Dict[str, str]]],
        output: str,
        completion_mode: bool,
        *args,
        **kwargs,
    ) -> None:
        """
        Logs the input and output to a file with a UUID.

        Args:
            input_data: Either a prompt string (completion mode) or messages list (chat mode)
            output: The model's response
            completion_mode: Whether this was a completion (True) or chat (False) request
        """
        # Extract prefix from input_data
        prefix = ""
        if completion_mode:
            # For completion mode, get first 10 chars from the string
            prefix = input_data[:25].lower().replace(" ", "_")
        else:
            # For chat mode, get first 10 chars from the first message content
            if input_data and len(input_data) > 0 and "content" in input_data[0]:
                prefix = input_data[0]["content"][:25].lower().replace(" ", "_")

        # Create log ID with prefix
        log_id = f"{prefix}_{str(uuid.uuid4())}"

        os.makedirs(full_output_path, exist_ok=True)
        if completion_mode:
            print(f"Saving output to {log_id}.txt")
            # For completion mode, save as .txt
            file_path = os.path.join(full_output_path, f"{log_id}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(input_data + output)
        else:
            print(f"Saving output to {log_id}.yaml")
            # For chat mode, save as .yaml
            file_path = os.path.join(full_output_path, f"{log_id}.yaml")

            # Create a full conversation history including the assistant's response
            full_conversation = input_data.copy()  # Copy the input messages
            full_conversation.append({"role": "assistant", "content": output})

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    full_conversation, f, default_flow_style=False, allow_unicode=True
                )

    return log_observer


def create_input_token_counter(
    counter: Dict[str, float],
    cost_per_million: float,
    count_tokens_fn: Callable,
    persistence_path: str = None,
) -> Callable:
    """
    Creates an input observer that counts tokens and calculates cost.

    Args:
        counter: A dictionary to track token counts and costs
        cost_per_million: Cost per million tokens
        count_tokens_fn: Function to count tokens in text
        persistence_path: Optional path to persist counter data to a JSON file

    Returns:
        A callback function that updates the counter
    """
    # Initialize lock for file access
    json_path = None

    if persistence_path:
        os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
        json_path = persistence_path

        # Load existing counter data if available
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    # Update the counter with saved values
                    counter.update(saved_data)
            except (json.JSONDecodeError, FileNotFoundError):
                # If file is corrupted or doesn't exist, start fresh
                pass

    def input_token_counter(
        input_data: Union[str, List[Dict[str, str]]],
        completion_mode: bool,
        *args,
        **kwargs,
    ) -> None:
        """
        Counts tokens in the input and updates the counter.

        Args:
            input_data: Either a prompt string (completion mode) or messages list (chat mode)
            completion_mode: Whether this was a completion (True) or chat (False) request
        """
        if completion_mode:
            # For completion mode, input is a string
            token_count = count_tokens_fn(input_data)
        else:
            # For chat mode, input is a list of message dicts
            token_count = 0
            for message in input_data:
                token_count += count_tokens_fn(message["content"])

        # Update counter
        if "input_tokens" not in counter:
            counter["input_tokens"] = 0
        if "input_cost" not in counter:
            counter["input_cost"] = 0.0

        counter["input_tokens"] += token_count
        counter["input_cost"] += (token_count / 1_000_000) * cost_per_million

        # Persist counter to file if path was provided
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(counter, f)

    return input_token_counter


def create_output_token_counter(
    counter: Dict[str, float],
    cost_per_million: float,
    count_tokens_fn: Callable,
    persistence_path: str = None,
) -> Callable:
    """
    Creates an output observer that counts tokens and calculates cost.

    Args:
        counter: A dictionary to track token counts and costs
        cost_per_million: Cost per million tokens
        count_tokens_fn: Function to count tokens in text
        persistence_path: Optional path to persist counter data to a JSON file

    Returns:
        A callback function that updates the counter
    """
    # Initialize lock for file access
    json_path = None

    if persistence_path:
        os.makedirs(os.path.dirname(persistence_path), exist_ok=True)
        json_path = persistence_path

        # Load existing counter data if available
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    # Update the counter with saved values
                    counter.update(saved_data)
            except (json.JSONDecodeError, FileNotFoundError):
                # If file is corrupted or doesn't exist, start fresh
                pass

    def output_token_counter(
        input_data: Union[str, List[Dict[str, str]]],
        output: str,
        completion_mode: bool,
        *args,
        **kwargs,
    ) -> None:
        """
        Counts tokens in the output and updates the counter.

        Args:
            input_data: Either a prompt string (completion mode) or messages list (chat mode)
            output: The model's response
            completion_mode: Whether this was a completion (True) or chat (False) request
        """
        token_count = count_tokens_fn(output)

        # Update counter
        if "output_tokens" not in counter:
            counter["output_tokens"] = 0
        if "output_cost" not in counter:
            counter["output_cost"] = 0.0

        counter["output_tokens"] += token_count
        counter["output_cost"] += (token_count / 1_000_000) * cost_per_million

        # Persist counter to file if path was provided
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(counter, f)

    return output_token_counter
