# Contains utils for truncating at random parts of sharegpt conversations, and adding a system prompt prefix, for the sake of making it ready for GRPO.

from typing import List, Dict, Any
import random
import logging
import hashlib
import json


def process_single_dataset(  # Deterministic truncation
    dataset: List[Dict[str, Any]],
    force_single_turn: bool = False,
    single_turn_ratio: float = 0.7,
    system_prompt: str = "",
) -> List[Dict[str, Any]]:
    """Process a single dataset with deterministic truncation points."""
    processed = []

    for conv in dataset:
        try:
            messages = conv["conversations"]

            # Handle system prompt injection
            if system_prompt:
                modified_messages = messages.copy()
                if modified_messages and modified_messages[0]["from"] == "system":
                    # Prepend to existing system message with newlines
                    modified_messages[0][
                        "value"
                    ] = f"{system_prompt}\n\n{modified_messages[0]['value']}"
                else:
                    # Insert new system message at beginning
                    modified_messages.insert(
                        0, {"from": "system", "value": system_prompt}
                    )
                messages = modified_messages

            # Create deterministic hash from conversation content
            original_conv = (
                json.dumps(conv["conversations"], sort_keys=True) + system_prompt
            )
            hash_int = int(hashlib.sha256(original_conv.encode()).hexdigest(), 16)
            max_hash = (1 << 256) - 1  # SHA-256 max value
            hash_ratio = hash_int / max_hash

            assistant_indices = [
                i for i, msg in enumerate(messages) if msg["from"] == "gpt"
            ]

            if not assistant_indices:
                continue

            # Deterministic truncation logic
            if force_single_turn:
                cut_idx = assistant_indices[0]
            else:
                use_single = hash_ratio < single_turn_ratio
                if use_single:
                    cut_idx = assistant_indices[0]
                else:
                    # Select index using hash-based deterministic "randomness"
                    chosen_idx = hash_int % len(assistant_indices)
                    cut_idx = assistant_indices[chosen_idx]

            # Preserve modified messages with system prompt in output
            truncated_conv = {
                "conversations": messages[:cut_idx],
                "answer": messages[cut_idx]["value"],
            }
            processed.append(truncated_conv)

        except Exception as e:
            logging.error(f"Error processing conversation: {str(e)}")
            continue

    return processed


def process_multiple_datasets(  # Deterministic sampling
    datasets: List[List[Dict[str, Any]]], percentages: List[float], total_rows: int
) -> List[Dict[str, Any]]:
    """Combine multiple datasets deterministically."""
    if len(datasets) != len(percentages):
        raise ValueError("Datasets and percentages must have same length")

    # Normalize percentages
    total_pct = sum(percentages)
    percentages = [p / total_pct for p in percentages]

    combined = []
    for dataset, pct in zip(datasets, percentages):
        # Calculate needed samples
        n_samples = int(total_rows * pct)
        processed = process_single_dataset(dataset)

        # Deterministic sampling using hashed sorting
        if len(processed) > n_samples:
            # Sort by hash of conversation content for deterministic selection
            processed.sort(
                key=lambda x: int(
                    hashlib.sha256(
                        json.dumps(x["conversations"], sort_keys=True).encode()
                    ).hexdigest(),
                    16,
                )
            )
            combined.append(processed[:n_samples])
        else:
            combined.append(processed)
            logging.warning(f"Insufficient samples: {len(processed)}/{n_samples}")

    # Deterministic shuffle using hashed sorting
    combined.sort(
        key=lambda x: int(
            hashlib.sha256(
                json.dumps(x["conversations"], sort_keys=True).encode()
            ).hexdigest(),
            16,
        )
    )

    return combined[:total_rows]
