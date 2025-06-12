import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any
import random
import json
import hashlib

# NOTE: deepseek r1 is the best "surgical changes" LLM, love the whale


def combine_single_turn_convs(
    factual_outputs: list, target_pairs: int, dataset_context: str
):
    """Combine single-turn conversations from multiple sources using combine_sharegpts-style logic"""
    all_qa_pairs = []

    for output in factual_outputs:
        # Extract conversations from each output
        conversations = output["output"]

        # Extract Q/A pairs from conversations
        for conv in conversations:
            messages = conv["conversations"]
            # Handle system message at start if present
            if messages and messages[0]["from"] == "system":
                system_message = messages[0]
                # Process pairs after system message
                for i in range(1, len(messages), 2):
                    if i + 1 < len(messages):
                        all_qa_pairs.append(
                            {
                                "system": system_message["value"],
                                "human": messages[i]["value"],
                                "gpt": messages[i + 1]["value"],
                            }
                        )
            else:
                # Process regular pairs
                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        all_qa_pairs.append(
                            {
                                "human": messages[i]["value"],
                                "gpt": messages[i + 1]["value"],
                            }
                        )

    # Shuffle and create new conversations
    # Sort QA pairs to make order irrelevant
    all_qa_pairs.sort(key=lambda x: tuple(sorted(x.items())))

    # Create deterministic seed based on input data
    serialized = json.dumps(all_qa_pairs, sort_keys=True)
    seed = int(hashlib.sha256(serialized.encode()).hexdigest(), 16) & 0xFFFFFFFF
    print(f"!!Seed: {seed}")
    rng = random.Random(seed)
    rng.shuffle(all_qa_pairs)

    # Replace dataset context placeholders
    context = dataset_context
    context_vars = {
        "{context}": context,
        "{context_uppercase}": context.upper(),
        "{context_lowercase}": context.lower(),
    }

    new_conversations = []
    for i in range(0, len(all_qa_pairs), target_pairs):
        conv_groups = all_qa_pairs[i : i + target_pairs]
        messages = []
        # if any of them have a system message, we need to add it to the beginning of the messages
        for group in conv_groups:
            if "system" in group:
                system_message = {"from": "system", "value": group["system"]}
                for var, replacement in context_vars.items():
                    system_message["value"] = system_message["value"].replace(
                        var, replacement
                    )
                messages.append(system_message)
                break

        for group in conv_groups:
            human_msg = {"from": "human", "value": group["human"]}
            gpt_msg = {"from": "gpt", "value": group["gpt"]}

            # Apply context replacements
            for var, replacement in context_vars.items():
                human_msg["value"] = human_msg["value"].replace(var, replacement)
                gpt_msg["value"] = gpt_msg["value"].replace(var, replacement)

            messages.append(human_msg)
            messages.append(gpt_msg)
        if messages:
            new_conversations.append({"conversations": messages})

    return new_conversations


### NEW CODE


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    # print("Reading JSONL file:", file_path) # Reduce verbosity
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # print("Found", len(lines), "lines in", file_path) # Reduce verbosity
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(data: List[Dict[str, Any]], file_path: str):
    # print("Writing", len(data), "conversations to", file_path) # Reduce verbosity
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\\n")


def combine_single_and_multi_turn(
    # single_qa_pairs: List[List[Dict[str, str]]],
    # multi_turn_convs: List[Dict[str, Any]],
    convs: List[Dict[str, Any]],
    min_pairs: int,
    max_pairs: int,
    dataset_context: str,
) -> List[Dict[str, Any]]:
    """
    Combines extracted single QA pairs and multi-turn conversations into longer conversations.

    Args:
        single_qa_pairs: A list where each item is a list of messages representing a single QA turn (e.g., [system, human, gpt] or [human, gpt]).
        multi_turn_convs: A list of ShareGPT-style conversations (dictionaries with a 'conversations' key).
        min_pairs: The minimum number of QA pairs desired in a combined conversation.
        max_pairs: The maximum number of QA pairs desired in a combined conversation.
        dataset_context: The context string to replace placeholders in system messages.

    Returns:
        A list of combined ShareGPT-style conversations.
    """
    # print(f"Combining {len(single_qa_pairs)} single pairs and {len(multi_turn_convs)} multi-turn convs.") # Reduce verbosity
    # print(f"Target pairs per conversation: {min_pairs}-{max_pairs}")

    # Replace dataset context placeholders
    context_vars = {
        "{context}": dataset_context,
        "{context_uppercase}": dataset_context.upper(),
        "{context_lowercase}": dataset_context.lower(),
    }

    combine_random = random.Random(11037)
    # combine_random.shuffle(single_qa_pairs)
    # Shuffle multi_turn_convs by shuffling their *indices* to avoid modifying the input list directly if passed by reference elsewhere
    all_indices = list(range(len(convs)))
    combine_random.shuffle(all_indices)

    all_conversations = []
    indices_idx = 0
    # iteration = 0 # Reduce verbosity

    # The way the loop works is by shuffling the convs and then incrementing hte indices until we run out of the length of hte list
    while indices_idx < len(all_indices):
        # iteration += 1 # Reduce verbosity
        # print(f"\\nIteration {iteration}: single_idx={single_idx}, multi_idx={multi_indices_idx}") # Reduce verbosity
        target_pairs = (
            combine_random.randint(min_pairs, max_pairs)
            if max_pairs >= min_pairs
            else min_pairs
        )
        # print("Target pairs for this conversation:", target_pairs) # Reduce verbosity
        current_pairs = 0
        messages = []
        current_system_message = (
            None  # Track system message for the *current* combined conversation
        )

        while current_pairs < target_pairs and indices_idx < len(all_indices):
            # print(f"Current pairs: {current_pairs}/{target_pairs}, S_idx={single_idx}, M_idx={multi_indices_idx}") # Reduce verbosity

            # Decide whether to use multi-turn or single-turn
            # Use multi-turn
            original_multi_idx = all_indices[indices_idx]
            conv_obj = convs[original_multi_idx]
            # print("!!CONVOBJ")
            # print(conv_obj)
            conv = conv_obj["conversations"]
            # print(f"Trying multi-turn conv index {original_multi_idx} ({len(conv)} messages)") # Reduce verbosity

            pairs_to_add = sum(1 for msg in conv if msg.get("from") == "gpt")
            # print(f"Pairs to add from multi-turn: {pairs_to_add}") # Reduce verbosity

            # Handle system prompt from multi-turn
            multi_system_message = conv[0] if conv[0].get("from") == "system" else None
            start_offset = 1 if multi_system_message else 0

            if multi_system_message:
                if not current_system_message:
                    current_system_message = multi_system_message.copy()
                    # Apply context replacements to the newly set system message
                    for var, replacement in context_vars.items():
                        current_system_message["value"] = current_system_message[
                            "value"
                        ].replace(var, replacement)
                    messages.append(current_system_message)
                # else: print("Multi-turn system message ignored, already have one.") # Reduce verbosity

            messages.extend(msg.copy() for msg in conv[start_offset:])  # Add copies
            current_pairs += pairs_to_add
            indices_idx += 1

        if messages:
            # No need to filter system messages here as we explicitly manage `current_system_message`
            # print(f"Adding conversation with {len(messages)} messages ({current_pairs} pairs)") # Reduce verbosity
            all_conversations.append({"conversations": messages})
        # else: # Reduce verbosity
        # print("No valid messages collected for this conversation iteration.")

    # print("Created", len(all_conversations), "combined conversations") # Reduce verbosity
    return all_conversations


def load_conversations_from_files(files: List[str]) -> List[Dict[str, Any]]:
    """Loads conversations from a list of JSONL file paths."""
    all_convs = []
    for file_path in files:
        # print("Loading conversations from", file_path) # Reduce verbosity
        convs = read_jsonl(file_path)
        # print("Loaded", len(convs), "conversations from", file_path) # Reduce verbosity
        all_convs.extend(convs)
    return all_convs
