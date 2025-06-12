import json
import sys
import glob
import random
from pathlib import Path
from typing import List, Dict, Any


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a jsonl file and return list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Write list of JSON objects to a jsonl file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def extract_qa_pairs(conversations: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """Extract all Q/A pairs from conversations."""
    qa_pairs = []

    for conv in conversations:
        messages = conv["conversations"]
        # Process pairs of messages
        # First, check if a system message is present as the first message. If so we want to include it in the qa pairs. and handle this edge case since otherwise it will break.
        # Check if the first message is a system message
        if messages and messages[0]["from"] == "system":
            system_message = messages[0]
            # Start from index 1 to skip the system message when processing pairs
            for i in range(1, len(messages), 2):
                if i + 1 < len(messages):  # Ensure we have both Q and A
                    if (
                        messages[i]["from"] == "human"
                        and messages[i + 1]["from"] == "gpt"
                    ):
                        # Include system message with each Q/A pair
                        qa_pairs.append([system_message, messages[i], messages[i + 1]])
            # Return early since we've handled this conversation
            continue

        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):  # Ensure we have both Q and A
                if messages[i]["from"] == "human" and messages[i + 1]["from"] == "gpt":
                    qa_pairs.append([messages[i], messages[i + 1]])

    return qa_pairs


def main():
    if len(sys.argv) != 4:
        print("Usage: script.py input_folder output_file target_pairs")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    target_pairs = int(sys.argv[3])

    # Get all JSONL files in input folder
    input_files = glob.glob(str(Path(input_folder) / "*.jsonl"))
    if not input_files:
        print(f"No .jsonl files found in {input_folder}")
        sys.exit(1)

    # Read and extract all Q/A pairs
    all_qa_pairs = []
    for file_path in input_files:
        conversations = read_jsonl(file_path)
        qa_pairs = extract_qa_pairs(conversations)
        all_qa_pairs.extend(qa_pairs)

    # Shuffle all Q/A pairs
    random.shuffle(all_qa_pairs)

    # Create new conversations
    new_conversations = []
    for i in range(0, len(all_qa_pairs), target_pairs):
        conv_pairs = all_qa_pairs[i : i + target_pairs]
        messages = []
        for pair in conv_pairs:
            messages.extend(pair)

        if messages:  # Only create conversation if we have messages
            new_conversations.append({"conversations": messages})

    # Write output
    write_jsonl(new_conversations, output_file)
    print(f"Created {len(new_conversations)} conversations in {output_file}")


if __name__ == "__main__":
    main()
