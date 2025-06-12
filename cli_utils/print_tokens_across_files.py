import argparse
import pyarrow.parquet as pq
import json
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
import ijson
import os

tokenizer = AutoTokenizer.from_pretrained("TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ")


def count_tokens(message):
    return len(tokenizer.encode(message))


def process_jsonl(file_path, count_all_turns):
    """Process JSONL file line by line with progress bar."""
    total_tokens = 0
    file_size = os.path.getsize(file_path)

    with open(file_path, "r") as file:
        with tqdm(
            total=file_size,
            desc=f"Processing {os.path.basename(file_path)}",
            unit="B",
            unit_scale=True,
        ) as pbar:
            for line in file:
                obj = json.loads(line)
                if "conversations" in obj:
                    for conversation in obj["conversations"]:
                        if count_all_turns or conversation["from"] == "gpt":
                            total_tokens += count_tokens(conversation["value"])
                elif "text" in obj:
                    total_tokens += count_tokens(obj["text"])

                pbar.update(len(line.encode("utf-8")))

    return total_tokens


def process_json(file_path, count_all_turns):
    """Process JSON file using streaming parser."""
    total_tokens = 0
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as file:
        with tqdm(
            total=file_size,
            desc=f"Processing {os.path.basename(file_path)}",
            unit="B",
            unit_scale=True,
        ) as pbar:
            prev_position = file.tell()
            for obj in ijson.items(file, "item"):  # Stream each top-level array item
                # Process the object like in JSONL
                if "conversations" in obj:
                    for conversation in obj["conversations"]:
                        if count_all_turns or conversation.get("from") == "gpt":
                            total_tokens += count_tokens(conversation["value"])
                if "full_input" in obj:
                    for item in obj["full_input"]:
                        total_tokens += count_tokens(item["content"])
                elif "text" in obj:
                    total_tokens += count_tokens(obj["text"])

                # Estimate progress by file position (may not be perfectly accurate)
                current_position = file.tell()
                pbar.update(current_position - prev_position)
                prev_position = current_position

    return total_tokens


def process_parquet(file_path, count_all_turns):
    """Process parquet file in chunks."""
    total_tokens = 0
    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    chunk_size = 1000  # Adjust based on your memory constraints

    with tqdm(
        total=total_rows, desc=f"Processing {os.path.basename(file_path)}"
    ) as pbar:
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_chunk = batch.to_pandas()

            if "text" in df_chunk.columns:
                chunk_tokens = df_chunk["text"].apply(count_tokens).sum()
                total_tokens += chunk_tokens

            pbar.update(len(df_chunk))

    return total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Count the total number of tokens from 'gpt' across files matching patterns."
    )
    parser.add_argument(
        "patterns", nargs="+", help="One or more patterns to match dataset file names"
    )
    parser.add_argument(
        "--all-turns",
        action="store_true",
        help="Count tokens from all conversation turns, not just GPT turns",
    )
    args = parser.parse_args()

    total_tokens = 0
    all_files = set()
    for pattern in args.patterns:
        matched_files = glob.glob(pattern)
        all_files.update(matched_files)

    files = list(all_files)
    print("Files being processed:")
    for f in files:
        print(f"- {f}")
    print()

    if not files:
        print(f"No files found matching the patterns: {args.patterns}")
        return

    for file_path in files:
        try:
            if file_path.endswith(".parquet"):
                file_tokens = process_parquet(file_path, args.all_turns)
            elif file_path.endswith(".json"):
                file_tokens = process_json(file_path, args.all_turns)
            elif file_path.endswith(".jsonl"):
                file_tokens = process_jsonl(file_path, args.all_turns)
            else:
                print(f"Skipping unsupported file format: {file_path}")
                continue

            total_tokens += file_tokens
            print(f"Tokens in {os.path.basename(file_path)}: {file_tokens:,}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"\nTotal tokens across {len(files)} files: {total_tokens:,}")


if __name__ == "__main__":
    main()
