import argparse
import pyarrow.parquet as pq
import json
import glob
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
)

def count_tokens(message):
    return len(tokenizer.encode(message))

def load_dataset(file_path):
    if file_path.endswith(".parquet"):
        table = pq.read_table(file_path)
        dataset = table.to_pandas()
    elif file_path.endswith(".json"):
        with open(file_path, "r") as file:
            dataset = json.load(file)
    elif file_path.endswith(".jsonl"):
        dataset = []
        with open(file_path, "r") as file:
            for line in file:
                dataset.append(json.loads(line))
    else:
        raise ValueError("Unsupported file format. Please provide a parquet, json, or jsonl file.")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Count the total number of tokens from 'gpt' across files matching a pattern.")
    parser.add_argument("pattern", help="Pattern to match dataset file names (e.g., 'data_*.json')")
    args = parser.parse_args()

    total_tokens = 0
    files = glob.glob(args.pattern)

    if not files:
        print(f"No files found matching the pattern: {args.pattern}")
        return

    for file_path in files:
        dataset = load_dataset(file_path)
        for obj in dataset:
            for conversation in obj['conversations']:
                if conversation['from'] == 'gpt':
                    total_tokens += count_tokens(conversation['value'])

    print(f"The total number of tokens from 'gpt' across {len(files)} files is: {total_tokens}")

if __name__ == "__main__":
    main()