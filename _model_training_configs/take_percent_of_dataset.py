import argparse
import json
import random
import pyarrow.parquet as pq
import pandas as pd

def load_dataset(file_path):
    if file_path.endswith(".parquet"):
        table = pq.read_table(file_path)
        dataset = table.to_pandas().to_dict(orient="records")
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

def save_output(dataset, output_file):
    with open(output_file, "w") as file:
        json.dump(dataset, file, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Select a random subset of samples from a dataset.")
    parser.add_argument("dataset_file", help="Path to the dataset file (parquet, json, or jsonl)")
    parser.add_argument("percentage", type=float, help="Percentage of samples to select (0-100)")
    parser.add_argument("output_file", help="Path to the output json file")
    args = parser.parse_args()

    if not (0 <= args.percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100.")

    dataset = load_dataset(args.dataset_file)
    num_samples = int(len(dataset) * args.percentage / 100)
    selected_samples = random.sample(dataset, num_samples)
    save_output(selected_samples, args.output_file)

if __name__ == "__main__":
    main()