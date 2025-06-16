import json
import pyarrow.parquet as pq


def load_dataset(file_path):
    if file_path.endswith(".parquet"):
        table = pq.read_table(file_path)
        dataset = table.to_pandas()
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)
    elif file_path.endswith(".jsonl"):
        dataset = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                dataset.append(json.loads(line))
    else:
        raise ValueError(
            "Unsupported file format. Please provide a parquet, json, or jsonl file."
        )
    return dataset
