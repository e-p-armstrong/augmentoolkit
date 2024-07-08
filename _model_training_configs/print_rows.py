import argparse
from augmentoolkit.utils.load_dataset import load_dataset
import glob

def main():
    parser = argparse.ArgumentParser(description="Get the total number of rows across files matching a pattern.")
    parser.add_argument("pattern", help="Pattern to match dataset file names (e.g., 'data_*.json')")
    args = parser.parse_args()

    total_rows = 0
    files = glob.glob(args.pattern)

    if not files:
        print(f"No files found matching the pattern: {args.pattern}")
        return

    for file_path in files:
        dataset = load_dataset(file_path)
        total_rows += len(dataset)

    print(f"The total number of rows across {len(files)} files is: {total_rows}")

if __name__ == "__main__":
    main()
    
# TODO make the classifier trainer accept .json and .parquet where they have a label field; each thing with a label is a chunk