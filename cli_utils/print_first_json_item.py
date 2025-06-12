import argparse
import ijson
import json
import sys
import os


def print_first_item(file_path):
    """
    Prints the first top-level item from a JSON file using a streaming parser.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, "rb") as file:
            # Assuming the JSON contains an array or a stream of objects at the top level.
            # 'item' targets elements within the top-level array/stream.
            # Adjust the prefix if your JSON structure is different (e.g., 'items.item' for nested).
            parser = ijson.items(file, "item")

            first_item = next(parser)
            print(json.dumps(first_item, indent=2))  # Pretty print the first item

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except StopIteration:
        print(f"Error: No items found in the JSON file {file_path}", file=sys.stderr)
        sys.exit(1)
    except ijson.JSONError as e:
        print(f"Error parsing JSON file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Print the first top-level item from a JSON file using a streaming parser."
    )
    parser.add_argument("json_file", help="Path to the input JSON file.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File not found at {args.json_file}", file=sys.stderr)
        sys.exit(1)

    print_first_item(args.json_file)


if __name__ == "__main__":
    main()
