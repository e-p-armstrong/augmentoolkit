#!/usr/bin/env python3

import sys
import yaml
from pathlib import Path
import re


def process_yaml_directory(source_dir_path_str: str):
    """
    Creates a copy of a directory containing YAML files, keeping only the
    first and last elements of lists found in each YAML file.
    The output YAML aims for unquoted keys and literal block style for multiline strings.

    Args:
        source_dir_path_str: The path to the source directory.
    """
    source_dir = Path(source_dir_path_str)

    if not source_dir.is_dir():
        print(
            f"Error: Source path '{source_dir}' is not a valid directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir_name = f"{source_dir.name}_local"
    dest_dir = source_dir.parent / dest_dir_name

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created destination directory: '{dest_dir}'")
    except OSError as e:
        print(
            f"Error creating destination directory '{dest_dir}': {e}", file=sys.stderr
        )
        sys.exit(1)

    processed_count = 0
    skipped_count = 0

    for source_file_path in source_dir.rglob("*.yaml"):
        relative_path = source_file_path.relative_to(source_dir)
        dest_file_path = dest_dir / relative_path

        print(f"Processing '{relative_path}'...")

        try:
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(source_file_path, "r") as infile:
                data = yaml.safe_load(infile)

            if not isinstance(data, list):
                print(f"  Skipping '{relative_path}': Content is not a list.")
                skipped_count += 1
                continue

            if len(data) >= 2:
                new_data = [data[0], data[-1]]
            elif len(data) == 1:
                new_data = [data[0]]
            else:
                new_data = []

            with open(dest_file_path, "w") as outfile:
                yaml.dump(
                    new_data,
                    outfile,
                    default_flow_style=False,
                    width=float("inf"),
                    sort_keys=False,
                )

            with open(dest_file_path, "r") as infile:
                content_after_dump = infile.read()

            def format_yaml_strings_to_block(match_obj):
                indent = match_obj.group(1)
                key = match_obj.group(2)
                double_quoted_raw = match_obj.group(3)
                single_quoted_raw = match_obj.group(4)

                raw_value_content = None
                is_double_quoted_match = False

                if double_quoted_raw is not None:
                    raw_value_content = double_quoted_raw
                    is_double_quoted_match = True
                elif single_quoted_raw is not None:
                    raw_value_content = single_quoted_raw
                else:
                    return match_obj.group(0)  # Should not happen

                # Construct a minimal valid YAML to parse the captured string value robustly
                # This handles all YAML unescaping rules (e.g., \n, \", '')
                if is_double_quoted_match:
                    temp_yaml_snippet = f'temp_key: "{raw_value_content}"'
                else:
                    temp_yaml_snippet = f"temp_key: '{raw_value_content}'"

                try:
                    actual_string_value = yaml.safe_load(temp_yaml_snippet)["temp_key"]
                    if not isinstance(actual_string_value, str):
                        return match_obj.group(0)  # Parsed value is not a string
                except yaml.YAMLError:
                    return match_obj.group(0)  # Failed to parse, leave original

                # Apply block formatting if the string has newlines or is a 'content' field (and not empty)
                if "\n" in actual_string_value or (
                    key == "content" and actual_string_value
                ):
                    # Split into lines. splitlines() handles \r\n, \n, \r
                    value_lines = actual_string_value.splitlines()

                    # Build the block lines
                    block_lines = [f"{indent}{key}: |"]
                    for line in value_lines:
                        block_lines.append(f"{indent}  {line}")

                    # Join lines with newline character. This correctly places newlines
                    # between lines, including preserving a trailing newline if
                    # splitlines() generated a final empty string from one.
                    return "\n".join(block_lines)
                else:
                    return match_obj.group(0)  # Not multiline, leave as is

            # Regex to find keys followed by EITHER a double-quoted string OR a single-quoted string.
            # Captures: (indentation), (key), (double_quoted_content OR None), (single_quoted_content OR None)
            # (?s) DOTALL equivalent for content match. ([^"\\]|\\.)* for double, ([^']|'')* for single.
            pattern = re.compile(
                r'^(\s*)([a-zA-Z0-9_-]+):\s+(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\']|\'\')*)\')$',
                re.MULTILINE,
            )

            modified_content = pattern.sub(
                format_yaml_strings_to_block, content_after_dump
            )

            if modified_content and not modified_content.endswith("\n"):
                modified_content += "\n"

            with open(dest_file_path, "w") as outfile:
                outfile.write(modified_content)

            print(f"  Successfully created and formatted '{relative_path}'.")
            processed_count += 1

        except yaml.YAMLError as e:
            print(f"  Error parsing YAML in '{relative_path}': {e}", file=sys.stderr)
            skipped_count += 1
        except Exception as e:
            print(
                f"  An unexpected error occurred processing '{relative_path}': {e}",
                file=sys.stderr,
            )
            skipped_count += 1

    print(f"\nProcessing complete.")
    print(f"  Processed files: {processed_count}")
    print(f"  Skipped files: {skipped_count}")
    print(f"  Output directory: '{dest_dir}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python localify_prompt.py <path_to_yaml_folder>", file=sys.stderr)
        sys.exit(1)

    source_directory = sys.argv[1]
    process_yaml_directory(source_directory)
