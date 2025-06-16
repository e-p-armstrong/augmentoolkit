import os
import json


def iterate_over_items(output_dir, output_file_name, func):

    output_path = os.path.join(output_dir, output_file_name)

    if not os.path.exists(output_path):
        return {}

    with open(output_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}

    # iterate over information in the data with the function provided
    for key, value in data.items():
        func(key, value)

    # save data over input file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    return data


# helpers to write to and read from the output dict files that are produced by augmentoolkit operation
# lets you use normal code to change a lot of these without having to get messier


def load_data(output_dir, output_file_name):

    output_path = os.path.join(output_dir, output_file_name)

    if not os.path.exists(output_path):
        return {}

    with open(output_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}

    return data


def save_data(input_dict, output_dir, output_file_name):
    if not output_file_name.endswith(".json"):
        output_file_name = output_file_name + ".json"
    output_path = os.path.join(output_dir, output_file_name)

    os.makedirs(output_dir, exist_ok=True)

    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}

    # Update existing data with new data, but don't remove existing keys
    for key, value in input_dict.items():
        if (
            key in existing_data
            and isinstance(existing_data[key], dict)
            and isinstance(value, dict)
        ):
            # For nested dictionaries, update recursively without removing existing keys
            for subkey, subvalue in value.items():
                existing_data[key][subkey] = subvalue
        else:
            # For non-dict values or new keys, just set the value
            existing_data[key] = value

    # Write the updated data back to the file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4)


def recombine_many_to_one(
    input_wide_dict,
):  # the policy of handling gnarly fancy things outside in code, rather than with a new abstraction for literally everything, will either pay off or be a nightmare. Little in-between. Well maybe not.
    text_hash_groups = {}

    # First pass - group by text hash
    for key, value in input_wide_dict.items():
        text_hash = key.split("-")[0]
        if text_hash not in text_hash_groups:
            text_hash_groups[text_hash] = {}
        index_hash = key.split("-")[1]
        text_hash_groups[text_hash][index_hash] = value

    # Second pass - sort values for each text hash group
    for text_hash in text_hash_groups:
        sorted_values = []
        for key in sorted(text_hash_groups[text_hash].keys(), key=lambda x: int(x)):
            sorted_values.append(text_hash_groups[text_hash][key])
        text_hash_groups[text_hash]["sorted"] = sorted_values

    return text_hash_groups
