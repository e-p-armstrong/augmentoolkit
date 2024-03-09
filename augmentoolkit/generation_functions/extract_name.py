import re


def extract_name(str):
    # Regular expression to match 'Name:' followed by any characters until the end of the line
    name_regex = r"^Name:\s*([^\s]*)"

    # Searching in the multiline string
    match = re.search(name_regex, str, re.MULTILINE)

    if match:
        name = match.group(1)
        print(f"Extracted name: {name}")
        return name
    else:
        name_regex = r"Name: *([^\\]*)"

        # Searching in the multiline string
        match = re.search(name_regex, str, re.MULTILINE)

        if match:
            name = match.group(1)
            print(f"Extracted name: {name}")
            return name
