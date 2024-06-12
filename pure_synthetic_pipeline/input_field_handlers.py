import re
from faker import Faker
import random
import datetime

fake = Faker()

def code_to_format_string(code):
    # Split the code into lines and extract variable names before "="
    lines = code.splitlines()
    variables = []
    for line in lines:
        parts = line.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            if ' ' not in var_name and var_name.isidentifier():
                variables.append(var_name)

    # Generate format string
    format_string = '\n'.join(f"{var.replace('_', ' ').capitalize()}: {{{var}}}" for var in variables)
    format_string += '\nMutators: {mutators}'  # Add 'Mutators' with formatting placeholder
    format_string += '\Category: {exclusive}'

    return format_string

def code_to_function(code):
    # Split the code into lines and extract variable names before "="
    # TODO sort?
    lines = code.splitlines()
    variables = []
    for line in lines:
        # Split the line at the first equals sign
        parts = line.split('=', 1)
        if len(parts) == 2:
            # Get the variable name, strip any whitespace or unwanted characters
            var_name = parts[0].strip()
            if ' ' not in var_name and var_name.isidentifier():
                variables.append(var_name)

    # Function that formats the dictionary values
    def format_dict(data_dict):
        output = []
        for var in variables:
            if var in data_dict:
                value = data_dict[var]
                output.append(f"{var.replace('_', ' ').capitalize()}: {value}")
            else:
                output.append(f"{var.replace('_', ' ').capitalize()}: Key missing in data")

        # Handle 'mutators' as part of the dictionary
        mutators = data_dict.get('mutators', 'None')
        output.append(f"Mutators: {mutators}")
        
        return "\n".join(output)

    return format_dict

def execute_code_to_dict(code):
    # Dictionary to hold the local variables after execution
    local_vars = {}
    
    # Execute the code and capture the local variables in `local_vars`
    exec(code, globals(), local_vars)

    # Extract only the variables that were assigned in the code
    result_dict = {}
    lines = code.splitlines()
    for line in lines:
        parts = line.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            if ' ' not in var_name and var_name.isidentifier():
                # Ensure the variable exists in local_vars and isn't a leftover from a previous execution
                if var_name in local_vars:
                    result_dict[var_name] = local_vars[var_name]

    return result_dict

def extract_first_code_block(text):
    # Regular expression to find code blocks, skipping optional language specifier
    pattern = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Return the content between the triple backticks
        return match.group(1).strip()
    else:
        # Return None if no code block is found
        return None

def create_dict_keys(code):
    """
    from code, creates text that, if placed in a dictionary, would add the variables as keys into the code
    """
    lines = code.splitlines()
    final_results = []
    for line in lines:
        if "=" in line:
            line_parts = line.split("=")
            key = line_parts[0].strip()
            final_string = f'"{key}": {key},'
            final_results.append(final_string)
    return "\n".join(final_results)

def format_prompt_yaml(prompt_list):
    # Makes a string of everything after the system prompt until right before the last user message in a human-readable way
    end_string = ""
    for msg in prompt_list:
        role = "HUMAN" if msg['role'] == "user" else "AI"
        end_string += f"**{role}**:\n{msg['content']}\n\n"
    return end_string