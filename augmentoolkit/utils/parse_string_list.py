import ast

def parse_string_list(input_data):
    if isinstance(input_data, list):
        # If input is already a list, validate its contents
        if all(isinstance(item, str) for item in input_data):
            return input_data
        else:
            print("Error: All items in the list must be strings")
            return None

    elif isinstance(input_data, str):
        try:
            # Use ast.literal_eval to safely evaluate the string
            parsed_data = ast.literal_eval(input_data)
            
            # Check if the result is a list
            if not isinstance(parsed_data, list):
                raise ValueError("Input is not a valid list")
            
            # Check if all elements are strings
            if not all(isinstance(item, str) for item in parsed_data):
                raise ValueError("All items in the list must be strings")
            
            return parsed_data
        except (ValueError, SyntaxError) as e:
            # Handle parsing errors
            print(f"Error parsing input: {e}")
            return None

    else:
        print("Error: Input must be a string or a list")
        return None