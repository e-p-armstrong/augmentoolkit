import json


def escape_string_for_json(string: str) -> str:
    escaped_string = json.dumps(string)
    
    return escaped_string[1:-1]



# Test the function
if __name__ == "__main__":
    test_str = (
        'This is a "test" [string] {} with some \\"escaped\\" quotes and "unescaped" ones.'
    )
    print(escape_string_for_json(test_str))
