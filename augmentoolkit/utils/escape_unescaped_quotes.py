def escape_unescaped_quotes(s):
    # Initialize a new string to store the result
    result = ""
    # Iterate through the string, keeping track of whether the current character is preceded by a backslash
    i = 0
    while i < len(s):
        # If the current character is a quote
        if s[i] == '"':
            # Check if it's the first character or if the preceding character is not a backslash
            if i == 0 or s[i - 1] != "\\":
                # Add an escaped quote to the result
                result += r"\""
            else:
                # Add the quote as is, because it's already escaped
                result += '"'
        else:
            # Add the current character to the result
            result += s[i]
        i += 1
    return result


# Test the function
if __name__ == "__main__":
    test_str = (
        'This is a "test" string with some \\"escaped\\" quotes and "unescaped" ones.'
    )
    print(escape_unescaped_quotes(test_str))
