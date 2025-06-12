# here's what I Want to do, right?
# we want the model to always output things in the right format.
# That being said, the model has often outputted things in its own format.
# I want to be able to extract teh structure from the lack of structure.
# what can we guarantee? The headings. And that the content will follow the heading. Until the next heading
# A heading may be any casing,  surrounded by any marks. But we know what headings to look for.

# I want to build a generalized component for data extraction that will 1. work to make this step not error regardless of opinionated AI, 2. will make extracting data and formatting it into different output formats in the data easy (because we'll want to extract and clean stuff in there, for sure); 3. it would be invaluable in other pipelines as well

"""
First Function requirements:
1. This is a function for extracting structured data from text with headings
2. we know what the headings are in advance (list of uncased strings)
3. the text will generally be of the format
Heading
Text

Heading
text

...etc...

(though, critically, it may be different: this is where the tricky part comes in)

So for instance it might be

Heading

Text

**Heading**

Text

etc...

or maybe even


Heading
- list
- items

or

Heading

- list
- items

or

# Heading

etc.


The only things we can latch onto are the fact that we know what heading names are; we know that content of a heading follows the heading itself. And also, bulleted lists will be with either * or -.

The goal is to extract a json of the heading: content. As the heading originally appears in the input of the function.

headings are input as a list of dicts of {heading: str, type: str}. Type right now can be either paragraph | list. If it is list then extract lists as lists of strings. Remember how I said lists can be with either * or -? we do not know which will be which, so work with both. And strip whitespace.

Build that."""

import re
from typing import List, Dict, Any, Union


def extract_structured_data(
    text: str, headings: Dict[str, Dict[str, Any]]
) -> Dict[str, Union[str, List[str], None]]:
    """
    Extracts structured data from text with various heading formats.

    Args:
        text: Input text containing headings and content
        headings: Dictionary where keys are heading strings (case-insensitive) and values are:
                  - For paragraph type: {'name': 'paragraph'}
                  - For list type: {'name': 'list'}
                  - For rating type: {'name': 'rating', 'prefix': str}

    Returns:
        Dictionary mapping lowercased original headings to their extracted content.
        Headings that were expected but not found will have value None.
    """
    # Create mapping of original headings to their properties with lowercase keys
    heading_configs = {
        k.lower(): {"heading": k.lower(), "type": v} for k, v in headings.items()
    }
    heading_names = [k.lower() for k in headings.keys()]

    # Initialize results dictionary with None for all expected headings
    result = {heading.lower(): None for heading in headings}

    # Split the text into lines for processing
    lines = text.split("\n")

    # Track current heading and content
    current_heading = None
    current_content = []

    # Process each line
    for i, line in enumerate(lines):
        # Clean up the line for examination
        clean_line = re.sub(r'[#\-*•"\'\[\](){}<>]', "", line).strip()

        # Check if this line contains a heading
        found_heading = None
        for heading in heading_names:
            # Case-insensitive search
            if heading.lower() in clean_line.lower():
                found_heading = heading
                break

        if found_heading:
            # Save the previous heading's content
            if current_heading:
                heading_type = heading_configs[current_heading]["type"]

                if heading_type.get("name") == "list":
                    # Extract list items
                    list_items = [
                        re.sub(r"^[\s\-*•]+", "", item).strip()
                        for item in current_content
                        if re.match(r"^\s*[\-*•]", item)
                    ]
                    result[current_heading] = list_items
                elif heading_type.get("name") == "rating":
                    # Find the last line containing the prefix
                    prefix = heading_type.get("prefix", "")
                    last_rating_line = None

                    for line in current_content:
                        if prefix in line:
                            last_rating_line = line

                    if last_rating_line:
                        # Extract text after the prefix on the same line
                        prefix_pos = last_rating_line.find(prefix)
                        extracted = last_rating_line[prefix_pos + len(prefix) :].strip()
                        # Clean markdown characters
                        cleaned_extracted = re.sub(
                            r"[#\\*_\\[\\](){}<>]", "", extracted
                        ).strip()
                        result[current_heading] = cleaned_extracted
                    else:
                        # If prefix not found, leave as None
                        pass
                else:  # paragraph type or fallback
                    # Join paragraph lines
                    cleaned = " ".join(
                        line.strip()
                        for line in current_content
                        if not re.match(r"^\s*[\-*•]", line)
                    )
                    result[current_heading] = cleaned.strip()

                # Reset content collection
                current_content = []

            # Store the new heading - use lowercase
            current_heading = found_heading.lower()
        elif current_heading:
            # Add content to current heading if not empty
            if line.strip():
                current_content.append(line)

    # Don't forget the last section
    if current_heading and current_content:
        heading_type = heading_configs[current_heading]["type"]

        if heading_type.get("name") == "list":
            list_items = [
                re.sub(r"^[\s\-*•]+", "", item).strip()
                for item in current_content
                if re.match(r"^\s*[\-*•]", item)
            ]
            result[current_heading] = list_items
        elif heading_type.get("name") == "rating":
            # Find the last line containing the prefix
            prefix = heading_type.get("prefix", "")
            last_rating_line = None

            for line in current_content:
                if prefix in line:
                    last_rating_line = line

            if last_rating_line:
                # Extract text after the prefix on the same line
                prefix_pos = last_rating_line.find(prefix)
                extracted = last_rating_line[prefix_pos + len(prefix) :].strip()
                # Clean markdown characters
                cleaned_extracted = re.sub(
                    r"[#\\*_\\[\\](){}<>]", "", extracted
                ).strip()
                result[current_heading] = cleaned_extracted
            else:
                # If prefix not found, leave as None
                pass
        else:  # paragraph type or fallback
            cleaned = " ".join(
                line.strip()
                for line in current_content
                if not re.match(r"^\s*[\-*•]", line)
            )
            result[current_heading] = cleaned.strip()

    return result


def test_extract_structured_data():
    """Test cases for structured data extraction"""
    test_cases = [
        # Mixed markdown headings
        (
            """# HEADING1
            Paragraph text here
            
            **Heading2**
            - list item 1
            - list item 2
            
            __heading3__
            * bullet a
            * bullet b""",
            {
                "heading1": {"name": "paragraph"},
                "heading2": {"name": "list"},
                "heading3": {"name": "list"},
            },
            {
                "heading1": "Paragraph text here",
                "heading2": ["list item 1", "list item 2"],
                "heading3": ["bullet a", "bullet b"],
            },
        ),
        # Heading-like text in content
        (
            """Summary asdffdsa
            This contains a - dash and * asterisk
            - But this is an actual list
            - With mixed markers
            * Different bullet type""",
            {"summary asdffdsa": {"name": "list"}},
            {
                "summary asdffdsa": [
                    "But this is an actual list",
                    "With mixed markers",
                    "Different bullet type",
                ]
            },
        ),
        # Testing the new rating type
        (
            """COHERENCE:
            The session's writing makes sense, but it repeats itself in a handful of areas.
            RATING: poor
            
            RULE FOLLOWING:
            I will fill out a checklist where each requirement is an item.
            RATING: awful
            
            QUALITY:
            The session has decently compelling literary content.
            RATING: good
            Extra text after the rating shouldn't be included.""",
            {
                "COHERENCE": {"name": "rating", "prefix": "RATING:"},
                "RULE FOLLOWING": {"name": "rating", "prefix": "RATING:"},
                "QUALITY": {"name": "rating", "prefix": "RATING:"},
            },
            {"coherence": "poor", "rule following": "awful", "quality": "good"},
        ),
        # Test rating extraction with markdown
        (
            """Rating Example:
            Some text before the rating.
            RATING: ** *good* **
            
            Another Section:
            Some other text.""",
            {"Rating Example": {"name": "rating", "prefix": "RATING:"}},
            {"rating example": "good"},
        ),
    ]

    for i, (text, headings, expected) in enumerate(test_cases):
        result = extract_structured_data(text, headings)
        assert (
            result == expected
        ), f"Test case {i+1} failed:\nExpected: {expected}\nGot: {result}"

    print("All test cases passed!")


"""
Next function:


But the question then becomes the writing of a format.

Well it's simple. Take a dict like is ready by the parsing, and take another dict of heading: style. Some paragraph styles (e.g, bold heading, not bold heading,, allcaps heading,  etc.) and some list styles (-, *, whitespace etc.).

So we can extract the content form the outputs. And randomly pick elements of the output format. And guarantee certain kinds of outputs/format obeying.

Basically the reverse of the previous thing, we want to take the output of the previous function, as well as a dict of heading: format, and format the input according to how it says things should be done.

The permissable formats should be many. They should be hardcoded within the function but done in a way that is easily extensible, like a dict of names to inner-defined functions that take the content of that heading and the heading itself as arguments and return a string. Yeah taht sounds good.

We'll also support it such that, if the user wants to have a heading: function instead of a string that maps to an inner function inside function's dict, that is OK too.

Got it? Build it.
"""


def format_structured_data(data: Dict[str, any], formats: Dict[str, any]) -> str:
    """
    Formats structured data into text using specified formatting styles.

    Args:
        data: Dictionary of heading-content pairs from extraction function
        formats: Dictionary mapping headings to format specifications
                 Each specification can be:
                 - A single string for content format only
                 - A tuple of (heading_format, content_format)
                 - A callable function

    Returns:
        Formatted text string
    """
    # Registry of predefined format functions
    formatters = {
        # Heading formats
        "bold_heading": lambda h: f"**{h}**",
        "underline_heading": lambda h: f"__{h}__",
        "hash_heading": lambda h: f"# {h}",
        "allcaps_heading": lambda h: h.upper(),
        "plain_heading": lambda h: h,
        # List formats
        "dash_list": lambda items: "\n".join(f"- {item}" for item in items),
        "asterisk_list": lambda items: "\n".join(f"* {item}" for item in items),
        "numbered_list": lambda items: "\n".join(
            f"{i+1}. {item}" for i, item in enumerate(items)
        ),
        # Paragraph formats
        "indent_paragraph": lambda text: f"    {text}",
        "blockquote_paragraph": lambda text: f"> {text}",
        "plain_paragraph": lambda text: text,
    }

    output = []
    for heading, content in data.items():
        # Get format specifier for this heading
        format_spec = formats.get(heading, "plain_paragraph")

        # Determine heading and content formatting
        if callable(format_spec):
            # If format_spec is a function, use it directly
            formatted_section = format_spec(heading, content)
            output.append(formatted_section)
            continue
        elif isinstance(format_spec, tuple) and len(format_spec) == 2:
            # If format_spec is a tuple, first element is heading format, second is content format
            heading_format, content_format = format_spec
        else:
            # If format_spec is a string, use it for content and default for heading
            heading_format = "plain_heading"
            content_format = format_spec

        # Get heading format function
        if callable(heading_format):
            heading_format_func = heading_format
        else:
            heading_format_func = formatters.get(
                heading_format, formatters["plain_heading"]
            )

        # Apply heading formatting
        formatted_heading = heading_format_func(heading)

        # Get content format function
        if callable(content_format):
            content_format_func = content_format
        else:
            content_format_func = formatters.get(
                content_format, formatters["plain_paragraph"]
            )

        # Format content based on type
        if isinstance(content, list):
            # List content
            formatted_content = content_format_func(content)
        else:
            # Paragraph content
            formatted_content = content_format_func(content)

        output.append(f"{formatted_heading}\n{formatted_content}\n")

    return "\n".join(output).strip()


def test_format_structured_data():
    """Test cases for structured data formatting"""
    test_data = {
        "Summary": ["First point", "Second item"],
        "Details": "This is a paragraph with important information",
    }

    # Test with tuple formats (heading_format, content_format)
    formats = {
        "Summary": ("hash_heading", "numbered_list"),
        "Details": ("bold_heading", "blockquote_paragraph"),
    }

    expected_output = """# Summary
1. First point
2. Second item

**Details**
> This is a paragraph with important information"""

    result = format_structured_data(test_data, formats)

    # Normalize whitespace for comparison
    assert re.sub(r"\s+", " ", result) == re.sub(
        r"\s+", " ", expected_output
    ), f"Formatting failed\nExpected: {expected_output}\nGot: {result}"

    # Test with string formats (content format only)
    simple_formats = {"Summary": "dash_list", "Details": "plain_paragraph"}

    expected_simple = """Summary
- First point
- Second item

Details
This is a paragraph with important information"""

    result_simple = format_structured_data(test_data, simple_formats)
    assert re.sub(r"\s+", " ", result_simple) == re.sub(r"\s+", " ", expected_simple)

    print("Format test passed!")


"""
Final function: detect a heading format based on a string. Given a dict of headings and their types like might be passed into the test format structured data, we want to identify what the format of each of those is (from among the hardcoded ones in the test format structured data function -- maybe make that a global constant for this file so taht it is more accessible)
"""

if __name__ == "__main__":
    test_extract_structured_data()
    test_format_structured_data()
