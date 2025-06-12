import re


def extract_first_words(character_name, text):
    # Regular expression pattern to extract first word after the character's name
    pattern = rf"{character_name}: \"(\w+)"

    # Find all matches in the text
    matches = re.findall(pattern, text)

    return matches


def remove_think_tags(text):
    # Use regex to remove anything between <think> and </think> tags
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text
