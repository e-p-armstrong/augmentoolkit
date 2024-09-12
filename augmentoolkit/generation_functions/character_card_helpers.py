import re

# from .character_card_grammar import character_card_grammar
from .format_qadicts import format_qatuples
import string
import random


def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name


def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"


def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters
