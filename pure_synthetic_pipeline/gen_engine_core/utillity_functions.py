import random
import itertools
import os
import asyncio
import json
import re
import sys
import time
from typing import List
from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.generation_functions.generation_step_class import GenerationStep
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
import logging
from math import ceil
import traceback
import glob
import uuid
import yaml
from collections import defaultdict, deque

# STANDARD FUNCTIONS USED IN ALL SITUATIONS

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")


def count_tokens(message):
    return len(tokenizer.encode(message))


with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT_FOLDER = obj_conf["PATH"]["OUTPUT"]
DEFAULT_PROMPT_PATH = obj_conf["PATH"]["DEFAULT_PROMPTS"]
COMPLETION_MODE = obj_conf["SYSTEM"]["COMPLETION_MODE"]
LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
API_KEY_A = obj_conf["API"]["API_KEY_A"]
API_KEY_B = obj_conf["API"]["API_KEY_B"]
BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
MODE = obj_conf["API"]["MODE"]
CONCURRENCY_LIMIT = obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"]
LOG_EVERYTHING = obj_conf["SYSTEM"]["LOG_EVERYTHING"]

engine_wrapper = EngineWrapper(
    model=LOGICAL_MODEL_A,
    api_key=API_KEY_A,
    base_url=BASE_URL_A,
    mode=MODE,
    # quantization="gptq" # modify if you want to do stuff with aphrodite
)

engine_wrapper_large = EngineWrapper(
    model=LOGICAL_MODEL_B,
    api_key=API_KEY_B,
    base_url=BASE_URL_B,
    mode=MODE,
)


# Used basically everywhere:
def make_id():
    return str(uuid.uuid4())


# Also used basically everywhere:
def write_output_to_file(output, directory, uuid):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path using the directory and UUID
    file_path = os.path.join(directory, f"{uuid}.txt")

    # Write the output to the file
    with open(file_path, "w") as file:
        file.write(output)

    print(f"Output written to {file_path}")


def fix_text(
    to_replace_arr, text
):  # see common errors across your input text? Give this an array of tuples ("bad string from input text","string it should be") and this'll fix up the raw inputs. Example for double spaces only: to_replace_arr = [("  ", "")]
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text

### OUTPUT PROCESSORS ###

def get_top_speakers(scene_text, n=2):
    """
    Extracts the character name from a scene text based on the rules provided.

    Parameters:
    - scene_text: The text containing the scene.
    - user_placeholder: The placeholder for the user's name in the scene text.

    Returns:
    - The name of the character that occurs most frequently, other than the user, based on the specified conditions.
    """
    from collections import Counter
    import math

    # Split the text into lines and initialize variables
    lines = scene_text.split("\n")
    character_counts = Counter()

    # Iterate over each line to find character names and count user occurrences
    for line in lines:
        # Check if line starts with a character name pattern
        if ":" in line:
            character_name = line.split(":", 1)[0].strip()
            character_counts[character_name] += 1


    # return a list of the n most common characters' names in descending order
    most_common_characters = character_counts.most_common(n)
    return [character for character, count in most_common_characters]



def stringify_chatlog_list(
    chatlog_list,
):  # converts from chatlog list back to a chatlog. Basically the inverse function (except the decision about what perspective to write in is lost in the first conversion to a list)
    return "\n\n".join(
        [f'{message["owner"]}: {message["content"]}' for message in chatlog_list]
    )


def parse_chatlog(chatlog, names):
    messages = []
    current_owner = None
    current_content = []

    for line in chatlog.split("\n"):
        # check if the line starts with any of the names in the names list
        if any([line.startswith(name + ":") for name in names]):
            if current_owner and current_content:
                messages.append(
                    {"owner": current_owner, "content": "\n".join(current_content)}
                )
                current_content = []
            current_owner = line.split(":")[0]
            current_content.append(line.split(":")[1])
        else:
            if current_owner:
                current_content.append(line)

    if current_owner and current_content:
        messages.append({"owner": current_owner, "content": "\n".join(current_content)})

    return [
        {"owner": message["owner"], "content": message["content"].strip()}
        for message in messages
    ]


def find_message_exceeding_threshold(messages, threshold):
    for i, msg in enumerate(messages):
        if count_tokens(msg["content"]) > threshold:
            return i
    return None


def parse_convo_messages(convo):
    names = get_top_speakers(convo)
    if not names:
        print("ERROR: names not found in convo, format horribly broken")
        return None
    
    chatlog_list = parse_chatlog(convo, names)
    truncated = False
    # print(chatlog_list)
    threshold_message_index = find_message_exceeding_threshold(chatlog_list, 650)
    if threshold_message_index:
        print("\n\TOO LONG MESSAGES DETECTED -- TRUNCATING CONVO")
        chatlog_list = chatlog_list[:threshold_message_index]
        truncated = True

    processed_convo_string = stringify_chatlog_list(
        chatlog_list
    )  # "\n\n".join([f'{message["owner"]}: {message["content"]}' for message in chatlog_list])

    print("==================================")
    return (processed_convo_string, truncated)


def parse_conversation_to_sharegpt_format(conversation):
    names = get_top_speakers(conversation)
    parsed = parse_chatlog(conversation, names)
    sharegpt_conversation = [
        {
            "from": "human" if idx % 2 == 1 else "gpt", # Hardcoded such that the first speaker is always AI
            "value": message["content"],
        }
        for idx, message in enumerate(parsed)
    ]
    return sharegpt_conversation

def random_select_items(lst, p_none=0.3, p_one=0.4, p_two=0.2, p_three=0.1):
    """
    Randomly selects 1–3 items from a list, or returns an empty list.

    Args:
        lst (list): The list from which items will be selected.
        p_none (float, optional): Probability of selecting no items. Default is 0.3.
        p_one (float, optional): Probability of selecting one item. Default is 0.4.
        p_two (float, optional): Probability of selecting two items. Default is 0.2.
        p_three (float, optional): Probability of selecting three items. Default is 0.1.

    Returns:
        list: A new list containing the selected items, or an empty list if no items were selected.
    """
    total_prob = p_none + p_one + p_two + p_three
    # Change: account for floating point math, give the following a margin of error:
    if abs(total_prob - 1) > 0.0001:
        raise ValueError("Probabilities must sum to 1.")

    random_value = random.random()
    if random_value < p_none:
        return []
    elif random_value < p_none + p_one:
        return [random.choice(lst)]
    elif random_value < p_none + p_one + p_two:
        return random.sample(lst, min(2,len(lst)))
    else:
        return random.sample(lst, min(3,len(lst)))