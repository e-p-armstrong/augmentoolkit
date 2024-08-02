import asyncio

import augmentoolkit.utils.group_by_text

# created with nbconvert, minimally cleaned up


# NOTE NOTEBOOK SETTINGS AND CONSTANTS (some script file constants are in generation_functions/constants.py)

# Put your desired quant of your desired model in the relevant directories

import logging
import yaml
import glob
from augmentoolkit.utils.group_by_text import group_by_text
from augmentoolkit.core import steps
import os

import augmentoolkit.utils.sentence_chunking_algorithm

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.exists(config["PATH"]["OUTPUT"]):
    os.makedirs(config["PATH"]["OUTPUT"])

# "airoboros-l2-70b-3.1.2.Q4_K_M.gguf" <- recommended for the large logical model
# "flatorcamaid-13b-v0.2.Q8_0.gguf" <- recommended for the normal logical model
# A6000s on Vast.ai are a good choice for running this notebook

if (
    not config["SYSTEM"]["COMPLETION_MODE"]
    and config["SYSTEM"]["MODE"] == "aphrodite"
):
    raise Exception("Aphrodite engine mode MUST use completion prompts!")

LOGICAL_MODEL = config["API"]["LOGICAL_MODEL"]

LARGE_LOGICAL_MODEL = config["API"]["LARGE_LOGICAL_MODEL"]

DOUBLE_CHECK_COUNTER = config["SYSTEM"][
    "DOUBLE_CHECK_COUNTER"
]  # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. Set to 0 to break everything in vet_question_loop() and elsewhere. Set to -1 and cause the universe to implode?

USE_SUBSET = config["SYSTEM"][
    "USE_SUBSET"
]  # Set to True if you want to use only a small subset of the text, to test whether it plays nicely with the current setup of the notebook

SUBSET_SIZE = config["SYSTEM"]["SUBSET_SIZE"]  # Set to the number of chunks you want to use if you're using a subset. If you're not using a subset, this will be ignored.

USE_FILENAMES = config["SYSTEM"][
    "USE_FILENAMES"
]  # Turn on if you want the model to use the names of your files as additional context (this is what original Augmentoolkit does). Useful if you have a small number of large input files grouped by subject matter, IE books. Turn off if you have a large number of files with meaningless names.

CONCURRENCY_LIMIT = config["SYSTEM"][
    "CONCURRENCY_LIMIT"
]  # Adjust this number based on the rate limit constraints of your api

API_KEY = config["API"]["API_KEY"]

BASE_URL = config["API"][
    "BASE_URL"
]  # Augmentoolkit-API should also be compatible with any other API provider that accepts OAI-style requests

COMPLETION_MODE = config["SYSTEM"]["COMPLETION_MODE"]

MODE = config["SYSTEM"]["MODE"]

LOG_LEVEL = logging.INFO

INPUT_FOLDER = config["PATH"]["INPUT"]

CONVERSATION_INSTRUCTIONS = config["SYSTEM"][
    "CONVERSATION_INSTRUCTIONS"
]

extensions = [".txt", ".md"]

source_texts = []
for extension in extensions:
    path = f"{INPUT_FOLDER}/**/*" + extension
    source_texts = source_texts + glob.glob(path, recursive=True)

print(source_texts)

import sys

# We have to define this up here so that two-step generation works, you'll see later.
multi_turn_convs_info_dir = (
    config["PATH"]["OUTPUT"] + "/multi_turn_convs_info"
)  # we generate all the information fed to the multiturn prompt, and generate the actual multiturn prompt, separately; since every step but the last is capable of being done by a 13b

sys.path.append("./generation_functions")
sys.path.append("./control_flow_functions")

from augmentoolkit.core import steps

sentence_chunks = []
for source_text in source_texts:
    sentence_chunks += augmentoolkit.utils.sentence_chunking_algorithm.sentence_chunking_algorithm(
        source_text, config["SYSTEM"]["CHUNK_SIZE"]
    )

conversions = [("\n", " "), ("  ", " ")]

paragraphs_processed = [
    (steps.fix_text(conversions, seq[0]), seq[1])
    for seq in sentence_chunks
]

print("LENGTH OF ALL CHUNKS, NOT FILTERED")
print(len(paragraphs_processed))