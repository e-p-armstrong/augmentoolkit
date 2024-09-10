import random
import itertools
import os
import asyncio
import json
import re
from typing import List
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
import logging
from math import ceil
import traceback
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
import uuid
import yaml
import nltk
from augmentoolkit.utils import parse_string_list
from augmentoolkit.utils.parse_bool import parse_bool


nltk.download('punkt_tab')

tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
    )

def count_tokens(message):
    return len(tokenizer.encode(message))

config_path = os.environ["CONFIG_PATH"]
with open (config_path, "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT = os.path.abspath(obj_conf["PATH"]["OUTPUT"])
DEFAULT_PROMPTS = os.path.abspath(obj_conf["PATH"]["DEFAULT_PROMPTS"])
PROMPTS = os.path.abspath(obj_conf["PATH"]["PROMPTS"])
COMPLETION_MODE = parse_bool(obj_conf["SYSTEM"]["COMPLETION_MODE"])
LOGGING_LEVEL = logging.INFO
LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
API_KEY_A = obj_conf["API"]["API_KEY_A"]
API_KEY_B = obj_conf["API"]["API_KEY_B"]
BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
MODE_A = obj_conf["API"]["MODE_A"]
MODE_B = obj_conf["API"]["MODE_B"]
CONCURRENCY_LIMIT = int(obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"])
USE_STOP = parse_bool(obj_conf["SYSTEM"]["STOP"])
USE_MIN_P = parse_bool(obj_conf["SYSTEM"]["USE_MIN_P"])

## Chunking Logic for Raw Input Text ##
def chunking_algorithm(file_path, max_token_length=1500):
    """
    This function takes a plaintext file and chunks it into paragraphs or sentences if the paragraph exceeds max_token_length.

    :param file_path: Path to the plaintext file
    :param tokenizer: SentencePiece tokenizer
    :param max_token_length: The maximum token length for a chunk
    :return: List of chunks with source text information
    """
    chunks_with_source = []
    current_chunk = []
    token_count = 0
    source_name = file_path.replace(".txt", "")


    with open(file_path, "r", encoding="utf-8",errors='ignore') as f:
        content = f.read()
        
    paragraphs = content.split('\n\n')  # Assuming paragraphs are separated by two newlines # TODO change so that if the length is 1 after this, split by tabs instead

    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove leading and trailing whitespace
        if not paragraph:  # Skip empty paragraphs
            continue
        
        paragraph_token_count = count_tokens(paragraph)
        
        # Check if the paragraph itself exceeds the max token length
        if paragraph_token_count > max_token_length:
            # Fallback to sentence chunking for this paragraph
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})

    return chunks_with_source
    
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
    

# A pipeline step to get you started

def validate_output(output, input_data): # some random validation function
    if input_data["chunk"][0] in output:
        return True
    else:
        print("FAILED ")
        print(input_data["chunk"][0])
        print(output)
        print("----")
        return False

test_prompt_path = "test_prompt"

class TestGenerator(PipelineStep): # pipeline steps store the settings and the prompt, and prevent us from having to repeat the "read previous output" code among other things
    def __init__(self):
        super().__init__(
            prompt_folder=PROMPTS,
            default_prompt_folder=DEFAULT_PROMPTS,
            prompt_path=test_prompt_path,
            sampling_params={
                "max_tokens": 2000,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "### Information",
                    "## Information",
                    "## Instruction",
                    "Name:",
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                ],
                "temperature": 0.8,
                # "top_k": -1,
                "top_p": 1,
                # "min_p": 0.6,
            },
            output_dir=OUTPUT,
            output_subdir="test_output",
            intermediate_output_path="intermediate_generations",
            save_path="saved_readable_generations",
            result_key="test",
            use_stop=USE_STOP,
            completion_mode=COMPLETION_MODE,
            validation_function=validate_output,
            max_retries=3,
        )
        
test_generator = TestGenerator() # make the singleton

async def add_key( # this is an example of a function you might use to generate data and add it to a new output list
    idx, 
    input_data,
    engine_wrapper,
    output_list
):
    await test_generator.run(idx, input_data=input_data, engine_wrapper=engine_wrapper, output_list=output_list)