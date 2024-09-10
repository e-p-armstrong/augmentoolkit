import random
import traceback
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from BOILERPLATE_TO_MAKE_YOUR_OWN_PIPELINE.steps import API_KEY_A, API_KEY_B, BASE_URL_A, BASE_URL_B, CONCURRENCY_LIMIT, LOGICAL_MODEL_A, LOGICAL_MODEL_B, MODE_A, MODE_B, add_key, chunking_algorithm, count_tokens, make_id


import nltk
from tqdm import asyncio as tqdmasyncio


import asyncio
import glob
import logging
import os
import sys
import time
import yaml

config_path = os.environ["CONFIG_PATH"]
with open (config_path, "r") as file:
    config = yaml.safe_load(file)

WORK_IN_PHASES = bool(config["PHASES"]["WORK_IN_PHASES"])
PHASE_INDEX = int(config["PHASES"]["PHASE_INDEX"])
USE_SUBSET = bool(config["SYSTEM"]["USE_SUBSET"])
SUBSET_SIZE = int(config["SYSTEM"]["SUBSET_SIZE"])
CHUNK_SIZE = int(config["SYSTEM"]["CHUNK_SIZE"])
INPUT = config["PATH"]["INPUT"]


async def main():
    # NOTE Load the source texts
    print("Welcome to your test pipeline!")
    print(f"Input folder: {INPUT}")
    start_time = time.time()
    print("Begun")

    # Set up rate-limit-conscious functions
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    extensions = [".txt", ".md"]

    source_texts = []
    for extension in extensions:
      path = f"{INPUT}/**/*" + extension
      source_texts = source_texts + glob.glob(path, recursive=True)

    if source_texts:
        print(source_texts)
    else:
        print(f"No source texts found in: {INPUT}")

    # NOTE Initialize the Engine (or API client)
    engine_wrapper = EngineWrapper(
        model=LOGICAL_MODEL_A,
        api_key=API_KEY_A,
        base_url=BASE_URL_A,
        mode=MODE_A,
    )

    engine_wrapper_large = EngineWrapper(
        model=LOGICAL_MODEL_B,
        api_key=API_KEY_B,
        base_url=BASE_URL_B,
        mode=MODE_B,
    )

    # any HF path to a transformer model will do, as long as it has a tokenizer

    sentence_chunks = []
    for source_text in source_texts:
        sentence_chunks += chunking_algorithm(source_text, max_token_length=CHUNK_SIZE)

    # NOTE Generate the data
    output_list = []
    data_generations_tasks = [add_key(input_data=chunk, engine_wrapper=engine_wrapper_large, idx=idx, output_list=output_list) for idx, chunk in enumerate(sentence_chunks)]
    coroutines = [run_task_with_limit(task) for task in data_generations_tasks]
    for future in tqdmasyncio.tqdm.as_completed(coroutines):
        await future

    print(f"Time taken: {time.time() - start_time}")
    print("You generated some data! Check the output folder for the results.")
    print("here's one of the results: ")
    print(output_list[0])
    
asyncio.run(main())
