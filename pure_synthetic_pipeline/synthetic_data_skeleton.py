file_template = """
import json
import asyncio
import os
import time
from apify_client import ApifyClient
import random
import yaml
from tqdm.asyncio import tqdm_asyncio
import re

from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.utillity_functions import (
    LOGICAL_MODEL_A,
    LOGICAL_MODEL_B,
    API_KEY_A,
    API_KEY_B,
    BASE_URL_A,
    BASE_URL_B,
    MODE,
    CONCURRENCY_LIMIT,
    COMPLETION_MODE,
    DEFAULT_PROMPT_PATH,
    OUTPUT_FOLDER,
    LOG_EVERYTHING,
    write_output_to_file,
    make_id,
    parse_conversation_to_sharegpt_format,
    random_select_items
)
from gen_engine_core.validation_functions import *
from faker import Faker
from gen_engine_core.generation_functions.generation_step_class import GenerationStep

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

# Set up faker for generating input fields
fake = Faker()

# Need test cases to produce data pipelines for

engine_wrapper_large = EngineWrapper(
    model=LOGICAL_MODEL_B,
    api_key=API_KEY_B,
    base_url=BASE_URL_B,
    mode=MODE,
)


### GENERATORS

{generators}

### END GENERATORS

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
async def run_task_with_limit(task):
    async with semaphore:
        return await task

# NEED to have a logging option


# AI-GENERATED via script
{data_routing}
###

async def main():
    print(obj_conf)
    output_file = "formatted_conversations.jsonl"
    mutators = {mutators}
    exclusives = {exclusives}
    count = obj_conf["REQUIREMENTS"]["COUNT"]
    
    target_set_size = count
    
    tasks = [] # For Asyncio, NOT for the AI
    for _ in range(target_set_size):
        exclusive = random.choice(exclusives)
        if mutators:
            mutators = " && ".join(
                random_select_items(mutators)
            )
        else:
            mutators = None

        ### FAKE INPUT FIELDS (AI WRITES THESE)

{input_fields}

        ### END AI-WRITTEN INPUT FIELDS (anything that needs actual AI to write it for each sample will be generated as part of the convo generation process)
        
        # Code places the AI-written input fields into the input object
        input_object = {{
            "exclusive": exclusive,
            "mutators": mutators if mutators else "None",
            ### AI-written input fields are added below
{input_field_dict_items}
            ###
        }}
        
        task = asyncio.create_task(
            run_task_with_limit(
                generate_conv(input_dict=input_object, output_file=output_file)
            )
        )
        tasks.append(task)

    # await asyncio.gather(*tasks)
    await tqdm_asyncio.gather(*tasks)


asyncio.run(main())
"""