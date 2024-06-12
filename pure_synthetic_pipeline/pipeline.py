
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


async def generate_data(args, id):
    generate_conversation_path = (
        "generate_conversation.txt" if COMPLETION_MODE else "generate_conversation.yaml"
    )
    conversation_generator = GenerationStep(
        prompt_path=generate_conversation_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_large,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
    )
    
    result, full_output = await conversation_generator.generate(args)
    write_output_to_file(full_output, OUTPUT_FOLDER + "/conversation_generation", id)
    return result


### END GENERATORS

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
async def run_task_with_limit(task):
    async with semaphore:
        return await task

# NEED to have a logging option


# AI-GENERATED via script

async def generate_conv(input_dict=None, output_file=None):
    id = make_id()
    try:
        data_output = await validate_generation(gen_func=generate_data, validation_functions=[validate_repetition_callback(25, 3, 400)], retries=2, gen_func_args=[input_dict, id])
    except:
        return
    conversation_sharegpt = {"conversations": parse_conversation_to_sharegpt_format(data_output)}
    with open(output_file, "a") as f:
        f.write(json.dumps(conversation_sharegpt) + "\n")
###

async def main():
    print(obj_conf)
    output_file = "formatted_conversations.jsonl"
    mutators = ['The student has a learning disability, such as dyslexia, and may need additional accommodations or support.', 'The conversation takes place in a specific cultural or historical context, such as a unit on ancient Greece or the American Renaissance.', 'The student is part of a group or class, and the AI is facilitating a discussion among multiple students.', 'The student is using the AI as a resource for a school project or assignment, and needs help with research or analysis.', 'The student has a strong emotional connection to a particular poem or poet, and wants to explore that connection with the AI.']
    exclusives = ['The student is interested in learning about Shakespearean sonnets.', 'The student wants to explore the works of a specific poet, such as Emily Dickinson or Robert Frost.', 'The student is struggling to understand a particular poem or concept, and needs additional support.', 'The student is eager to learn about different poetic forms, such as haikus or free verse.', 'The student has a favorite poem or poet, and wants to discuss it with the AI.']
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

        student_age = random.randint(6,11)
        student_name = fake.name()
        ai_name = fake.first_name()

        ### END AI-WRITTEN INPUT FIELDS (anything that needs actual AI to write it for each sample will be generated as part of the convo generation process)
        
        # Code places the AI-written input fields into the input object
        input_object = {
            "exclusive": exclusive,
            "mutators": mutators if mutators else "None",
            ### AI-written input fields are added below
            "student_age": student_age,
            "student_name": student_name,
            "ai_name": ai_name,
            ###
        }
        
        task = asyncio.create_task(
            run_task_with_limit(
                generate_conv(input_dict=input_object, output_file=output_file)
            )
        )
        tasks.append(task)

    # await asyncio.gather(*tasks)
    await tqdm_asyncio.gather(*tasks)


asyncio.run(main())
