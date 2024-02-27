import yaml
import os
import uuid

# This is in no way best practices, but all my prompts being searchable and separate files is a good way to make my life easier.
import pkgutil
import importlib
import sys
from tqdm import asyncio as tqdmasyncio
import asyncio
import json
import os
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import nltk
import glob


sys.path.append("./generation_functions")
sys.path.append("./control_flow_functions")

import augmentoolkit.generation_functions as generation_functions  # This is the package directory
from augmentoolkit.control_flow_functions import control_flow_functions

with open("config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

folder_path = obj_conf["PATH"]["INPUT"]
source_texts = []

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            source_texts.append(file_path)


import os


def convert_to_utf8(file_path):
    with open(file_path, "r", encoding="latin-1") as f:
        content = f.read()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def convert_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Converting {file_path} to UTF-8...")
                convert_to_utf8(file_path)


directory = folder_path
convert_files_in_directory(directory)


# Set up rate-limit-conscious functions
semaphore = asyncio.Semaphore(obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"])


async def run_task_with_limit(task):
    async with semaphore:
        # Run your task here
        return await task


# We have to define this up here so that two-step generation works, you'll see later.
multi_turn_convs_info_dir = (
    obj_conf["PATH"]["OUTPUT"] + "/multi_turn_convs_info"
)  # we generate all the information fed to the multiturn prompt, and generate the actual multiturn prompt, separately; since every step but the last is capable of being done by a 13b


# First, import all modules so they can be reloaded
for _, module_name, _ in pkgutil.iter_modules(
    generation_functions.__path__, generation_functions.__name__ + "."
):
    importlib.import_module(module_name)

# Now, reload each module and import all callable attributes
for _, module_name, _ in pkgutil.iter_modules(
    generation_functions.__path__, generation_functions.__name__ + "."
):
    # Reload the module
    module = importlib.reload(sys.modules[module_name])
    # Iterate through each attribute in the reloaded module
    for attribute_name in dir(module):
        # Retrieve the attribute
        attribute = getattr(module, attribute_name)
        if callable(attribute):
            # If it's callable, it's a function or class, so you set it in the globals dictionary
            globals()[attribute_name] = attribute


# Initialize API Client
engine_wrapper = EngineWrapper(
    model=obj_conf["API"]["LOGICAL_MODEL"],
    api_key=obj_conf["API"]["API_KEY"],
    base_url=obj_conf["API"]["BASE_URL"],
)


nltk.download("punkt")
from nltk.tokenize import sent_tokenize

tokenizer = AutoTokenizer.from_pretrained(
    "Gryphe/MythoMax-L2-13b"
)  # It doesn't matter what model goes here, really

sentence_chunks = []
for source_text in source_texts:
    sentence_chunks += control_flow_functions.sentence_chunking_algorithm(
        source_text, tokenizer
    )

conversions = [("\n", " "), ("  ", " ")]

paragraphs_processed = [
    (control_flow_functions.fix_text(conversions, seq[0]), seq[1])
    for seq in sentence_chunks
]

print(len(paragraphs_processed))
print(paragraphs_processed[1])


# Create directory if it doesn't exist
output_dir = obj_conf["PATH"]["OUTPUT"] + "/worthy_for_questions"
os.makedirs(output_dir, exist_ok=True)

# Determine which paragraphs are worthy of making questions from
judged_worthy_for_questions = []


async def filter_paragraphs(
    paragraphs_processed,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    take_subset=True,
    use_filenames=True,
    rtwl=run_task_with_limit,
):
    await control_flow_functions.filter_all_questions(
        paragraphs_processed,
        judged_worthy_for_questions,
        engine_wrapper,
        output_dir,
        take_subset=take_subset,
        use_filenames=use_filenames,
        rtwl=rtwl,
    )


# Call the async function
asyncio.run(
    filter_paragraphs(
        paragraphs_processed,
        judged_worthy_for_questions,
        engine_wrapper,
        output_dir,
        take_subset=obj_conf["SYSTEM"]["USE_SUBSET"],
        use_filenames=obj_conf["SYSTEM"]["USE_FILE_NAMES"],
        rtwl=run_task_with_limit,
    )
)

# ----- NEEDS SIMILAR FUNCTION JUST W/O PLOTTING
filtered_worthy_for_questions = control_flow_functions.filter_and_graph(
    judged_worthy_for_questions
)

print(filtered_worthy_for_questions[0])


# Directory for QA tuples
qa_tuples_dir = obj_conf["PATH"]["OUTPUT"] + "/qatuples_raw"
if not os.path.exists(qa_tuples_dir):
    os.makedirs(qa_tuples_dir)

# Initialize vetted_qa_tuples
vetted_qa_tuples = []  # tuple list of qa tuples that have been judged good

# Attempt to initialize filtered_worthy_for_questions
try:
    _ = filtered_worthy_for_questions
except NameError:
    filtered_worthy_for_questions = []

if not filtered_worthy_for_questions:
    # Load all files in the qa_tuples_dir if filtered_worthy_for_questions is not initialized
    existing_files = glob.glob(os.path.join(qa_tuples_dir, "*.json"))
    for file_path in existing_files:
        with open(file_path, "r") as file:
            qa_tuple = tuple(json.load(file))
            print(f"Loaded {file}")
        vetted_qa_tuples.append(qa_tuple)
else:
    tasks = [
        control_flow_functions.generate_qatuples_from_para(
            idx,
            para,
            engine_wrapper=engine_wrapper,
            vetted_qa_tuples=vetted_qa_tuples,
            qa_tuples_dir=qa_tuples_dir,
            double_check_counter=obj_conf["SYSTEM"]["DOUBLE_CHECK_COUNTER"],
            use_filenames=obj_conf["SYSTEM"]["USE_FILE_NAMES"],
        )
        for idx, para in enumerate(filtered_worthy_for_questions)
    ]
    limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]

    async def run_tasks(limited_tasks_qgen):
        for future in tqdmasyncio.tqdm.as_completed(limited_tasks_qgen):
            await future

    asyncio.run(run_tasks(limited_tasks_qgen))

    # Print stats related to revised qatuples, and filter out nones (questions that were unanswerable due to lack of context).
import json
import os

print("-------------- QUESTIONS REVISED ------------- STATS SO FAR:")
nones = list(filter(lambda x: x is None, vetted_qa_tuples))
print(f"Nones: {len(nones)}")
print(f"Non-nones: {len(vetted_qa_tuples) - len(nones)}")
print(f"Total: {len(vetted_qa_tuples)}")
# filter out all None values
vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa is not None]
print("---------------- ONTO EXAMPLES GENERATION-------------------")


qa_tuples_by_paragraph = control_flow_functions.group_by_text(vetted_qa_tuples)


if not os.path.exists(multi_turn_convs_info_dir):
    os.makedirs(multi_turn_convs_info_dir)


# In[ ]:


import json
import random
import itertools

multi_turn_convs_info = []


tasks = [
    control_flow_functions.create_info(
        idx,
        group,
        engine_wrapper,
        obj_conf["SYSTEM"]["ASSISTANT_MODE"],
        multi_turn_convs_info,
        multi_turn_convs_info_dir,
        obj_conf["SYSTEM"]["REARRANGEMENTS_TO_TAKE"],
        obj_conf["SYSTEM"]["USE_FILE_NAMES"],
    )
    for idx, group in enumerate(qa_tuples_by_paragraph)
]


async def limited_tasks():
    limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]

    async def process_tasks(future, as_completed, limited_tasks_infocreation):
        for future in tqdmasyncio.tqdm(as_completed(limited_tasks_infocreation)):
            await future

    # Run the async function
    await process_tasks()


# Call the main async function
asyncio.run(limited_tasks())
