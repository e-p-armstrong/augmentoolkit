import asyncio

async def main():

    import logging
    import yaml


    with open("./config.yaml",'r') as f:
        config = yaml.safe_load(f)

    # "airoboros-l2-70b-3.1.2.Q4_K_M.gguf" <- recommended for the large logical model
    # "flatorcamaid-13b-v0.2.Q8_0.gguf" <- recommended for the normal logical model
    # A6000s on Vast.ai are a good choice for running this notebook

    if not config["SYSTEM"]["COMPLETION_MODE"] and config["SYSTEM"]["MODE"] == "aphrodite":
        raise Exception("Aphrodite engine mode MUST use completion prompts!")

    LOGICAL_MODEL = config["API"]["LOGICAL_MODEL"]

    LARGE_LOGICAL_MODEL = config["API"]["LARGE_LOGICAL_MODEL"]

    ASSISTANT_MODE = config["SYSTEM"]["ASSISTANT_MODE"]  # change to true if you want all conversations to be with an "AI language model" and not characters. Useful for more professional use cases.

    DOUBLE_CHECK_COUNTER = config["SYSTEM"]["DOUBLE_CHECK_COUNTER"]  # Set to 1 to check outputs only once; set to 2 to check twice; set to 3 to check thrice, etc. Set to 0 to break everything in vet_question_loop() and elsewhere. Set to -1 and cause the universe to implode?

    USE_SUBSET = config["SYSTEM"]["USE_SUBSET"] # Set to True if you want to use only a small subset of the text, to test whether it plays nicely with the current setup of the notebook

    REARRANGEMENTS_TO_TAKE = config["SYSTEM"]["REARRANGEMENTS_TO_TAKE"] # How many of the possible permutations of tuples in a group to take and make multiturn convs out of. Adjust higher to get more data out of less text, but it might be a bit repetitive. NOTE your eval loss will be basically worthless if you aren't careful with how you shuffle your dataset when you're about to train.

    USE_FILENAMES = config["SYSTEM"]["USE_FILENAMES"] # Turn on if you want the model to use the names of your files as additional context (this is what original Augmentoolkit does). Useful if you have a small number of large input files grouped by subject matter, IE books. Turn off if you have a large number of files with meaningless names.

    CONCURRENCY_LIMIT = config["SYSTEM"]["CONCURRENCY_LIMIT"]  # Adjust this number based on the rate limit constraints of your api

    API_KEY = config["API"]["API_KEY"]

    BASE_URL = config["API"]["BASE_URL"]# Augmentoolkit-API should also be compatible with any other API provider that accepts OAI-style requests

    COMPLETION_MODE = config["SYSTEM"]["COMPLETION_MODE"]

    MODE = config["SYSTEM"]["MODE"]

    LOG_LEVEL = logging.INFO

    source_texts = [ # add your texts here
        "./raw_txt_input/Simple Sabotage, by the Office of Strategic Services, published 1944.txt",
    ]

    import os
    import uuid

    # This is in no way best practices, but all my prompts being searchable and separate files is a good way to make my life easier.
    import pkgutil
    import importlib
    import sys
    from tqdm import asyncio as tqdmasyncio
    import asyncio

    # Set up rate-limit-conscious functions
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_task_with_limit(task):
        async with semaphore:
            # Run your task here
            return await task


    # We have to define this up here so that two-step generation works, you'll see later.
    multi_turn_convs_info_dir = config["PATH"]["OUTPUT"] + "/multi_turn_convs_info"  # we generate all the information fed to the multiturn prompt, and generate the actual multiturn prompt, separately; since every step but the last is capable of being done by a 13b

    sys.path.append("./generation_functions")
    sys.path.append("./control_flow_functions")

    import augmentoolkit.generation_functions as generation_functions  # This is the package directory
    from augmentoolkit.control_flow_functions import control_flow_functions

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
        model=LOGICAL_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL, 
        mode=MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )


    from transformers import AutoTokenizer
    import re
    from tqdm import tqdm
    import nltk

    nltk.download("punkt")
    from nltk.tokenize import sent_tokenize

    tokenizer = AutoTokenizer.from_pretrained(
        "Gryphe/MythoMax-L2-13b"
    )  # It doesn't matter what model goes here, really

    sentence_chunks = []
    for source_text in source_texts:
        sentence_chunks += control_flow_functions.sentence_chunking_algorithm(source_text, tokenizer)

    conversions = [("\n", " "), ("  ", " ")]

    paragraphs_processed = [
        (control_flow_functions.fix_text(conversions, seq[0]), seq[1]) for seq in sentence_chunks
    ]


    print(paragraphs_processed[:3])

    import json
    import os
    from tqdm import tqdm
    import asyncio

    # Create directory if it doesn't exist
    output_dir = config["PATH"]["OUTPUT"] + "/worthy_for_questions"
    os.makedirs(output_dir, exist_ok=True)

    # Determine which paragraphs are worthy of making questions from
    judged_worthy_for_questions = []

    await control_flow_functions.filter_all_questions(paragraphs_processed, judged_worthy_for_questions, engine_wrapper, output_dir, take_subset=USE_SUBSET, use_filenames=False, rtwl=run_task_with_limit, completion_mode=COMPLETION_MODE,logging_level=LOG_LEVEL)


    filtered_worthy_for_questions = control_flow_functions.filter_and_graph(judged_worthy_for_questions)



    print(filtered_worthy_for_questions[0])

    import json
    import os
    import glob

    # Directory for QA tuples
    qa_tuples_dir = config["PATH"]["OUTPUT"]  + "/qatuples_raw"
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
        tasks = [control_flow_functions.generate_qatuples_from_para(
            idx,
            para,
            engine_wrapper=engine_wrapper,
            vetted_qa_tuples=vetted_qa_tuples,
            qa_tuples_dir=qa_tuples_dir,
            double_check_counter=DOUBLE_CHECK_COUNTER,
            use_filenames=USE_FILENAMES,
            completion_mode=COMPLETION_MODE,
            logging_level=LOG_LEVEL) for idx,para in enumerate(filtered_worthy_for_questions)]
        limited_tasks_qgen = [run_task_with_limit(task) for task in tasks]
        for future in tqdmasyncio.tqdm.as_completed(limited_tasks_qgen):
                await future
        
    print(
        "-------------- QUESTIONS CREATED ------------- STATS SO FAR (may be wrong if run was continued from interruption):"
    )
    nones = list(filter(lambda x: x[0] is None, vetted_qa_tuples))
    print(f"Nones: {len(nones)}")
    print(f"Non-nones: {len(vetted_qa_tuples) - len(nones)}")
    print(f"Total: {len(vetted_qa_tuples)}")
    # filter out all None values
    vetted_qa_tuples = [qa for qa in vetted_qa_tuples if qa[0] is not None]
    print("---------------- ONTO EXAMPLES GENERATION-------------------")


    # Check for and fix the common mistake: mentioning "the text".
    writepath = config["PATH"]["OUTPUT"] + "/qatuples_revised"
    import json

    # Assuming vetted_qa_tuples is a list that might or might not exist
    try:
        _ = vetted_qa_tuples
    except NameError:
        vetted_qa_tuples = []

    # Load all files at the start if vetted_qa_tuples is empty
    if not vetted_qa_tuples:
        # Check if the directory exists
        if os.path.exists(writepath):
            # List all files in directory
            for file_name in os.listdir(writepath):
                file_path = os.path.join(writepath, file_name)
                try: # for each file already generated, see if it succeeded or failed; if it succeeded, append its contents; if it failed, append None for stats logging
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        print(f"Loading file: {file_path}")
                        if content == "failed":
                            vetted_qa_tuples.append(None)
                        else:
                            try:
                                data = json.loads(content)
                                vetted_qa_tuples.append(
                                    (data[0], data[1], data[2], data[3])
                                )
                            except json.JSONDecodeError:
                                print("JSON decode error with the contents:", content)
                                vetted_qa_tuples.append(None)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    else:
        old_tuples = vetted_qa_tuples.copy()
        tasks = [control_flow_functions.repair_qatuple_context(idx, tup, engine_wrapper, writepath, vetted_qa_tuples,use_filenames=USE_FILENAMES) for idx, tup in enumerate(vetted_qa_tuples)]
        limited_tasks_qcorrection = [run_task_with_limit(task) for task in tasks]
        for future in tqdmasyncio.tqdm.as_completed(limited_tasks_qcorrection): 
            await future



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



    import os

    if not os.path.exists(multi_turn_convs_info_dir):
        os.makedirs(multi_turn_convs_info_dir)


    # In[ ]:


    import json
    import random
    import itertools

    multi_turn_convs_info = []


    tasks = [control_flow_functions.create_info(idx,group,engine_wrapper, ASSISTANT_MODE, multi_turn_convs_info,multi_turn_convs_info_dir, rearrangements_to_take=REARRANGEMENTS_TO_TAKE,use_filenames=USE_FILENAMES, completion_mode=COMPLETION_MODE, logging_level=LOG_LEVEL) for idx,group in enumerate(qa_tuples_by_paragraph)]
    limited_tasks_infocreation = [run_task_with_limit(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks_infocreation):
        await future



    # Initialize API Client
    engine_wrapper = EngineWrapper(
        model=LARGE_LOGICAL_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL, 
        mode=MODE,
        # quantization="gptq" # modify if you want to do stuff with the aphrodite branch
    )


    import os
    import json

    convs_info = control_flow_functions.read_json_files_info(multi_turn_convs_info_dir)


    import os
    import json
    import random
    import itertools
    import asyncio

    multi_turn_convs_dir = config["PATH"]["OUTPUT"] + "/multi_turn_convs"
    if not os.path.exists(multi_turn_convs_dir):
        os.makedirs(multi_turn_convs_dir)

    multi_turn_convs = []

    tasks = [control_flow_functions.create_conversation(idx,info, engine_wrapper, multi_turn_convs, multi_turn_convs_dir, assistant_mode=ASSISTANT_MODE, completion_mode=COMPLETION_MODE, logging_level=LOG_LEVEL) for idx,info in enumerate(convs_info)]
    limited_tasks_convwriting = [run_task_with_limit(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks_convwriting):
        await future


    import os
    import json

    # Make ShareGPT-format dataset (I think, still need verification it actually works)
    control_flow_functions.convert_directory_to_list(config["PATH"]["OUTPUT"] + "/multi_turn_convs/")
    # Make dataset in a format that has all the information. See README for details on this format.
    control_flow_functions.convert_directory_and_process_conversations(config["PATH"]["OUTPUT"] + "/multi_turn_convs/")



    with open(config["PATH"]["OUTPUT"] + "/processed_master_list.json") as f:
        first = f.read()
        data = json.loads(first)


    # For curiosity's sake, you can find out how many lines of dialogue you generated
    def filter_and_flatten(lst):
        # Initialize an empty list to hold the flattened elements
        flat_list = []

        # Loop through each sublist in the main list
        for sublst in lst:
            # Check if the first element of the sublist is itself a list (subsublist1)
            if isinstance(sublst[0], list):
                # Extend the flat_list with the elements from subsublist1
                flat_list.extend(sublst[0])

        return flat_list


asyncio.run(main())