from collections import defaultdict
import random
import re
import traceback
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import (
    PipelineStep,
    filter_out_nonpresent_keys,
)
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from augmentoolkit.utils.write_output_to_file import write_output_to_file
import json
import os
from typing import Dict, List

from generation.core_components.chunking import (
    count_tokens,
    count_total_tokens,
    format_conversation_history,
    process_sharegpt_conversations,
    process_sharegpt_pairs,
    read_sharegpt_conversations,
    read_sharegpt_pairs,
    slice_conversation_history,
    subset_text_list,
)
from generation.core_components.filter_chunks import filter_out_failed_items_dict
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from tqdm import asyncio as tqdmasyncio


import asyncio
import glob
import logging
import os
import sys

# import time
import yaml
import json

from redis_config import set_progress

# # from datetime import datetime


def save_dict_to_jsonl(
    big_dict: Dict[str, dict], cot_preface: str, cot_suffix: str, output_path: str
):
    """
    Saves conversations from big_dict to JSONL files, restructuring messages
    with provided chain-of-thought preface and suffix.

    Args:
        big_dict (Dict[str, dict]): Dictionary containing conversations.
        cot_preface (str): Prefix for chain-of-thought explanation.
        cot_suffix (str): Suffix for chain-of-thought explanation.
        output_path (str): Directory to save output JSONL files.
    """

    # For each source file (dict)
    # For each conversation idx (dict inside source file)
    # add to it in order of pairs etc.

    all_conversations = []  # a list of sharegpts from across all files

    source_files = {}
    for subdict in big_dict.values():
        if subdict["source_file"] not in source_files:
            source_files[subdict["source_file"]] = {}  # dict of conversation indices

    for file in source_files.keys():
        # filter subdicts in the big dict to only include those of from that source file. And sort them by the conv_idx key.
        dicts_from_source_file = sorted(
            [d for i, d in big_dict.items() if d["source_file"] == file],
            key=lambda x: x["conv_idx"],
        )

        file_conversations = {}
        # iterate over this and create the conversation lists directly. Sort at the end.
        for d in dicts_from_source_file:
            if d["conv_idx"] not in file_conversations:
                file_conversations[d["conv_idx"]] = [d]
            else:
                file_conversations[d["conv_idx"]].append(d)

        for k, v in file_conversations.items():
            file_conversations[k] = sorted(v, key=lambda x: x["pair_idx"])

        file_conversations_list = []
        for k, v in file_conversations.items():
            # we want to save a jsonl to the source file of this. First convert each thing to a proper conversations: conversation and append it to both all lists and file list.
            l = []
            system_message = None
            for list_item in v:
                if "system" in list_item and list_item["system"] and not system_message:
                    system_message = list_item["system"]
                l.append({"from": "human", "value": list_item["human"]})
                thought_process = list_item["thought_process"]
                gpt = list_item["gpt"]
                l.append(
                    {
                        "from": "gpt",
                        "value": f"{cot_preface}\n{thought_process}\n{cot_suffix}\n{gpt}",
                    }
                )
            l.insert(0, {"from": "system", "value": system_message})
            conversation_obj = {"conversations": l}
            file_conversations_list.append(conversation_obj)
            all_conversations.append(conversation_obj)

        with open(
            os.path.join(output_path, os.path.basename(file)), "w", encoding="utf-8"
        ) as f:
            for obj in file_conversations_list:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(
        os.path.join(output_path, "final_total_output.jsonl"), "w", encoding="utf-8"
    ) as f:
        for obj in all_conversations:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return all_conversations


def check_token_length(output, input_data):
    if (
        count_tokens(output) > 4500
    ):  # we can check for errors by assuming that anything past a certain point is probably going to have been looping to no end. If it looped then it failed. So we validate on not generating up to the max token length. Actually we can make that a generic catch in the engine wrapper...
        return {
            "result": False,
            "message": "Response was too long; assuming that it fell into an infinite loop; failing validation...",
        }
    return {"result": True, "message": "Success"}


def process_thought_process_tags(text):
    """
    Removes <think> tags and their content, and extracts content from <thought_process> tags.

    Args:
        text (str): Input text containing tags

    Returns:
        str: Text with <think> tags removed and <thought_process> content extracted
    """
    # Remove <think> tags and their content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Extract content from <thought_process> tags
    thought_process_match = re.search(
        r"<thought_process>(.*?)</thought_process>", text, flags=re.DOTALL
    )
    if thought_process_match:
        return thought_process_match.group(1).strip()
    return text


async def transform_generic_data_pipeline(
    large_api_key,
    small_api_key,
    large_base_url,
    small_base_url,
    large_model,
    small_model,
    large_mode,
    small_mode,
    default_prompts,
    input_dir,
    output_dir,
    prompts,
    completion_mode,
    concurrency_limit,
    use_stop,
    subset_size,
    use_subset,
    cot_preface,
    cot_suffix,
    cost_per_million_small_input,
    cost_per_million_small_output,
    cost_per_million_large_input,
    cost_per_million_large_output,
    sharegpt_convs_passed_in=[],
    read_files_manually=True,
    do_meta_datagen=False,
    meta_datagen_keys=["thought_process_addition_details"],
    meta_datagen_extras=[],
    task_id=None,
    seed=1048596,
    **kwargs,
):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # # # start_time = time.time()
    logger.info(
        "Starting generic transform pipeline. Bringing generic stuff in line with the new format so that we generalize better."
    )

    # NOTE Load the source texts
    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

    # setup
    logger.info("Setting up semaphore and engines")
    # # setup_start = time.time()

    small_token_counter = {
        "input_tokens": 0,
        "input_cost": 0.0,
        "output_tokens": 0,
        "output_cost": 0.0,
        "name": "Small model",
    }
    large_token_counter = {
        "input_tokens": 0,
        "input_cost": 0.0,
        "output_tokens": 0,
        "output_cost": 0.0,
        "name": "Large model",
    }

    run_task_with_limit, engine_wrapper, engine_wrapper_large, _ = (
        setup_semaphore_and_engines(
            concurrency_limit,
            small_model,
            small_api_key,
            small_base_url,
            small_mode,
            large_model,
            large_api_key,
            large_base_url,
            large_mode,
            engine_input_observers=[
                create_input_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_input,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                )
            ],
            engine_output_observers=[
                create_log_observer(output_dir),
                create_output_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                ),
            ],
            large_engine_input_observers=[
                create_input_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_input,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                )
            ],
            large_engine_output_observers=[
                create_log_observer(output_dir),
                create_output_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                ),
            ],
        )
    )
    # # logger.info(f"Setup completed in {time.time() - setup_start:.2f}s")

    set_progress(task_id, progress=0.1, message="Pipeline starting; reading files")
    # # load_start = time.time()
    if read_files_manually:
        logger.info("Loading conversations from config")
        conversations, original_conversations = (
            read_sharegpt_pairs(  # except this should actually be reading sharegpt not text chunks
                input_dir=input_dir,
            )
        )
    else:
        logger.info("Using text chunks passed in")
        conversations, original_conversations = process_sharegpt_pairs(
            sharegpt_convs_passed_in
        )
    # # logger.info(f"Loaded {len(conversations)} conversations in {time.time() - load_start:.2f}s")

    if use_subset:
        before_subset = len(conversations)
        if len(conversations) >= subset_size:
            random.seed(seed)  # guaranteed determinism for this sampling
            conversations = random.sample(conversations, subset_size)
            logger.info(f"Subsetting from {before_subset} to {subset_size} objects")

    class GenericThoughtsAddStep(PipelineStep):

        def process_input_data(
            self, input_data
        ):  # this runs for a decent number of seconds BEFORE all the async requests are sent. Interesting. Could have something to do with some slowness. Why does it wait to do all the code before sending off the api requests?
            conv_history_str = format_conversation_history(
                slice_conversation_history(
                    original_conversations[input_data["source_file"]][
                        input_data["conv_idx"]
                    ],
                    input_data["pair_idx"],
                )
            )

            return input_data, {"conv_history_str": conv_history_str}

    thought_process_addition_pipeline = GenericThoughtsAddStep(
        prompt_path="thought_process_addition",
        sampling_params={
            "max_tokens": 5000,
            # "min_p": 0.4,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n\n\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.7,
        },
        output_processor=process_thought_process_tags,
        result_key="thought_process",
        output_file="revised_generics",
        details_key="thought_process_addition_details",
        validation_function=check_token_length,
    )

    # process input data now returns a dict which we will passi n as kwargs

    logger.info("Hashing input list")
    convs_dict = hash_input_list(conversations)

    total_tokens = count_total_tokens(conversations)

    set_progress(
        task_id,
        progress=0.2,
        message=f"Files read and chunked; {total_tokens} available; proceeding with generation",
    )

    await thought_process_addition_pipeline.execute_pipeline(
        input_dict=convs_dict,
        rtwl=run_task_with_limit,
        engine_wrapper=engine_wrapper_large,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        prompt_folder=prompts,
        default_prompt_folder=default_prompts,
        include_details=do_meta_datagen,
    )

    set_progress(
        task_id, progress=0.9, message="Generations done; saving final dataset"
    )  # even though the distance between here and the very end is going to be very slight, we establish a difference because in case it errors during the final saving, we want the progress to express where it failed

    logger.info("Saving final dataset")

    final_output_dataset = save_dict_to_jsonl(
        convs_dict,
        cot_preface=cot_preface,
        cot_suffix=cot_suffix,
        output_path=output_dir,
    )  # todo modify
    # # logger.info(f"Dataset saved in {time.time() - save_start:.2f}s")

    # # # # total_time = time.time() - start_time
    # # # # # logger.info(f"Total pipeline execution time: {total_time:.2f}s ({datetime.timedelta(seconds=int(total_time))})")

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[convs_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    set_progress(task_id, progress=1.0, message="Pipeline Complete")

    return final_output_dataset


# Does empathic work for multi-turn?
# IDFK


# start command shall have to reload the huey tasks in the event of modifications
