import random
import re
import traceback
from augmentoolkit.generation_functions.cleanup import cleanup_dir
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from augmentoolkit.utils.parse_bool import parse_bool
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from generation.core_pipelines.train_a_model_to_do_rag.rag_data_pipeline_helpers import (
    save_combined_conversations,
    stringify_rag_chunks,
)
import augmentoolkit.utils.sentence_chunking_algorithm as sta
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
)
from generation.core_components.filter_chunks import (
    create_filter_chunks_step,
    filter_out_failed_items_dict,
)

# Idea: do multiple passes with the pretraining stuff generator, with different chunk sizes. So that we group different facts. Set up multiple configs, line em up and knock em down

import nltk
from tqdm import asyncio as tqdmasyncio, tqdm


import asyncio
import glob
import logging
import os
import sys
import time
import yaml
import json

from redis_config import set_progress

# want to borrow loaders from ATK will be useful. But later?


def extract_qa_tuples(text):
    pattern = r"\*\*QUESTION:\*\*\s*((?:.|\n)*?)\s*\*\*ANSWER:\*\*\s*((?:.|\n)*?)(?=\s*\*\*QUESTION:\*\*|\Z)"
    matches = re.findall(
        pattern, text + "\n\n**QUESTION:**", re.DOTALL
    )  # The addition is a hack to get around the tricky lookahead problem
    res = [
        {"question": question.strip(), "answer": answer.strip()}
        for question, answer in matches
    ]
    if any(["Ground Truth" in item["answer"] for item in res]):
        raise Exception(
            "Ground Truth found in answer! This is a problem with the response! Raising and retrying!"
        )
    return res


# NOTE we could have accomplished the whole "dynamic system prompt" thing by just having the system prompt be a template {} and then interpolating it
# darn it!!!
# worth remembering this pattern for later though


async def rag_data_pipeline(
    input_dir: str,
    output_dir: str,
    use_subset: bool,
    subset_size: int,
    chunk_size: int,
    rag_failure_percentage: float,
    rag_max_chunks: int,
    final_assistant_prompts: list,
    system_format: str,
    user_format: str,
    assistant_format: str,
    bos: str,
    num_items_per_group: int,
    concurrency_limit: int,
    completion_mode: bool,
    use_stop: bool,
    prompts: str,
    default_prompts: str,
    skip_filter_chunks: bool,
    small_model: str,
    small_api_key: str,
    small_base_url: str,
    small_mode: str,
    large_model: str,
    large_api_key: str,
    large_base_url: str,
    large_mode: str,
    cost_per_million_small_input: float,
    cost_per_million_small_output: float,
    cost_per_million_large_input: float,
    cost_per_million_large_output: float,
    read_files_manually: bool = True,
    do_meta_datagen: bool = False,
    meta_datagen_keys: list[str] = [],
    meta_datagen_extras: list[str] = [],
    text_chunks_passed_in: list = [],
    chunking_output_dir=None,
    task_id=None,
    seed=1048596,
    **kwargs,  # All nodes MUST have **kwargs
):
    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)
    filter_chunks_step = create_filter_chunks_step(output_file="rag_convs")

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

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    # NOTE Load the source texts
    print("Welcome to your RAG reasoning multiturn data generation pipeline!")

    set_progress(
        task_id=task_id,
        progress=0.0,
        message="Pipeline starting; reading and chunking files",
    )

    if read_files_manually:
        print("Using config...")
        if chunking_output_dir:
            sentence_chunks = read_and_chunk_text(
                input_dir=input_dir,
                chunk_size=chunk_size,
                use_subset=use_subset,
                subset_size=subset_size,
                output_dir=chunking_output_dir,
                seed=seed,
            )
        else:
            sentence_chunks = read_and_chunk_text(
                input_dir=input_dir,
                chunk_size=chunk_size,
                use_subset=use_subset,
                subset_size=subset_size,
                output_dir=output_dir,
                seed=seed,
            )

    else:
        print("Using text chunks passed in...")
        sentence_chunks = chunk_text_list(text_chunks_passed_in, chunk_size)

    set_progress(
        task_id,
        progress=0.1,
        message="Files read and chunked! Proceeding with LLM chunk filtering",
    )

    total_tokens = count_total_tokens(sentence_chunks)

    sentence_dict = hash_input_list(input_list=sentence_chunks, key_to_hash_with="text")

    if not skip_filter_chunks:
        await filter_chunks_step.execute_pipeline(
            input_dict=sentence_dict,
            engine_wrapper=engine_wrapper,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )

        filter_out_failed_items_dict(sentence_dict, key_to_check="judgement")

    set_progress(
        task_id,
        progress=0.2,
        message=f"Chunks filtered! {len(sentence_dict.items())} items left; proceeding with RAG context simulation",
    )

    # Define pipeline steps
    class RagFailedStep(PipelineStep):
        def __init__(self):
            super().__init__(
                prompt_path="rag_failed_conversation",
                sampling_params={
                    "max_tokens": 3000,
                    "temperature": 0.5,
                    # "top_k": -1,
                    "top_p": 0.8,
                    # "min_p": 0.6,
                },
                output_file="rag_failed_convs",
                result_key="rag_conversation",
                output_processor=extract_qa_tuples,
                details_key="rag_failed_details",
            )

        def process_input_data(self, input_data):
            # print(input_data)
            # print(f"[RagFailedStep] Processing input data with key: {input_data.get('_key', 'unknown')}")
            # print(f"[RagFailedStep] Number of RAG chunks: {len(input_data.get('rag_chunks', []))}")
            input_data["stringified_retrieved_chunks"] = stringify_rag_chunks(
                input_data["rag_chunks"]
            )
            # print(f"[RagFailedStep] Stringified chunks length: {len(input_data['stringified_retrieved_chunks'])}")
            return input_data, {}

    class RagSuccessStep(PipelineStep):
        def __init__(self):
            super().__init__(
                prompt_path="rag_success_conversation",
                sampling_params={
                    "max_tokens": 3000,
                    "temperature": 0.5,
                    "top_p": 0.8,
                },
                output_file="rag_success_convs",
                result_key="rag_conversation",
                output_processor=extract_qa_tuples,
                details_key="rag_success_details",
            )

        def process_input_data(self, input_data):
            # print(f"[RagSuccessStep] Processing input data with key: {input_data.get('_key', 'unknown')}")
            # print(f"[RagSuccessStep] Number of RAG chunks: {len(input_data.get('rag_chunks', []))}")
            input_data["stringified_retrieved_chunks"] = stringify_rag_chunks(
                input_data["rag_chunks"]
            )
            # print(f"[RagSuccessStep] Stringified chunks length: {len(input_data['stringified_retrieved_chunks'])}")
            return input_data, {}

    rag_failed_step = RagFailedStep()
    rag_success_step = RagSuccessStep()

    # simple rule: if in-memory modifications are made to the list, the steps AFTER That point should have a different output file name, otherwise it will load the state of the list from before the modifications and it will be overwritten.

    rag_output_dir = chunking_output_dir if chunking_output_dir else output_dir

    this_random = (
        random.Random()
    )  # we don't want funky global random state things messing with the order across runs
    this_random.seed(11037)
    # Process RAG preparation
    rag_prepared_path = os.path.join(rag_output_dir, "rag_prepared_data.jsonl")
    if not os.path.exists(rag_prepared_path):
        print("Preparing RAG context simulations")
        all_paragraphs = {
            k: {"text": v["text"], "metadata": v["metadata"]}
            for k, v in sentence_dict.items()
        }
        # print(f"[RAG Prep] Total paragraphs available: {len(all_paragraphs)}")

        async def prepare_rag_item(key, item):
            # print(f"[RAG Prep] Processing item with key: {key}")
            item["rag_failed"] = this_random.choices(
                [True, False],
                weights=[rag_failure_percentage, 1 - rag_failure_percentage],
            )[0]
            # print(f"[RAG Prep] Item marked as RAG failed: {item['rag_failed']}")

            num_chunks = this_random.randint(1, rag_max_chunks)
            # print(f"[RAG Prep] Selected {num_chunks} chunks for this item")

            if item["rag_failed"]:  # select random chunks
                available_keys = [k for k in all_paragraphs if k != key]
                # print(f"[RAG Prep] Failed RAG - Available keys for random selection: {len(available_keys)}")
                selected_keys = this_random.sample(available_keys, num_chunks)
                # print(f"[RAG Prep] Failed RAG - Selected keys: {selected_keys}")
                item["rag_chunks"] = [all_paragraphs[k] for k in selected_keys]
            else:  # select original chunk + random others
                # print(f"[RAG Prep] Successful RAG - Including original chunk")
                item["rag_chunks"] = [all_paragraphs[key]]
                if num_chunks > 1:
                    additional_keys = this_random.sample(
                        [k for k in all_paragraphs if k != key], num_chunks - 1
                    )
                    # print(f"[RAG Prep] Successful RAG - Adding {len(additional_keys)} additional chunks: {additional_keys}")
                    item["rag_chunks"].extend(
                        [all_paragraphs[k] for k in additional_keys]
                    )
                this_random.shuffle(item["rag_chunks"])
                # print(f"[RAG Prep] Successful RAG - Chunks shuffled")

            # print(f"[RAG Prep] Final chunk count for item {key}: {len(item['rag_chunks'])}")
            return (key, item)

        # print(f"[RAG Prep] Creating preparation tasks for {len(sentence_dict)} items")
        prep_tasks = [prepare_rag_item(k, v) for k, v in sentence_dict.items()]
        # print(f"[RAG Prep] Awaiting completion of {len(prep_tasks)} preparation tasks")
        prepared_items = await asyncio.gather(*prep_tasks)
        # print(f"[RAG Prep] All preparation tasks completed")

        # Convert list of (key, item) tuples back to dict
        rag_prepared_data = {k: v for k, v in prepared_items}
        # print(f"[RAG Prep] Prepared data dictionary created with {len(rag_prepared_data)} items")

        # Save with keys preserved
        # print(f"[RAG Prep] Saving prepared data to {rag_prepared_path}")
        with open(rag_prepared_path, "w") as f:
            for key, item in rag_prepared_data.items():
                item["_key"] = key  # Store key in the item
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # print(f"[RAG Prep] Prepared data saved successfully")

    # Load prepared data and reconstruct dictionary
    else:
        # print(f"[RAG Prep] Loading existing prepared data from {rag_prepared_path}")
        rag_prepared_data = {}
        with open(rag_prepared_path, "r") as f:
            for line in f:
                item = json.loads(line)
                key = item.pop("_key")  # Retrieve stored key
                rag_prepared_data[key] = item
        # print(f"[RAG Prep] Loaded {len(rag_prepared_data)} prepared items")

    # Split into failed/success groups
    rag_failed_list = {k: v for k, v in rag_prepared_data.items() if v["rag_failed"]}
    rag_success_list = {
        k: v for k, v in rag_prepared_data.items() if not v["rag_failed"]
    }
    # print(f"[RAG Prep] Split data into {len(rag_failed_list)} failed RAG items and {len(rag_success_list)} successful RAG items")
    # Print the first item from each list for debugging/inspection
    set_progress(
        task_id,
        progress=0.3,
        message=f'RAG context simulation finished! {len(rag_failed_list.items())} "RAG failure" simulations and {len(rag_success_list.items())} "RAG success" simulations. Proceeding with rag question generation',
    )

    # Generate conversations
    await rag_failed_step.execute_pipeline(
        input_dict=rag_failed_list,
        engine_wrapper=engine_wrapper_large,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )

    set_progress(
        task_id,
        progress=0.6,
        message=f"RAG failure conversations generated! Proceeding with RAG success generation",
    )

    await rag_success_step.execute_pipeline(
        input_dict=rag_success_list,
        engine_wrapper=engine_wrapper_large,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )

    # Combine and save results
    combined_conversations = []
    # for key, item in rag_prepared_data.items():
    #     if "rag_conversation" in item:
    #         combined_conversations.append(item)

    set_progress(
        task_id,
        progress=0.9,
        message=f"All RAG generated! {len(combined_conversations)} conversations in total. Proceeding with final saving...",
    )  # coding style (be sure to mention): progress updates should start enthusiastically. Because work is more fun when you're being encouraged and there's positivity and Augmentoolkit is meant to be engaging to use.

    # Also check for conversations in both success and failed lists
    for key, item in rag_failed_list.items():
        if "rag_conversation" in item and item not in combined_conversations:
            combined_conversations.append(item)

    for key, item in rag_success_list.items():
        if "rag_conversation" in item and item not in combined_conversations:
            combined_conversations.append(item)

    save_combined_conversations(
        combined_conversations,
        output_dir,
        num_items_per_group,
        final_assistant_prompts,
        system_format,
        user_format,
        assistant_format,
        bos,
    )

    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    for idx, prompt in enumerate(meta_datagen_extras):
        meta_datagen_extras[idx] = make_relative_to_self(prompt)

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[sentence_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete!")
