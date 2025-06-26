import re
import json
import os
from pathlib import Path
import random
import traceback
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.generation_functions.random_variation_step_class import (
    RandomVariationStep,
)
from augmentoolkit.generation_functions.run_pipeline_step import (
    create_random_variation_step_function,
)
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from augmentoolkit.utils.extract_first_words import remove_think_tags
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from augmentoolkit.utils.parse_bool import parse_bool
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from generation.core_components.filter_chunks import create_filter_chunks_step
from generation.core_components.filter_chunks import (
    filter_out_failed_items,
    filter_out_failed_items_dict,
)
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    setup_semaphore_and_engines,
    make_relative_to_self,
)
from generation.core_components.chunking import (
    chunk_text_list,
    chunking_algorithm_str,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
    read_jsonl_completions,
    read_text,
    subset_text_list,
)
import augmentoolkit.utils.sentence_chunking_algorithm as sta
import nlpaug.augmenter.char as nac


# Idea: do multiple passes with the pretraining stuff generator, with different chunk sizes. So that we group different facts. Set up multiple configs, line em up and knock em down

import nltk
from tqdm import asyncio as tqdmasyncio, tqdm


import asyncio

from redis_config import set_progress


def validate_atomic_fact_extraction(output, input_data):
    return {"result": True, "message": "Success"}


def extract_atomic_facts(output):
    # Remove content between <think> and </think> if present
    if "<think>" in output and "</think>" in output:
        output = output.split("<think>")[0] + output.split("</think>")[1]

    # First try to find atomic facts tags
    if "<atomic_facts>" in output and "</atomic_facts>" in output:
        return output.split("<atomic_facts>")[1].split("</atomic_facts>")[0]
    else:
        # Fallback: Find last numbered list in the text
        lines = output.split("\n")
        collected = []
        in_list = False

        # Iterate in reverse to find the last contiguous numbered list
        for line in reversed(lines):
            stripped_line = line.strip()
            # Match lines starting with "1. ", "2. ", etc.
            if re.match(r"^\d+\.\s", stripped_line):
                collected.append(stripped_line)
                in_list = True
            elif in_list:  # Stop when we hit non-list line after list starts
                break

        # Reverse to restore original order and join lines
        collected.reverse()
        return "\n".join(collected) if collected else ""


make_atomic_facts_step = PipelineStep(
    prompt_path="atomic_fact_extraction",
    sampling_params={
        "max_tokens": 8000,  # high max tokens ecause common problem was the model running out of these while generating facts. Especially cot models.
        "temperature": 0.5,
        "top_p": 0.8,
    },
    output_file="synthetic_pretrain",
    intermediate_output_path="intermediate_generations",
    save_path="saved_readable_generations",
    validation_function=validate_atomic_fact_extraction,
    max_retries=3,
    result_key="atomic_facts",
    output_processor=extract_atomic_facts,
    details_key="atomic_facts_details",
)


def extract_inferred_facts(output):
    cleaned_output = remove_think_tags(output)
    return cleaned_output.split("<inferred_facts>")[1].split("</inferred_facts>")[0]


make_inferred_facts_step = PipelineStep(
    prompt_path="inferred_facts_writing",
    sampling_params={
        "max_tokens": 5000,  # high max tokens ecause common problem was the model running out of these while generating facts. Especially cot models.
        "temperature": 0.5,
        "top_p": 0.8,
    },
    output_file="synthetic_pretrain",
    intermediate_output_path="intermediate_generations",
    save_path="saved_readable_generations",
    validation_function=validate_atomic_fact_extraction,
    max_retries=3,
    result_key="inferred_facts",
    output_processor=extract_atomic_facts,
    details_key="inferred_facts_details",
)


def simulate_typing_errors(text: str) -> str:
    # Get the indices of all spaces in the string.
    space_indices = [i for i, char in enumerate(text) if char == " "]

    # Calculate the number of spaces to remove (at least 1 if any spaces exist).
    num_spaces = len(space_indices)
    if num_spaces == 0:
        return text  # Nothing to remove if there are no spaces.

    num_to_remove = max(1, int(round(0.05 * num_spaces)))

    # Randomly choose indices (from the list of space indices) to remove.
    indices_to_remove = set(random.sample(space_indices, num_to_remove))

    # Build a new string omitting the chosen spaces.
    modified_chars = [
        char
        for i, char in enumerate(text)
        if not (char == " " and i in indices_to_remove)
    ]
    modified_text = "".join(modified_chars)

    # Determine how many spaces to add back (half of the removed ones).
    num_to_add = int(round(num_to_remove / 2))

    # Insert a space at a random index for each space to add.
    for _ in range(num_to_add):
        # There are len(modified_text)+1 possible positions to insert.
        pos = random.randint(0, len(modified_text))
        modified_text = modified_text[:pos] + " " + modified_text[pos:]

    return modified_text


keyboard_aug = nac.KeyboardAug()


def allcaps(string):
    return string.upper()


def lowercase(string):
    return string.lower()


def keyboard_augmentation(string):
    augmented = keyboard_aug.augment(string)
    if augmented:
        augmented = augmented[0]
    else:
        augmented = string

    return simulate_typing_errors(augmented)


def serialkillercase(string):
    # cApS lIkE tHiS
    result = ""
    for i, char in enumerate(string):
        if i % 2 == 0:
            result += char.lower()
        else:
            result += char.upper()
    return result


def titlecase(string):
    return string.title()


def sentencecase(string):
    # Capitalizes first letter of each sentence
    sentences = string.split(". ")
    return ". ".join(s.capitalize() for s in sentences)


def snakecase(string):
    # Converts spaces to underscores and lowercase
    return string.lower().replace(" ", "_")


def kebabcase(string):
    # Converts spaces to hyphens and lowercase
    return string.lower().replace(" ", "-")


def camelcase(string):
    # Removes spaces and capitalizes first letter of each word except first
    words = string.split()
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def pascalcase(string):
    # Removes spaces and capitalizes first letter of each word
    return "".join(word.capitalize() for word in string.split())


def randomcase(string):
    # Randomly capitalizes letters
    import random

    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in string)


def invertcase(string):
    # Inverts the case of each character
    return "".join(c.lower() if c.isupper() else c.upper() for c in string)


code_variation_functions = []


def apply_capitalization_variation(working_dict, code_variation_functions=[]):
    """
    Applies capitalization variations to each item's text exactly once.
    For each item:
    - Applies each function once to the original chunk
    - Applies each function once to each original variation
    """
    # Map string identifiers to actual functions
    function_map = {
        "allcaps": allcaps,
        "lowercase": lowercase,
        "serialkillercase": serialkillercase,
        "titlecase": titlecase,
        "sentencecase": sentencecase,
        "snakecase": snakecase,
        "kebabcase": kebabcase,
        "camelcase": camelcase,
        "pascalcase": pascalcase,
        "randomcase": randomcase,
        "invertcase": invertcase,
        "keyboard_augmentation": keyboard_augmentation,
    }

    # Convert string identifiers to function objects
    resolved_functions = []
    for func in code_variation_functions:
        if isinstance(func, str):
            resolved = function_map.get(func)
            if resolved:
                resolved_functions.append(resolved)
        elif callable(func):
            resolved_functions.append(func)

    for key, value in tqdm(
        working_dict.items(), desc="Applying capitalization variations"
    ):
        new_variations = []

        # Process the chunk
        for function in resolved_functions:
            new_variations.append(function(value["text"]))

        # Process original variations
        if "variations" in value:
            original_variations = value["variations"].copy()
            for variation in original_variations:
                for function in resolved_functions:
                    new_variations.append(function(variation))

        # Add all new variations to the item
        if "variations" not in value:
            value["variations"] = []
        value["variations"].extend(new_variations)

    # return output_list


def save_dataset(output_dict, output_dir, include_context_in_dataset, dataset_context):
    """
    Saves dataset to a JSONL file where each line is a JSON object with a single "text" key.
    Each input chunk and its variations are saved as separate entries.

    Args:
        output_list: List of dicts containing "chunk", "source", and "variations" keys
        output_dir: Directory to save the output file
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "final_output.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        # Process each item in the output list
        for key, item in output_dict.items():
            # Write original chunk
            if not include_context_in_dataset:
                f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
                f.write(
                    json.dumps({"text": item["atomic_facts"]}, ensure_ascii=False)
                    + "\n"
                )
            else:
                f.write(
                    json.dumps(
                        {
                            "text": f"[[[OVERALL_CONTEXT_IS -> {dataset_context}]]]\nSpecific source: {os.path.basename(item['metadata'])}\n{item['text']}"
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write(
                    json.dumps(
                        {
                            "text": f"[[[OVERALL_CONTEXT_IS -> {dataset_context}]]]\nSpecific source: {os.path.basename(item['metadata'])}\n{item['atomic_facts']}"
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Write variations if they exist
            if "variations" in item and isinstance(item["variations"], list):
                for variation in item["variations"]:
                    if include_context_in_dataset:
                        f.write(
                            json.dumps(
                                {
                                    "text": f"[[[OVERALL_CONTEXT_IS -> {dataset_context}]]]\nSpecific source: {os.path.basename(item['metadata'])}\n{variation}"
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    else:
                        f.write(
                            json.dumps({"text": variation}, ensure_ascii=False) + "\n"
                        )

    print(f"Dataset saved to {output_file}")


def extract_rewrite(output):
    return output.split("<rewrite>")[1].split("</rewrite>")[0]


async def representation_variation_pipeline(
    use_subset: bool,
    subset_size: int,
    chunk_size: int,
    input_dir: str,
    output_dir: str,
    completion_mode: bool,
    small_model: str,
    small_api_key: str,
    small_base_url: str,
    small_mode: str,
    large_model: str,
    large_api_key: str,
    large_base_url: str,
    large_mode: str,
    concurrency_limit: int,
    use_stop: bool,
    prompts: str,
    default_prompts: str,
    variation_generator_count: int,
    include_context_in_dataset: bool,
    dataset_context: str,
    code_variation_functions,
    make_inferred_facts: bool,
    cost_per_million_small_input=0.0,
    cost_per_million_small_output=0.0,
    cost_per_million_large_input=0.0,
    cost_per_million_large_output=0.0,
    read_files_manually=True,
    texts_passed_in=[],
    do_meta_datagen=False,
    meta_datagen_keys=[],
    meta_datagen_extras=[],
    chunking_output_dir=None,
    additional_dataset_context: str = "",
    task_id=None,
    seed=1048596,
    **kwargs,
):

    filter_chunks_step = create_filter_chunks_step(output_file="synthetic_pretrain")
    generate_variations_step = RandomVariationStep(
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.5,
            "top_p": 0.8,
        },
        prompt_path="variations",  # Subdirectory containing variation prompts
        output_file="synthetic_pretrain",
        intermediate_output_path="intermediate_generations",
        save_path="saved_readable_generations",
        max_retries=3,  # really all we do here is that the prompt folder is the thing we pick a random yaml from.
        result_key="variations",
        variation_generator_count=variation_generator_count,
        output_processor=extract_rewrite,
        details_key="variations_details",
    )

    # Check if kwargs is not empty and print all keys and values if present
    if kwargs:
        print("Additional arguments provided:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

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

    # Set up engines and semaphore
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
                create_log_observer(output_dir, do_meta_datagen),
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
                create_log_observer(output_dir, do_meta_datagen),
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

    set_progress(
        task_id=task_id,
        progress=0.0,
        message="Pipeline starting; reading and chunking files",
    )
    if read_files_manually:

        if chunking_output_dir:
            working_list = read_and_chunk_text(
                input_dir,
                chunk_size=chunk_size,
                use_subset=use_subset,
                subset_size=subset_size,
                keep_folder_structure=True,
                output_dir=chunking_output_dir,
                seed=seed,
            )
        else:
            working_list = read_and_chunk_text(
                input_dir,
                chunk_size=chunk_size,
                use_subset=use_subset,
                subset_size=subset_size,
                keep_folder_structure=True,
                output_dir=output_dir,
                seed=seed,
            )

        # loaded_list_1 = read_jsonl_completions(input_dir=input_dir)
        # loaded_list_2 = read_text(input_dir=input_dir)
        # print("LENGTH OF LOADED LISTS")
        # print(len(loaded_list_1))
        # print(len(loaded_list_2))
        # full_list = loaded_list_1 + loaded_list_2

        # # Need to use sentence chunking algorithm here.
        # working_list = []
        # for item in tqdm(full_list, desc="Chunking texts"):
        #     working_list.extend(chunking_algorithm_str(item["text"], item["metadata"], max_token_length=chunk_size, keep_folder_structure=True, input_dir=input_dir))
    else:
        working_list = texts_passed_in

        working_list = chunk_text_list(
            text_list=working_list,
            chunk_size=chunk_size,
            keep_folder_structure=True,
            input_dir=input_dir,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
        )

        if use_subset:
            working_list = subset_text_list(
                working_list, subset_size=subset_size, seed=seed
            )

    print("LENGTH OF WORKING LIST")
    print(len(working_list))
    # simple fix to sampling easy:
    # convert to hashed input list AFTER random sampling and other such modifications

    # NOTE on the majority vote step when we get there -- I will want to make it so that only votes which agreed with the majority were saved for training

    total_tokens = count_total_tokens(working_list)
    print(f"Total input tokens: {total_tokens}")

    working_dict = hash_input_list(input_list=working_list, key_to_hash_with="text")

    print("Here is one of the conversation pairs:")
    print(working_list[0])

    set_progress(
        task_id,
        progress=0.1,
        message="Files read and chunked! Proceeding with LLM chunk filtering",
    )

    await filter_chunks_step.execute_pipeline(
        input_dict=working_dict,
        engine_wrapper=(
            engine_wrapper if not make_inferred_facts else engine_wrapper_large
        ),
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )  # use large engine wrapper if inferred facts, since small engine wrapper is a reasoner model.

    filter_out_failed_items_dict(working_dict)

    set_progress(
        task_id,
        progress=0.2,
        message=f"Chunks filtered! {len(working_dict.items())} items left; proceeding with Atomic Fact generation.",
    )

    await make_atomic_facts_step.execute_pipeline(
        input_dict=working_dict,
        engine_wrapper=engine_wrapper,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
        additional_dataset_context=additional_dataset_context,
    )

    set_progress(
        task_id,
        progress=0.4,
        message=f"Atomic Facts generated! {len(working_dict.items())} items left; proceeding with Inferred Facts generation (or skipping it and moving onto variations step)",
    )

    if make_inferred_facts:
        await make_inferred_facts_step.execute_pipeline(
            input_dict=working_dict,
            engine_wrapper=engine_wrapper,  # here, engine wrapper is the reasoner, engine wrapper large is a large non-reasoner
            rtwl=run_task_with_limit,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
            additional_dataset_context=additional_dataset_context,
        )

    set_progress(
        task_id,
        progress=0.4,
        message=f"Inferred Facts generated! {len(working_dict.items())} items left; proceeding with Representation Variation generation",
    )

    # NOTE generate the variations
    await generate_variations_step.execute_pipeline(
        working_dict,
        engine_wrapper_large,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        include_details=do_meta_datagen,
        additional_dataset_context=additional_dataset_context,
    )

    set_progress(
        task_id,
        progress=0.9,
        message=f"Varied Representations generated! {len(working_dict.items())} items left; proceeding with programmatic text processing and dataset saving",
    )

    # TODO caps variation and other programmatic stuff. Use code to go over the list and append more versions.

    apply_capitalization_variation(working_dict, code_variation_functions)

    # # Save output to jsonl file
    # output_dir = os.path.join(config["PATH"]["OUTPUT"], "processed_outputs")
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"processed_output_{int(time.time())}.jsonl")

    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for item in output_list:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # print(f"Saved {len(output_list)} items to {output_file}")

    # Save the final dataset format
    save_dataset(
        output_dict=working_dict,
        output_dir=output_dir,
        include_context_in_dataset=include_context_in_dataset,
        dataset_context=dataset_context,
    )

    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    print("Pipeline complete!")

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[working_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete!")


# NOTE we could have accomplished the whole "dynamic system prompt" thing by just having the system prompt be a template {} and then interpolating it


# NOTE this is now an exemplar of the new dict-based paradigm
# NOTE we will want to improve this such that atomic fact lists are added as data too
