import asyncio
import os
import re

from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    read_and_chunk_text,
)
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from redis_config import set_progress


def extract_cleaned_chunk(llm_output):
    # Remove any <think> tags and their content

    if "think" in llm_output and "</think>" not in llm_output:
        raise Exception("Think tags broken")
    llm_output = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL)
    cleaned_text = re.search(
        r"<cleaned_text>(.*?)</cleaned_text>", llm_output, re.DOTALL
    )
    if cleaned_text:
        return cleaned_text.group(1)
    else:
        raise Exception("LLM extraction tags broken")


clean_pdf_step = PipelineStep(
    prompt_path="clean_pdf",
    sampling_params={
        "max_tokens": 5000,
        "temperature": 0.5,
        # "top_k": -1,
        "top_p": 0.9,
        # "min_p": 0.6,
    },
    output_file="pdf_cleaned_text",
    result_key="pdf_cleaned_text",
    output_processor=extract_cleaned_chunk,
    max_retries=3,
    details_key="clean_pdf_details",
)


async def pdf_clean_and_convert_pipeline(
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
    concurrency_limit: int,
    use_stop: bool,
    prompts: str,
    default_prompts: str,
    cost_per_million_small_input,
    cost_per_million_small_output,
    cost_per_million_large_input,
    cost_per_million_large_output,
    do_meta_datagen: bool = False,
    meta_datagen_keys: list[str] = [],
    meta_datagen_extras: list[str] = [],
    read_files_manually=True,
    texts_passed_in=[],
    chunking_output_dir=None,
    task_id=None,
    seed=1048596,
    do_not_use_llm=False,
    **kwargs,
):
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

    run_task_with_limit, engine_wrapper, engine_wrapper_large, _ = (
        setup_semaphore_and_engines(
            concurrency_limit,
            small_model,
            small_api_key,
            small_base_url,
            small_mode,
            "large_model",
            "large_api_key",
            "large_base_url",
            "large_mode",
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

    set_progress(
        task_id, progress=0.1, message="Pipeline starting; reading and chunking files"
    )

    if read_files_manually:
        chunks = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=use_subset,
            subset_size=subset_size,
            extensions=[".pdf"],
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
            seed=seed,
        )
        if use_subset:
            chunks = chunks[:subset_size]
    else:
        if texts_passed_in is None:
            print("No texts passed in")
            return
            # chunks = read_and_chunk_text( # is the pdf OCR deterministic?
            #     input_dir=input_dir,
            #     chunk_size=chunk_size,
            #     use_subset=use_subset,
            #     subset_size=subset_size,
            #     extensions=[".pdf"]
            # )
        else:
            chunks = chunk_text_list(texts_passed_in)
            chunks = chunks[
                :subset_size
            ]  # not using the typical random subsetting because texts passed in subsetting is literally only used in test environments.
        print("Chunk length!!!")
        print(len(chunks))
    if len(chunks) == 0:
        print("No chunks found")
        return []

    # Add index to each chunk for proper recombination
    for idx, chunk in enumerate(chunks):
        chunk["index"] = idx

    set_progress(
        task_id,
        progress=0.2,
        message="Files read and chunked; proceeding with PDF cleaning",
    )

    working_dict = hash_input_list(input_list=chunks)

    if not do_not_use_llm:
        await clean_pdf_step.execute_pipeline(
            input_dict=working_dict,
            engine_wrapper=engine_wrapper,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            do_meta_datagen=do_meta_datagen,
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            include_details=do_meta_datagen,
        )
    else:
        # Skip LLM step and just copy the input text to the cleaned text field
        for key, value in working_dict.items():
            if "text" in value:
                value["pdf_cleaned_text"] = value["text"]
            else:
                print(f"Warning: 'text' key not found for item {key}. Skipping.")
                value["pdf_cleaned_text"] = (
                    ""  # Assign empty string or handle as appropriate
                )

    set_progress(
        task_id,
        progress=0.9,
        message="Generations done; recombining chunks and saving final dataset",
    )  # even though the distance between here and the very end is going to be very slight, we establish a difference because in case it errors during the final saving, we want the progress to express where it failed

    # Group and recombine chunks by source
    grouped_chunks = {}
    for key, value in working_dict.items():
        if "pdf_cleaned_text" in value:
            metadata = os.path.basename(value["metadata"])
            index = value["index"]
            cleaned_text = value["pdf_cleaned_text"]

            if metadata not in grouped_chunks:
                grouped_chunks[metadata] = []
            grouped_chunks[metadata].append((index, cleaned_text))
        else:
            print("Should not have happened -- cleaned text not present")

    output_text_list = []
    for metadata, chunks in grouped_chunks.items():
        # Sort chunks by index and join text
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        combined_text = "".join([text for _, text in sorted_chunks])

        # Write combined text to file
        output_path = os.path.join(output_dir, f"{metadata}.txt")
        # print("OUTPUT PATH")
        # print(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        output_text_list.append({"metadata": metadata, "text": combined_text})

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[working_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete")

    return output_text_list
