# the whole composition

import json
import os
import random
import shutil
import sys
import subprocess
import time
import re
import platform
import glob

from generation.utilities.llm_server.llm_server import llm_server
from generation.utilities.rag_server.rag_server import rag_server
from huggingface_hub import HfApi, list_repo_files
from generation.core_components.chunking import read_jsonl_completions, read_text, write_text
from generation.core_components.data_prep_operations import (
    completionify_sharegpt,
    count_item_tokens,
    count_tokens_glob,
    create_subset,
    route_template_to_preset,
    save_hf_dataset,
    write_jsonl,
)
from generation.core_components.sharegpt_operations import (
    combine_single_and_multi_turn,
    combine_single_turn_convs,
)
from generation.core_components.simple_chat_loop import simple_chat_loop
from generation.core_pipelines.clean_and_convert_pdf.clean_and_convert_pdf import (
    pdf_clean_and_convert_pipeline,
)
from generation.core_pipelines.correction_pipeline.corrections import (
    correction_pipeline,
)
from generation.core_pipelines.factual_generation_individual.factual_generation import (
    generate_factual_qa_dataset,
)
from generation.core_pipelines.generic_data_rephrase.transform_generic_data import (
    transform_generic_data_pipeline,
)
from generation.core_pipelines.representation_variation.repvar import (
    representation_variation_pipeline,
)
from generation.core_pipelines.train_a_model_to_do_rag.rag_data_pipeline import (
    rag_data_pipeline,
)
from generation.core_components.write_config_files import (
    write_training_config,
    create_completion_dataset,
    create_input_output_dataset,
)
from generation.core_pipelines.recall_multiple_sources.multi_source_recall import (
    generate_multi_source_dataset,
)
from generation.core_components.data_prep_operations import (
    completionify_sharegpt,
    count_tokens_glob,
    create_subset,
)
from generation.core_components.add_sysprompt_to_text import (
    add_sysprompt_to_directory_files,
)
from redis_config import set_progress


# NOTE we will want the capability to add system prompts to each of the generic things we use. Specify the system prompt context to add to the starts of messages.


async def factual_datagen_full(  # there will be quite a few args here
    input_dirs: list[dict[str, str]],
    output_dir: str,
    models_dir: str,
    use_subset: bool,
    subset_size: int,
    shared_instruction: str,
    completion_mode: bool,
    concurrency_limit: int,
    what_percent_of_sft_is_pretrain: float,
    num_tokens_pretraining_in_sft: int,
    pdf_cleaning_chunk_size: int,
    pdf_cleaning_small_model: str,
    pdf_cleaning_large_model: str,
    pdf_cleaning_small_mode: str,
    pdf_cleaning_large_mode: str,
    pdf_cleaning_small_base_url: str,
    pdf_cleaning_large_base_url: str,
    pdf_cleaning_small_api_key: str,
    pdf_cleaning_large_api_key: str,
    pdf_cleaning_cost_small_input,
    pdf_cleaning_cost_small_output,
    pdf_cleaning_cost_large_input,
    pdf_cleaning_cost_large_output,
    pdf_cleaning_use_stop: bool,
    pdf_cleaning_prompts: str,
    pdf_cleaning_default_prompts: str,
    representation_variation_chunk_size: int,
    representation_variation_small_model: str,
    representation_variation_large_model: str,
    representation_variation_small_mode: str,
    representation_variation_large_mode: str,
    representation_variation_small_base_url: str,
    representation_variation_large_base_url: str,
    representation_variation_small_api_key: str,
    representation_variation_large_api_key: str,
    representation_variation_use_stop: bool,
    representation_variation_cost_small_input: float,
    representation_variation_cost_small_output: float,
    representation_variation_cost_large_input: float,
    representation_variation_cost_large_output: float,
    representation_variation_prompts: str,
    representation_variation_default_prompts: str,
    representation_variation_prompts_inferred: str,
    representation_variation_default_prompts_inferred: str,
    # include_context_in_dataset: bool,
    dataset_context: str,
    code_variation_functions: list[str],
    number_of_factual_sft_generations_to_do: int,
    factual_sft: dict[str, dict[str, str]],
    final_assistant_prompts_no_rag: list[str],
    items_per_conversation: int,
    factual_chunk_size: int,
    factual_completion_mode: bool,
    factual_use_stop: bool,
    factual_small_model: str,
    factual_small_api_key: str,
    factual_small_base_url: str,
    factual_small_mode: str,
    factual_large_model: str,
    factual_large_api_key: str,
    factual_large_base_url: str,
    factual_large_mode: str,
    factual_cost_per_million_small_input: float,
    factual_cost_per_million_small_output: float,
    factual_cost_per_million_large_input: float,
    factual_cost_per_million_large_output: float,
    rag_failure_percentage: float,
    rag_max_chunks: int,
    user_format: str,
    system_format: str,
    assistant_format: str,
    bos: str,
    final_assistant_prompts: list[str],
    num_items_per_group: int,
    rag_small_model: str,
    rag_small_api_key: str,
    rag_small_base_url: str,
    rag_small_mode: str,
    rag_large_model: str,
    rag_large_api_key: str,
    rag_large_base_url: str,
    rag_large_mode: str,
    combine_sharegpt_target_pairs: int,
    rag_cost_per_million_small_input: float,
    rag_cost_per_million_small_output: float,
    rag_cost_per_million_large_input: float,
    rag_cost_per_million_large_output: float,
    rag_use_stop: bool,
    rag_prompts: str,
    rag_default_prompts: str,
    template: str,
    template_kwargs: dict[str, str],
    huggingface_cache_dir: str,
    max_samples_per_dataset: int,
    generic_dataset_paths: list[str],
    generic_dataset_percentages: list[int],
    correction_chunk_size: int,
    correction_small_model: str,
    correction_small_api_key: str,
    correction_small_base_url: str,
    correction_small_mode: str,
    correction_large_model: str,
    correction_large_api_key: str,
    correction_large_base_url: str,
    correction_large_mode: str,
    correction_cost_per_million_small_input: float,
    correction_cost_per_million_small_output: float,
    correction_cost_per_million_large_input: float,
    correction_cost_per_million_large_output: float,
    correction_prompt_template: str,
    correction_use_stop: bool,
    correction_completion_mode: bool,
    correction_prompts: str,
    correction_default_prompts: str,
    minimum_generic_sft: int,
    pretrain_hub_model_id: str,
    pretrain_hub_strategy: str,
    finetune_hub_model_id: str,
    finetune_hub_strategy: str,
    context_length: int,
    remove_system_prompt_ratio: float,
    remove_thought_process_ratio: float,
    remove_thought_process_prompt: str,
    final_answer_str: str,
    generic_thought_process_on_domain_data: bool,  # this one OUGHT to be done inside the factual gen pipeline. Also, Evan, remember the lesson of revelex: do things elegantly, as they ought to be done, and you will have an easier time. Actually no this has to be done in complete factual dataset.
    cite_sources_at_end: bool,  # This one has to be done inside the factual gen pipeline
    transform_generic_data_use_stop: bool,
    transform_generic_data_large_model: str,
    transform_generic_data_large_api_key: str,
    transform_generic_data_large_base_url: str,
    transform_generic_data_large_mode: str,
    transform_generic_data_small_model: str,
    transform_generic_data_small_api_key: str,
    transform_generic_data_small_base_url: str,
    transform_generic_data_small_mode: str,
    transform_generic_cot_preface: str,
    transform_generic_cot_suffix: str,
    transform_generic_data_cost_per_million_small_input: float,
    transform_generic_data_cost_per_million_small_output: float,
    transform_generic_data_cost_per_million_large_input: float,
    transform_generic_data_cost_per_million_large_output: float,
    transform_generic_prompts: str,
    wandb_project: str,
    is_mistral_derived_model: str,
    do_train: str,
    do_run: str,
    runpod_api_key: str,
    huggingface_token: str,
    wandb_api_key: str,
    pod_id: str,
    cache_dir: str,
    task_id=None,  # task ID must always be specified as a pipeline arg. And it must be a kwarg, not an arg.
    base_model=None,
    other_pretrain_kwargs={},
    other_finetune_kwargs={},
    server_type="normal", # normal | rag | none, whether to use a rag server or normal one
    do_not_use_llm_for_pdf_processing=True, 
    *args,
    **kwargs,
):

    if kwargs:
        print("WARNING -- UNUSED KWARGS")
        print(json.dumps(kwargs, indent=4))
        print("-----------")
    # notably, no engine wrapper or even concurrency limit at the top level, since that is handled per-pipeline

    # create the appropriate chunks that will be used everywhere

    text_chunks_dict = {}
    # Target by the end of the input dirs is to have 0.30 progress. The good thing about the current progress interfaec is that the pipeline creator can be as detailed or as lazy as they want to be.

    set_progress(task_id, progress=0, message="Starting pretraining creation")
    progress = 0.0
    num_input_dirs = len(input_dirs)
    per_pretrain_group_score = 0.15 / num_input_dirs
    per_pretrain_step_score = (
        per_pretrain_group_score / 3
    )  # rather than calculating what the progress should be at each step, in cases where things are complex, it is better to make progress a variable and add to it as appropriate. The replay will always catch it up to where it was.
    for input_dir in input_dirs:
        input_dir["path"] = input_dir["path"].rstrip("/")  # footgun avoidance
        input_path = input_dir["path"]
        variation_generation_counts = input_dir["variation_generation_counts"]

        input_dir["input_dir_name"] = os.path.basename(input_path)

        texts_json = read_jsonl_completions(input_dir=input_path)
        texts_txt = read_text(
            input_dir=input_path, extensions=[".txt", ".md", ".epub", ".html", ".docx"]
        )
        texts_pdf = read_text(
            input_dir=input_path, extensions=[".pdf"]
        )  # only this one will be passed to a pdf-cleaning pipeline
        # Notably we don't work with chunks in the orchestration; we work with full texts and then chunk inside

        # First, we clean the PDFs
        pdf_text_list = await pdf_clean_and_convert_pipeline(
            use_subset=False,  # here, use subset means AFTER we chunk the texts
            subset_size=1000,
            chunk_size=pdf_cleaning_chunk_size,
            input_dir=input_path,
            output_dir=os.path.join(
                output_dir, f"pdf_cleaning_{input_dir['input_dir_name']}"
            ),
            completion_mode=completion_mode,
            small_model=pdf_cleaning_small_model,
            small_api_key=pdf_cleaning_small_api_key,
            small_base_url=pdf_cleaning_small_base_url,
            small_mode=pdf_cleaning_small_mode,
            large_model=pdf_cleaning_large_model,
            large_api_key=pdf_cleaning_large_api_key,
            large_base_url=pdf_cleaning_large_base_url,
            large_mode=pdf_cleaning_large_mode,
            concurrency_limit=concurrency_limit,
            use_stop=pdf_cleaning_use_stop,
            prompts=pdf_cleaning_prompts,
            default_prompts=pdf_cleaning_default_prompts,
            cost_per_million_small_input=pdf_cleaning_cost_small_input,
            cost_per_million_small_output=pdf_cleaning_cost_small_output,
            cost_per_million_large_input=pdf_cleaning_cost_large_input,
            cost_per_million_large_output=pdf_cleaning_cost_large_output,
            using_config=False,
            texts_passed_in=texts_pdf,
            do_meta_datagen=False,
            read_files_manually=False,
            meta_datagen_extras=[],
            meta_datagen_keys=[],
            chunking_output_dir=output_dir,
            do_not_use_llm=do_not_use_llm_for_pdf_processing,  # bugs detected in the prompt of this pipeline too late for us to change the model, so this basically does nothing special. MIND YOU, if you are using an API the bugs have been fixed in the pipeline itself, so you can safely turn do not use LLM OFF in the config you use.
        )

        text_chunks = (
            texts_json + texts_txt + pdf_text_list
        )  # does not matter if we hash inside or outside of the pipelines themselves since we can match the hashes in the output with the hashes in the input since they are, after all, hashes
        text_chunks_dict[input_dir["input_dir_name"]] = text_chunks  # for later access
        assert len(text_chunks) > 0
        # Overall we want progress to = 0.3 by the end of the repvar, inferred facts, and pdf. So whatever that is, it all adds to 0.3.
        # And however many things there are, they must each sum to = 0.3/len(input_dirs)
        # and since there are three steps here, pdf, repvar, and inferred, each step is worth 0.3/len(input_dirs) / 3
        progress = progress + per_pretrain_step_score
        set_progress(
            task_id=task_id,
            progress=progress,
            message=f"PDF cleaning for {input_dir} done! Moving onto representation variation",
        )

        # next, representation variation
        print(f"Generating representation variations for {input_dir['input_dir_name']}")
        representation_variations = await representation_variation_pipeline(
            use_subset=use_subset,
            subset_size=subset_size,
            chunk_size=representation_variation_chunk_size,
            input_dir=input_path,
            output_dir=os.path.join(
                output_dir, f"representation_variation_{input_dir['input_dir_name']}"
            ),
            completion_mode=completion_mode,
            small_model=representation_variation_small_model,
            small_api_key=representation_variation_small_api_key,
            small_base_url=representation_variation_small_base_url,
            small_mode=representation_variation_small_mode,
            large_model=representation_variation_large_model,
            large_api_key=representation_variation_large_api_key,
            large_base_url=representation_variation_large_base_url,
            large_mode=representation_variation_large_mode,
            concurrency_limit=concurrency_limit,
            use_stop=representation_variation_use_stop,
            prompts=representation_variation_prompts,
            default_prompts=representation_variation_default_prompts,
            variation_generator_count=variation_generation_counts,
            include_context_in_dataset=True,
            dataset_context=dataset_context,
            code_variation_functions=code_variation_functions,
            using_config=False,
            texts_passed_in=text_chunks,
            make_inferred_facts=False,
            chunking_output_dir=output_dir,
            cost_per_million_large_input=representation_variation_cost_large_input,
            cost_per_million_large_output=representation_variation_cost_large_output,
            cost_per_million_small_input=representation_variation_cost_small_input,
            cost_per_million_small_output=representation_variation_cost_large_output,
        )

        progress = progress + per_pretrain_step_score
        set_progress(
            task_id=task_id,
            progress=progress,
            message=f"Representation variation for {input_dir} done! Moving onto inferred facts variation",
        )

        print(f"Generating inferred facts for {input_dir['input_dir_name']}")
        inferred_facts = await representation_variation_pipeline(  # turns out inferred facts can just be repvar but with a different prompt
            use_subset=use_subset,
            subset_size=subset_size,
            chunk_size=representation_variation_chunk_size,
            input_dir=input_path,
            output_dir=os.path.join(
                output_dir, f"inferred_facts_{input_dir['input_dir_name']}"
            ),
            completion_mode=completion_mode,
            small_model=representation_variation_small_model,
            small_api_key=representation_variation_small_api_key,
            small_base_url=representation_variation_small_base_url,
            small_mode=representation_variation_small_mode,
            large_model=representation_variation_large_model,
            large_api_key=representation_variation_large_api_key,
            large_base_url=representation_variation_large_base_url,
            large_mode=representation_variation_large_mode,
            concurrency_limit=concurrency_limit,
            use_stop=representation_variation_use_stop,
            prompts=representation_variation_prompts_inferred,
            default_prompts=representation_variation_default_prompts_inferred,
            variation_generator_count=variation_generation_counts,
            include_context_in_dataset=True,
            dataset_context=dataset_context,
            code_variation_functions=code_variation_functions,
            using_config=False,
            texts_passed_in=text_chunks,
            make_inferred_facts=True,
            chunking_output_dir=output_dir,
        )

        progress = progress + per_pretrain_step_score
        set_progress(
            task_id=task_id,
            progress=progress,
            message=f"Representation variation for {input_dir} done! Moving onto next step (either saving all the pretraining, or the next input dir)",
        )

    print("\n\nText Chunks Dict Keys")  # DEBUG
    print(text_chunks_dict.keys())

    pretraining_dir = os.path.join(output_dir, "pretraining_run")
    os.makedirs(pretraining_dir, exist_ok=True)
    # Create pretraining run directory

    for input_dir in input_dirs:
        input_dir_name = input_dir["input_dir_name"]
        print(f"Processing files for input directory: {input_dir_name}")

        # Copy final_output.jsonl from representation variations
        repvar_output = os.path.join(
            output_dir,
            f"representation_variation_{input_dir_name}",
            "final_output.jsonl",
        )
        repvar_dest = os.path.join(
            pretraining_dir, f"representation_variation_{input_dir_name}.jsonl"
        )
        if os.path.exists(repvar_output):
            print(
                f"Copying representation variation data from {repvar_output} to {repvar_dest}"
            )
            shutil.copy2(repvar_output, repvar_dest)
        else:
            print(
                f"Warning: Representation variation file not found at {repvar_output}"
            )

        # Copy final_output.jsonl from inferred facts
        inferred_facts_output = os.path.join(
            output_dir, f"inferred_facts_{input_dir_name}", "final_output.jsonl"
        )
        inferred_facts_dest = os.path.join(
            pretraining_dir, f"inferred_facts_{input_dir_name}.jsonl"
        )
        if os.path.exists(inferred_facts_output):
            print(
                f"Copying inferred facts data from {inferred_facts_output} to {inferred_facts_dest}"
            )
            shutil.copy2(inferred_facts_output, inferred_facts_dest)
        else:
            print(f"Warning: Inferred facts file not found at {inferred_facts_output}")

        # Save text chunks to jsonl
        text_chunks_path = os.path.join(
            pretraining_dir, f"text_chunks_{input_dir_name}.jsonl"
        )
        print(f"Saving {len(text_chunks)} text chunks to {text_chunks_path}")
        with open(text_chunks_path, "w", encoding="utf-8") as f:
            for chunk in text_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Successfully saved text chunks for {input_dir_name}")

    # Write the Axolotl config file
    axolotl_config_path = os.path.join(
        pretraining_dir, "axolotl_pretraining_config.yaml"
    )

    # build dataset paths
    dataset_paths = []
    for input_dir in input_dirs:
        input_dir_name = os.path.basename(input_dir["path"])
        dataset_paths.append(
            create_completion_dataset(
                f"representation_variation_{input_dir_name}.jsonl"
            )
        )
        dataset_paths.append(
            create_completion_dataset(f"text_chunks_{input_dir_name}.jsonl")
        )
        dataset_paths.append(
            create_completion_dataset(f"inferred_facts_{input_dir_name}.jsonl")
        )

    write_training_config(
        dataset_paths=dataset_paths,
        base_model=base_model,
        output_path=axolotl_config_path,
        wandb_project=wandb_project,
        hub_strategy=pretrain_hub_strategy,
        hub_model_id=pretrain_hub_model_id,
        is_mistral_derived_model=is_mistral_derived_model,
        sequence_length=context_length,
        **other_pretrain_kwargs,
    )

    # this version of augmentoolkit: how I learned to stop worrying and love the dict (over the list)

    factual_sft_generation_weight = 0.2  # progress tracking
    per_all_factual_sft_score = (
        factual_sft_generation_weight / number_of_factual_sft_generations_to_do
    )
    per_factual_sft_method_score = per_all_factual_sft_score / len(factual_sft.items())

    all_by_index = {}
    multi_turn_by_index = {}
    combined_chats = []

    for idx, prompt in enumerate(final_assistant_prompts_no_rag):
        if input_dir["final_system_prompt_additional_context"]:
            final_assistant_prompts_no_rag[idx] = (
                prompt + " " + str(input_dir["final_system_prompt_additional_context"])
            )
    for input_dir in input_dirs:
        # first, we need to load the prompts
        factual_sft_outputs = []
        input_dir_name = input_dir["input_dir_name"]
        for i in range(number_of_factual_sft_generations_to_do):
            for way in factual_sft:
                print(
                    f"""\n\n==================== CURRENTLY EXECUTING FACTUAL GENERATION WITH {way} INDEX {i} PROMPT SET WITH INPUT DIR {input_dir} ====================\n\n"""
                )
                prompts = factual_sft[way]["prompts"]
                default_prompts = factual_sft[way]["default_prompts"]
                single_turn = factual_sft[way]["single_turn"]
                skip_question_check = factual_sft[way]["skip_question_check"]
                skip_answer_relevancy_check = factual_sft[way][
                    "skip_answer_relevancy_check"
                ]
                skip_answer_accuracy_check = factual_sft[way][
                    "skip_answer_accuracy_check"
                ]
                skip_repair_qa_tuples = factual_sft[way]["skip_repair_qa_tuples"]

                print("OUTPUT DIR", os.path.join(output_dir, f"factual_sft_{way}"))
                # Then we call the pipeline

                this_kwargs = {  # the interface of these two pipelines is the same since multi source is a stripped-down and modified version of factual generation
                    "completion_mode": completion_mode,
                    "phase_index": 0,
                    "work_in_phases": False,
                    "skip_filter_chunks": False,
                    "skip_repair_qa_tuples": skip_repair_qa_tuples,
                    "chunk_size": factual_chunk_size,
                    "use_gutenberg": False,
                    "start_url": "",
                    "max_books": 0,
                    "max_failures": 0,
                    "skip_conversation_generation": True,
                    "hub_path": "",
                    "private": False,
                    "push_to_hub": False,
                    "use_filenames": True,
                    "input_dir": input_dir["path"],
                    "prompts": prompts,
                    "default_prompts": default_prompts,
                    "use_stop": False,
                    "skip_answer_relevancy_check": skip_answer_relevancy_check,
                    "skip_answer_accuracy_check": skip_answer_accuracy_check,
                    "conversation_instructions": None,
                    "do_not_use_system_prompts": False,
                    "skip_question_check": skip_question_check,
                    "final_assistant_prompts_no_rag": final_assistant_prompts_no_rag,
                    "final_assistant_prompts_rag": [
                        "does not matter, RAG in this pipeline is deprecated, there's a separate pipeline for that now"
                    ],
                    "rag_failure_percentage": 0.50,
                    "items_per_conversation": 1,
                    "concurrency_limit": concurrency_limit,
                    "small_model": factual_small_model,
                    "small_api_key": factual_small_api_key,
                    "small_base_url": factual_small_base_url,
                    "small_mode": factual_small_mode,
                    "large_model": factual_large_model,
                    "large_api_key": factual_large_api_key,
                    "large_base_url": factual_large_base_url,
                    "large_mode": factual_large_mode,
                    "use_subset": input_dir["factual_gen_use_subset"],
                    "subset_size": input_dir["factual_gen_subset_size_per_way"],
                    "double_check_counter": 1,
                    "output_dir": os.path.join(
                        output_dir, f"factual_sft_{input_dir_name}_{way}_{i}"
                    ),
                    "cost_per_million_small_input": factual_cost_per_million_small_input,
                    "cost_per_million_small_output": factual_cost_per_million_small_output,
                    "cost_per_million_large_input": factual_cost_per_million_large_input,
                    "cost_per_million_large_output": factual_cost_per_million_large_output,
                    "read_files_manually": False,
                    "text_chunks_passed_in": text_chunks_dict[input_dir_name],
                    "cite_sources_at_end": cite_sources_at_end,
                }
                if factual_sft[way]["multi_source"]:
                    if factual_sft[way]["multi_source"] == None:
                        raise Exception(f"PLEASE MANUALLY SPECIFY multi_source for way {way} as True or False, do not leave it blank! This will lead to unexpected behavior")
                    this_output, _ = await generate_multi_source_dataset(
                        **this_kwargs, chunking_output_dir=output_dir
                    )
                else:
                    this_output, _ = await generate_factual_qa_dataset(
                        **this_kwargs, chunking_output_dir=output_dir
                    )
                factual_sft_outputs.append(
                    {"way": way, "output": this_output, "index": i}
                )  # TODO replace {data} with the dataset context; {data_uppercase} and {data_lowercase} are also available and replace them with dataset context uppercased and lowercased respectively
                # Note we will combine and shuffle the singleturn ones together, the multiturns stay separate
                progress = progress + per_factual_sft_method_score
                set_progress(
                    task_id=task_id,
                    progress=progress,
                    message=f"Completed factual SFT method {way} (iteration {i})! Continuing on...",
                )

        # Separate single and multi-turn outputs
        all_outputs = []

        # TODO update the repvar pipeline with the new atomic fact reading logic

        for output in factual_sft_outputs:
            all_outputs.append(output)

        # Add the system prompt prefix to the single- and multi-turn outputs
        # if input_dir["final_system_prompt_additional_context"]:
        #     for output in all_outputs:
        #         # if there is a system message, add the prefix to the first message
        #         # First check that there are messages at all, don't want to error:
        #         for conv in output["output"]:
        #             if conv["conversations"] and conv["conversations"][0]["from"] == "system":
        #                 conv["conversations"][0]["value"] = input_dir["final_system_prompt_additional_context"] + " " + conv["conversations"][0]["value"]
        #             else:
        #                 # if there is no system message, add the prefix to the first message
        #                 output["output"].insert(0, {"conversations": [{"from": "system", "value": input_dir["final_system_prompt_additional_context"]}]})

        # NOTE we do not do this for multi-turn outputs because the only multi-turn output pipeline is the followup questions, which are deliberately excluded from additional processing and work due to a complex issue where it sort-of biased the LLM towards simplistic outputs if it had thought processes, followups, etc.

        # for output in multi_turn_outputs:
        #     for conv in output["output"]:
        #         if conv["conversations"] and conv["conversations"][0]["from"] == "system":
        #             conv["conversations"][0]["value"] = input_dir["final_system_prompt_additional_context"] + " " + conv["conversations"][0]["value"]
        #         else:
        #             # if there is no system message, add one at the beginning of the conversation
        #             conv["conversations"].insert(0, {"from": "system", "value": input_dir["final_system_prompt_additional_context"]})

        # Process single-turn outputs
        if all_outputs:
            # Group outputs by index
            outputs_by_index = {}
            for output in all_outputs:
                index = output["index"]
                if index not in outputs_by_index:
                    outputs_by_index[index] = []
                outputs_by_index[index].append(output)

            # print(next(iter(outputs_by_index.items())))

            for out_idx, outs in outputs_by_index.items():

                # combined_single_turn = combine_single_turn_convs(
                #     outs,
                #     target_pairs=combine_sharegpt_target_pairs,  # Hardcoded as per requirement
                #     dataset_context=dataset_context
                # )
                for output_obj in outs:  # obj with way key, output key
                    all_by_index.setdefault(out_idx, []).extend(
                        output_obj["output"]
                    )  # it is a list of conversations

                # Save combined single-turn conversations
                # combined_path = os.path.join(output_dir, f"combined_sharegpts_{input_dir_name}_{out_idx}.jsonl")
                # with open(combined_path, 'w', encoding='utf-8') as f:
                #     for conv in combined_single_turn:
                #         f.write(json.dumps(conv, ensure_ascii=False) + '\n')

        # if multi_turn_by_index:
        #     first_index = next(iter(multi_turn_by_index))
        #     first_item = multi_turn_by_index[first_index][0] if multi_turn_by_index[first_index] else None
        # print("First item in multi_turn_by_index:", first_item)

        set_progress(
            task_id=task_id,
            progress=progress,
            message="Starting RAG dataset generation!",
        )  # Progress value is the same, but we want to change the message because we're on a different step now

        rag_asst_prompts = [
            shared_instruction
            + "\n"
            + p.replace("{context_lowercase}", dataset_context.lower())
            .replace("{context_uppercase}", dataset_context.upper())
            .replace("{context}", dataset_context)
            for p in final_assistant_prompts
        ]

        rag_data = await rag_data_pipeline(
            input_dir=input_dir["path"],
            output_dir=os.path.join(output_dir, f"rag_data_{input_dir_name}"),
            use_subset=input_dir["rag_use_subset"],
            subset_size=input_dir["rag_subset_size"],
            chunk_size=factual_chunk_size,
            completion_mode=completion_mode,
            using_config=False,
            text_chunks_passed_in=text_chunks,
            user_format=user_format,
            system_format=system_format,
            assistant_format=assistant_format,
            bos=bos,
            final_assistant_prompts=rag_asst_prompts,
            num_items_per_group=num_items_per_group,
            skip_filter_chunks=False,
            small_model=rag_small_model,
            small_api_key=rag_small_api_key,
            small_base_url=rag_small_base_url,
            small_mode=rag_small_mode,
            large_model=rag_large_model,
            large_api_key=rag_large_api_key,
            large_base_url=rag_large_base_url,
            large_mode=rag_large_mode,
            cost_per_million_small_input=rag_cost_per_million_small_input,
            cost_per_million_small_output=rag_cost_per_million_small_output,
            cost_per_million_large_input=rag_cost_per_million_large_input,
            cost_per_million_large_output=rag_cost_per_million_large_output,
            prompts=rag_prompts,
            default_prompts=rag_default_prompts,
            rag_failure_percentage=rag_failure_percentage,
            rag_max_chunks=rag_max_chunks,
            concurrency_limit=concurrency_limit,
            use_stop=rag_use_stop,
            chunking_output_dir=output_dir,
        )

        progress = progress + (
            0.05 / len(input_dirs)
        )  # we have 0.15 score left to divide up before we reach 0.50, 50%, the middle of the pipeline before training. Too little.
        set_progress(
            task_id=task_id,
            progress=progress,
            message=f"{idx} RAG dataset generation complete! Moving onto correction pipeline...",
        )

        # create correction data
        await correction_pipeline(
            use_subset=input_dir["correction_use_subset"],
            subset_size=input_dir["correction_subset_size"],
            chunk_size=correction_chunk_size,
            input_dir=input_dir["path"],
            output_dir=os.path.join(output_dir, f"corrections_{input_dir_name}"),
            small_model=correction_small_model,
            small_api_key=correction_small_api_key,
            small_base_url=correction_small_base_url,
            small_mode=correction_small_mode,
            large_model=correction_large_model,
            large_api_key=correction_large_api_key,
            large_base_url=correction_large_base_url,
            large_mode=correction_large_mode,
            cost_per_million_small_input=correction_cost_per_million_small_input,
            cost_per_million_small_output=correction_cost_per_million_small_output,
            cost_per_million_large_input=correction_cost_per_million_large_input,
            cost_per_million_large_output=correction_cost_per_million_large_output,
            prompt_template=correction_prompt_template,
            concurrency_limit=concurrency_limit,
            using_config=False,
            text_chunks_passed_in=text_chunks,
            use_stop=correction_use_stop,
            default_prompts=correction_default_prompts,
            prompts=correction_prompts,
            completion_mode=correction_completion_mode,
            chunking_output_dir=output_dir,
        )

        progress = progress + (0.05 / len(input_dirs))
        set_progress(
            task_id=task_id,
            progress=progress,
            message=f"{idx} Correction pipeline finished! Saving all final datasets and creating the SFT training config...",
        )

        # first we need to get all the factual domain SFT data in one place
        # This is the combined data, the multiturn data, the rag data, and the conversational start data. There's also the corrections data but that is not used as part of the token counting

        # pipelines to get into the factual_sft dir

        # Create directory for combined factual SFT data
        factual_sfts_combined_dir = os.path.join(
            output_dir, f"factual_sft_{input_dir_name}"
        )
        os.makedirs(factual_sfts_combined_dir, exist_ok=True)

    all_combined_files_info = []
    all_indices = set(all_by_index.keys())

    for out_idx in sorted(list(all_indices)):  # Sort for deterministic processing order
        single_pairs = all_by_index[out_idx]

        print(
            f"Combining for index {out_idx}: {len(single_pairs)} convs (across all dirs) "
        )

        # Determine combination range (assuming combine_sharegpt_target_pairs is max)
        min_combined_pairs = 1  # Or fetch from config if available
        # Ensure combine_sharegpt_target_pairs is an int; provide a default if necessary
        max_combined_pairs = (
            int(combine_sharegpt_target_pairs) if combine_sharegpt_target_pairs else 3
        )
        assert max_combined_pairs > min_combined_pairs

        # if not single_pairs and not multi_convs:
        #     print(f"Skipping combination for index {out_idx} - no data.")
        #     continue

        combined_conversations = combine_single_and_multi_turn(
            convs=single_pairs,
            min_pairs=min_combined_pairs,
            max_pairs=max_combined_pairs,
            dataset_context=dataset_context,
        )

        # insert the generic thought process adder if we are indeed using one of those

        if generic_thought_process_on_domain_data:
            # first remove ALL thought processes
            first_debug = False
            for c in combined_conversations:
                # print("entered for")
                # if not first_debug:
                #     print(c)
                for m in c["conversations"]:
                    if m["from"] == "gpt":
                        # print("entered from assistant branch thingy")
                        parts = m["value"].split(final_answer_str, 1)

                        if len(parts) > 1:
                            m["value"] = parts[1].strip()
                            if not first_debug:
                                print("CONVERSATION AFTER")
                                print(m["value"])
                                print(parts)
                                first_debug = True
                        else:
                            print("Problem! message did not contain final answer str!")

            combined_conversations = await transform_generic_data_pipeline(
                input_dir="notused",
                output_dir=os.path.join(
                    output_dir, f"transformed_generic_data_{out_idx}"
                ),  # Define an appropriate output dir
                use_stop=transform_generic_data_use_stop,
                large_model=transform_generic_data_large_model,
                large_api_key=transform_generic_data_large_api_key,
                large_base_url=transform_generic_data_large_base_url,
                large_mode=transform_generic_data_large_mode,
                small_model=transform_generic_data_small_model,
                small_api_key=transform_generic_data_small_api_key,
                small_base_url=transform_generic_data_small_base_url,
                small_mode=transform_generic_data_small_mode,
                cot_preface=transform_generic_cot_preface,
                cot_suffix=transform_generic_cot_suffix,
                cost_per_million_small_input=transform_generic_data_cost_per_million_small_input,
                cost_per_million_small_output=transform_generic_data_cost_per_million_small_output,
                cost_per_million_large_input=transform_generic_data_cost_per_million_large_input,
                cost_per_million_large_output=transform_generic_data_cost_per_million_large_output,
                concurrency_limit=concurrency_limit,  # Assuming concurrency_limit is available in this scope
                completion_mode=False,
                subset_size=30,
                use_subset=False,
                read_files_manually=False,
                default_prompts="prompts",
                prompts=transform_generic_prompts,
                sharegpt_convs_passed_in=combined_conversations,
                # Add other necessary parameters for transform_generic_data_pipeline if any
            )

        # if there is a rephrase pipeline in the future, it will go here
        set_progress(task_id=task_id, progress=0.50, message="All data pipeline steps completed; processing and preparing final set for training")

        deterministic_rand = random.Random(11037)
        # Apply system prompt and thought process removal based on ratios
        for conv in combined_conversations:

            # print(
            #     conv["conversations"] and conv["conversations"][0]["from"] == "system"
            # )
            # print(
            #     f'Because {conv["conversations"]} and {conv["conversations"][0]["from"] == "system"}'
            # )
            if (
                deterministic_rand.random() < remove_system_prompt_ratio
                and conv["conversations"]
                and conv["conversations"][0]["from"] == "system"
            ):
                # print("CONV")
                # print(conv)
                conv["conversations"].pop(0)
                # removed_any_sysprompt = True

            # removed_any_thoughts = False
            # Remove thought process with probability remove_thought_process_ratio
            if deterministic_rand.random() < remove_thought_process_ratio:
                # Add thought process removal prompt to system message if it exists
                if (
                    conv["conversations"]
                    and conv["conversations"][0]["from"] == "system"
                ):
                    conv["conversations"][0]["value"] = (
                        conv["conversations"][0]["value"]
                        + " "
                        + remove_thought_process_prompt
                        if not remove_thought_process_prompt.startswith(" ")
                        else conv["conversations"][0]["value"]
                        + remove_thought_process_prompt
                    )

                # Process each assistant message
                for msg in conv["conversations"]:
                    if msg["from"] == "gpt":
                        # Split on final_answer_str and keep only the answer part
                        parts = msg["value"].split(final_answer_str, 1)
                        if len(parts) > 1:
                            msg["value"] = parts[
                                1
                            ].strip()  # the .strip is essential to maintaining prompt format in this case

                # removed_any_thoughts = True

        # Define output path for the combined data for this index (across all dirs)
        combined_output_filename = f"combined_all_{out_idx}.jsonl"
        # Save temporarily in main output dir before moving to sft_run/combined_factual_data
        combined_output_path = os.path.join(output_dir, combined_output_filename)

        print(
            f"Saving {len(combined_conversations)} combined conversations for index {out_idx} to {combined_output_path}"
        )
        write_jsonl(
            combined_output_path, combined_conversations
        )  # Corrected order: path, data
        all_combined_files_info.append(
            {  # Add path to the global list for later copying
                "source_path": combined_output_path,
                "index": out_idx,
                "target_filename": f"combined_all_{out_idx}.jsonl",  # Filename for sft_run/combined_factual_data
            }
        )

    # TODO here we start the saving and downloading of generic sets and balancing

    template_str = route_template_to_preset(
        template_input=template, template_kwargs=template_kwargs
    )

    # make the SFT output directory
    # called "sft_run"
    sft_run_dir = os.path.join(output_dir, "sft_run")
    os.makedirs(sft_run_dir, exist_ok=True)
    sft_combined_factual_dir = os.path.join(sft_run_dir, "combined_factual_data")
    os.makedirs(sft_combined_factual_dir, exist_ok=True)

    print(f"\nMoving combined ALL files to {sft_combined_factual_dir}")
    for file_info in all_combined_files_info:
        source_path = file_info["source_path"]
        target_filename = file_info["target_filename"]
        target_path = os.path.join(sft_combined_factual_dir, target_filename)
        if os.path.exists(source_path):
            print(f"Copying {source_path} to {target_path}")
            # Use move instead of copy to avoid keeping temp files in output_dir
            shutil.move(source_path, target_path)
        else:
            print(f"Warning: Combined file not found at {source_path}")

    # Add shared instruction system prompt to the combined factual SFT files BEFORE completionifying
    print(f"Adding shared instruction to files in {sft_combined_factual_dir}")
    add_sysprompt_to_directory_files(
        directory_path=sft_combined_factual_dir, instruction=shared_instruction
    )

    # Completionify the combined factual data
    factual_sft_completion_dir = os.path.join(sft_run_dir, "factual_sft_completion")
    print(
        f"Completionifying combined data from {sft_combined_factual_dir} to {factual_sft_completion_dir}"
    )
    completionify_sharegpt(
        input_files=sft_combined_factual_dir,
        output_dir=factual_sft_completion_dir,
        template_str=template_str,
        **template_kwargs,
    )

    for input_dir in input_dirs:
        input_dir_name = input_dir["input_dir_name"]
        # The completionify step was moved above as combined data is not per-input-dir

        # Move that input dir's corrections to sft run dir
        # Ensure source directory exists before attempting copy
        correction_source_dir = os.path.join(
            output_dir, f"corrections_{input_dir_name}"
        )
        correction_source_file = os.path.join(
            correction_source_dir, "axolotl_correction_conversations.json"
        )
        correction_target_file = os.path.join(
            sft_run_dir, f"axolotl_correction_conversations_{input_dir_name}.json"
        )
        if os.path.exists(correction_source_file):
            print(
                f"Copying corrections from {correction_source_file} to {correction_target_file}"
            )
            shutil.copy(correction_source_file, correction_target_file)
        else:
            print(f"Warning: Correction file not found at {correction_source_file}")

        # Move that input dir's RAG data to sft run dir
        # Ensure source directory exists before attempting copy
        rag_source_dir = os.path.join(output_dir, f"rag_data_{input_dir_name}")
        rag_source_file = os.path.join(
            rag_source_dir, "axolotl_rag_conversations.jsonl"
        )
        rag_target_file = os.path.join(
            sft_run_dir, f"axolotl_rag_conversations_{input_dir_name}.jsonl"
        )
        if os.path.exists(rag_source_file):
            print(f"Copying RAG data from {rag_source_file} to {rag_target_file}")
            shutil.copy(rag_source_file, rag_target_file)
        else:
            print(f"Warning: RAG data file not found at {rag_source_file}")

    # These are sharegpt because the user might have a prompt template in mind
    # we want to support prompt template variety
    # I didn't make a new template because I wanted to fuck compatibility with every interface! I did it because it works better!
    # generic stuff is obviously not per-input-dir
    for hf_path, percentage in zip(generic_dataset_paths, generic_dataset_percentages):
        local_filename = (
            hf_path["path"].replace("/", "-") + ".jsonl"
        )  # Convert path to filename
        save_hf_dataset(
            hf_dataset_path=hf_path["path"],
            local_path=os.path.join(
                huggingface_cache_dir, "generic_sft", local_filename
            ),
            split="train",
            max_samples=max_samples_per_dataset,
        )

    # First, count the number of tokens in ALL the domain SFT data (across all input dirs)
    # i.e., in os.path.join(sft_run_dir, "factual_sft_completion")
    # Updated path to point to the single factual_sft_completion directory
    domain_sft_token_count = count_tokens_glob(
        paths=[factual_sft_completion_dir],
        count_all_turns=False,  # meaningless since it's completions anyway
    )

    domain_sft_token_count = max(domain_sft_token_count, minimum_generic_sft)
    print(f"Calculated domain SFT token count: {domain_sft_token_count:,}")

    # Now we subset the generic sft data to match the token count of the domain sft data
    # using create_subset
    generic_sft_subset_dir = os.path.join(
        output_dir, "generic_sft"
    )  # Define subset dir path
    os.makedirs(generic_sft_subset_dir, exist_ok=True)  # Ensure it exists
    for hf_path, percentage in zip(generic_dataset_paths, generic_dataset_percentages):
        local_filename_base = hf_path["path"].replace("/", "-")
        local_filename = local_filename_base + ".jsonl"  # Convert path to filename

        target_tokens_for_subset = int(
            domain_sft_token_count * percentage / 100
        )  # Ensure integer
        print(
            f"Targeting {target_tokens_for_subset:,} tokens for subset of {hf_path['path']}"
        )

        output_filename = local_filename_base + f"_{target_tokens_for_subset}.jsonl"
        create_subset(
            input_file=os.path.join(
                huggingface_cache_dir, "generic_sft", local_filename
            ),
            output_file=os.path.join(
                generic_sft_subset_dir, output_filename
            ),  # Save to subset dir
            target_tokens=target_tokens_for_subset,
            count_all_turns=True,
            context_to_add=hf_path["context_to_add"],
            context_to_add_type=hf_path["context_to_add_type"],
        )

    # Add shared instruction system prompt to the subsetted generic SFT files BEFORE completionifying
    print(
        f"Adding shared instruction to generic subset files in {generic_sft_subset_dir}"
    )
    add_sysprompt_to_directory_files(
        directory_path=generic_sft_subset_dir,  # Use the defined subset dir path
        instruction=shared_instruction,
    )

    # completionize the generic sft data and save it to the sft run directory NOTE This can only happen after we subset the cached data and save the subsets to the output directory (we deliberately do not save them to the sft run directory, the non-completionized ones; only the stuff used in training, not intermediate stuff, goes in sft run)
    generic_sft_completion_dir = os.path.join(
        sft_run_dir, "generic_sft_completion"
    )  # Define completion dir path
    print(
        f"Completionifying generic data from {generic_sft_subset_dir} to {generic_sft_completion_dir}"
    )

    os.makedirs(generic_sft_completion_dir, exist_ok=True)
    for filename in os.listdir(generic_sft_completion_dir):
        file_path = os.path.join(generic_sft_completion_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    completionify_sharegpt(
        input_files=generic_sft_subset_dir,  # Reads from the subsetted generic data directory
        output_dir=generic_sft_completion_dir,  # Save to completion dir
        template_str=template_str,
        **template_kwargs,
    )

    # --- START: Add Pretraining Data Subset to SFT ---

    # Calculate target pretrain tokens for SFT
    target_pretrain_tokens = 0
    if what_percent_of_sft_is_pretrain is not None:
        target_pretrain_tokens = (
            domain_sft_token_count * what_percent_of_sft_is_pretrain / 100
        )

    if num_tokens_pretraining_in_sft is not None:
        target_pretrain_tokens = max(
            target_pretrain_tokens, num_tokens_pretraining_in_sft
        )

    pretrain_subset_path = None  # Initialize path variable

    if target_pretrain_tokens > 0:
        print(
            f"Targeting {target_pretrain_tokens:,} tokens for pretraining data subset in SFT."
        )

        # Combine all text chunks from different input directories
        all_text_chunks = []
        for chunks_list in text_chunks_dict.values():
            all_text_chunks.extend(chunks_list)

        if not all_text_chunks:
            print("Warning: No text chunks found to create pretraining subset.")
        else:
            # Randomly sample text chunks to reach the target token count
            pretrain_subset = []
            current_pretrain_tokens = 0
            used_indices = set()

            random.seed(11037)  # puhuhu
            emergency_stop = 100000  # TODO verify that this will stop naturally even without the emergency stop. That is a silly thing to rely on.
            print(
                f"Starting random selection from {len(all_text_chunks)} text chunks..."
            )
            while (
                current_pretrain_tokens < target_pretrain_tokens
                and len(used_indices) < len(all_text_chunks)
                and emergency_stop > 0
            ):
                index = random.randint(0, len(all_text_chunks) - 1)
                # print("Looped")

                if index in used_indices:
                    continue

                item = all_text_chunks[index]
                # Assuming text chunks are dicts with a 'text' key for token counting
                tokens = count_item_tokens(item)

                pretrain_subset.append(item)
                current_pretrain_tokens += tokens
                used_indices.add(index)
                emergency_stop = emergency_stop - 1

            print(
                f"Selected {len(pretrain_subset)} items with total {current_pretrain_tokens:,} tokens for pretraining subset."
            )

            # Save the subset to the sft_run_dir
            pretrain_subset_filename = (
                f"pretraining_subset_{current_pretrain_tokens}.jsonl"
            )
            pretrain_subset_path = os.path.join(sft_run_dir, pretrain_subset_filename)
            print(f"Writing pretraining subset to {pretrain_subset_path}")
            # Use the write_jsonl function from data_prep_operations
            # Make sure write_jsonl is imported if not already
            write_jsonl(pretrain_subset_path, pretrain_subset)

    # --- END: Add Pretraining Data Subset to SFT ---

    # Save final config with proper dataset typing
    dataset_configs = []

    for root, dirs, files in os.walk(sft_run_dir):
        for fname in files:
            # Get path relative to sft_run_dir
            rel_path = os.path.relpath(os.path.join(root, fname), sft_run_dir)
            if "factual_sft_completion" in rel_path and fname.endswith(".jsonl"):
                dataset_configs.append(create_completion_dataset(rel_path))
            elif "axolotl_correction_conversations" in rel_path and fname.endswith(
                ".json"
            ):
                dataset_configs.append(create_input_output_dataset(rel_path))
            elif "axolotl_rag_conversations" in rel_path and fname.endswith(".jsonl"):
                dataset_configs.append(create_input_output_dataset(rel_path))
            elif "generic_sft_completion" in rel_path and fname.endswith(".jsonl"):
                dataset_configs.append(create_completion_dataset(rel_path))
            elif "pretraining_subset" in rel_path and fname.endswith(
                ".jsonl"
            ):  # Add condition for pretraining subset
                dataset_configs.append(create_completion_dataset(rel_path))
    print(dataset_configs)
    print("\nContents of SFT run directory:")
    for root, dirs, files in os.walk(sft_run_dir):
        level = root.replace(sft_run_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    write_training_config(
        dataset_paths=dataset_configs,
        base_model=pretrain_hub_model_id,
        output_path=os.path.join(sft_run_dir, "sft_training_config.yaml"),
        wandb_project=wandb_project,
        hub_strategy=finetune_hub_strategy,
        hub_model_id=finetune_hub_model_id,
        is_mistral_derived_model=is_mistral_derived_model,
        num_epochs=5,  # ideal is probably 4 but we want variations to try
        dataset_prepared_path="last_finetune_prepared",
        output_dir="./finetune-model-output",
        sequence_length=context_length,
        **other_finetune_kwargs,
    )

    rag_source_data_dir = os.path.join(output_dir, "rag_source_data")
    os.makedirs(rag_source_data_dir, exist_ok=True)
    print(f"Created RAG source data directory: {rag_source_data_dir}")

    all_rag_items_combined = (
        []
    )  # TODO with the automatic training, make it not start training again if the model has been trained and we see it is on huggingface and the final checkpoint commit name is end of training. Make it independently check for pretrain vs finetuning of that. And when downloading, make it pre-check for the model files there, and not download if it is present. And not convert if it is present. This keeps the auto-resume strong.

    for input_dir_config in input_dirs:
        current_input_dir_name = input_dir_config["input_dir_name"]
        current_input_dir_rag_items = []
        print(
            f"Processing RAG source data for input directory: {current_input_dir_name}"
        )

        for way in factual_sft.keys():
            if (
                "not_in_rag_data" in factual_sft[way]
                and factual_sft[way]["not_in_rag_data"] == True
            ):
                print(f"SKIPPING {input_dir_config} {way}!!!")
                continue
            for i in range(number_of_factual_sft_generations_to_do):
                sft_dir_path = os.path.join(
                    output_dir, f"factual_sft_{current_input_dir_name}_{way}_{i}"
                )
                factual_questions_path = os.path.join(
                    sft_dir_path, "factual_questions.json"
                )

                if os.path.exists(factual_questions_path):
                    print(f"Found factual_questions.json in {sft_dir_path}")
                    try:
                        with open(factual_questions_path, "r", encoding="utf-8") as f:
                            factual_questions_data = json.load(f)

                        if isinstance(factual_questions_data, dict):
                            for _hash, item_data in factual_questions_data.items():
                                if isinstance(item_data, dict):
                                    question = item_data.get("question")
                                    text = item_data.get("text")
                                    metadata = item_data.get("metadata")

                                    related_chunks_list = item_data.get(
                                        "related_chunks", []
                                    )
                                    processed_related_chunks = []
                                    if isinstance(related_chunks_list, list):
                                        for rc in related_chunks_list:
                                            if isinstance(rc, dict):
                                                processed_related_chunks.append(
                                                    {
                                                        "text": rc.get("text"),
                                                        "metadata": rc.get("metadata"),
                                                    }
                                                )

                                    if (
                                        question and text and metadata
                                    ):  # Ensure essential fields are present
                                        current_input_dir_rag_items.append(
                                            {
                                                "question": question,
                                                "source_text": text,
                                                "source_metadata": metadata,
                                                "related_chunks": processed_related_chunks,
                                            }
                                        )
                                    else:
                                        print(
                                            f"Skipping item in {factual_questions_path} due to missing essential fields (question, text, or metadata). Hash: {_hash}"
                                        )
                        else:
                            print(
                                f"Warning: factual_questions.json in {sft_dir_path} is not a dictionary. Skipping."
                            )
                    except json.JSONDecodeError:
                        print(
                            f"Error decoding JSON from {factual_questions_path}. Skipping."
                        )
                    except Exception as e:
                        print(
                            f"An unexpected error occurred while processing {factual_questions_path}: {e}. Skipping."
                        )
                # else:
                # print(f"factual_questions.json not found in {sft_dir_path}")

        if current_input_dir_rag_items:
            output_file_path = os.path.join(
                rag_source_data_dir, f"rag_data_{current_input_dir_name}.jsonl"
            )
            print(
                f"Saving {len(current_input_dir_rag_items)} RAG items for {current_input_dir_name} to {output_file_path}"
            )
            write_jsonl(output_file_path, current_input_dir_rag_items)
            all_rag_items_combined.extend(current_input_dir_rag_items)
        else:
            print(f"No RAG items found for input directory: {current_input_dir_name}")

    if all_rag_items_combined:
        combined_output_file_path = os.path.join(
            rag_source_data_dir, "rag_data_combined.jsonl"
        )
        print(
            f"Saving {len(all_rag_items_combined)} combined RAG items to {combined_output_file_path}"
        )
        write_jsonl(combined_output_file_path, all_rag_items_combined)
    else:
        print("No RAG items found across all input directories to combine.")

    if do_train:
        print("Automated training is enabled. Starting training process...")
        set_progress(task_id=task_id, progress=0.51, message="Automatic training is starting!")
        await _run_automated_training(
            output_dir=output_dir,
            runpod_api_key=runpod_api_key,
            huggingface_token=huggingface_token,
            wandb_api_key=wandb_api_key,
            pod_id=pod_id,
            pretrain_hub_model_id=pretrain_hub_model_id,
            finetune_hub_model_id=finetune_hub_model_id,
            task_id=task_id,
        )
    else:
        print("Automated training is disabled. Skipping training process.")
        
    all_texts = []
    for k, v in text_chunks_dict.items():
        all_texts.extend(v)
        
    # Then we download it from huggingface
    set_progress(task_id=task_id, progress=0.99, message="Now that training is finished, automatic downloading and conversion for inference is starting. To change this, turn do_train off. Make sure you have enough disk space!")

    api = HfApi()
    if (
        do_run and do_train
    ):  # need to train to run. If not training, what could we possibly be running? Since the model would have existed before our dataset generation
        finished_model_path = os.path.join(
            models_dir, finetune_hub_model_id.split("/")[1]
        )
        os.makedirs(finished_model_path, exist_ok=True)

        # Check if finished_model directory exists and has files
        if os.path.exists(finished_model_path) and os.listdir(finished_model_path):
            print(
                f"Finished model directory {finished_model_path} is not empty. Skipping download."
            )
        else:
            print(f"Downloading model files from {finetune_hub_model_id}...")
            files = api.list_repo_tree(finetune_hub_model_id)
            # download each file at the top level of the repo that does not have "optimizer" or "trainer_state" in its name
            for item in files:
                # Check if it's a file (not folder) and at top level (no '/' in path)
                if hasattr(item, "size") and "/" not in item.path:
                    # Skip files with "optimizer" or "trainer_state" in the name
                    if (
                        "optimizer" not in item.path.lower()
                        and "trainer_state" not in item.path.lower()
                    ):
                        # Download the file
                        api.hf_hub_download(
                            repo_id=finetune_hub_model_id,
                            filename=item.path,
                            local_dir=finished_model_path,  # specify where to save
                        )
                        print(f"Downloaded: {item.path}")

        # Check for a ./llama.cpp/ folder in the root of this project
        llama_cpp_dir = "./llama.cpp"
        if not os.path.exists(llama_cpp_dir):
            print("llama.cpp directory not found. Cloning repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggml-org/llama.cpp.git"],
                check=True,
            )
        
        venv_path = os.path.join(llama_cpp_dir, ".lcpp_venv")
        if not os.path.exists(venv_path):
            # Create virtual environment inside llama.cpp folder
            print(f"Creating virtual environment at {venv_path}...")
            subprocess.run([
                sys.executable, "-m", "venv", venv_path
            ], check=True)
            
        # Determine the python executable path for the virtual environment
        if platform.system() == "Windows":
            venv_python = os.path.join(venv_path, "Scripts", "python.exe")
            venv_pip = os.path.join(venv_path, "Scripts", "pip.exe")
        else:
            venv_python = os.path.join(venv_path, "bin", "python")
            venv_pip = os.path.join(venv_path, "bin", "pip")
        
        # Install requirements if requirements.txt exists
        requirements_path = os.path.join(llama_cpp_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Installing llama.cpp requirements in virtual environment...")
            subprocess.run([
                venv_pip, "install", "-r", requirements_path
            ], check=True)
        else:
            print("No requirements.txt found in llama.cpp, skipping pip install")

        # Check if llama-server exists
        llama_server_path = os.path.join(llama_cpp_dir, "build", "bin", "llama-server")
        if platform.system() == "Windows":
            llama_server_path += ".exe"

        if not os.path.exists(llama_server_path):
            print("llama-server not found. Building llama.cpp...")

            # Detect if NVIDIA GPU is available
            has_nvidia_gpu = False
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=False, text=True)
                has_nvidia_gpu = result.returncode == 0
            except FileNotFoundError:
                has_nvidia_gpu = False

            # Build with appropriate flags
            build_cmd = ["cmake", "-B", "build"]
            if has_nvidia_gpu:
                build_cmd.append("-DGGML_CUDA=ON")
                print("NVIDIA GPU detected. Building with CUDA support...")
            else:
                print("No NVIDIA GPU detected. Building CPU-only version...")

            # Run cmake configure
            subprocess.run(build_cmd, cwd=llama_cpp_dir, check=True)

            # Build the project
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=llama_cpp_dir,
                check=True,
            )

        # Convert HF model to GGUF
        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

        print(f"Converting model from {finished_model_path} to GGUF format...")
        # Check if a .gguf file already exists in the finished_model_path directory
        existing_gguf_files = glob.glob(os.path.join(finished_model_path, "*.gguf"))
        if existing_gguf_files:
            print(
                f"GGUF file already exists in {finished_model_path}: {existing_gguf_files[0]}"
            )
        else:
            outfile = os.path.join(
                finished_model_path,
                finetune_hub_model_id.split("/")[1] + ".gguf",
            )
            
            # Determine the python executable path for the virtual environment
            venv_path = os.path.join(llama_cpp_dir, ".lcpp_venv")
            if platform.system() == "Windows":
                venv_python = os.path.join(venv_path, "Scripts", "python.exe")
            else:
                venv_python = os.path.join(venv_path, "bin", "python")
            
            # Use the virtual environment python if it exists, otherwise fall back to system python
            if os.path.exists(venv_python):
                python_executable = venv_python
                print(f"Using virtual environment python: {python_executable}")
            else:
                python_executable = "python"
                print("Virtual environment not found, using system python")
            
            command_str = f'"{python_executable}" "{convert_script}" "{finished_model_path}" --outtype "q8_0" --outfile "{outfile}"'
            result = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                check=False,  # Don't raise exception immediately
            )

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

        print("Model conversion to GGUF complete!")

        # Build system prompt based on the information we have.
        system_prompt = (
            shared_instruction
            + "\n\n"
            + final_assistant_prompts_no_rag[0]
            .replace("{context_lowercase}", dataset_context.lower())
            .replace("{context_uppercase}", dataset_context.upper())
            .replace("{context}", dataset_context)
        )  # No addition of the context which we already have. # + ' ' + input_dirs[0]['final_system_prompt_additional_context']

        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(system_prompt)  # save prompt

        with open(os.path.join(output_dir, "template.txt"), "w") as f:
            f.write(correction_prompt_template)  # save template

        # Find the GGUF model file
        gguf_files = glob.glob(os.path.join(finished_model_path, "*.gguf"))
        if not gguf_files:
            print("Error: No GGUF model file found!")
            return
        gguf_model_path = gguf_files[0]
        print(f"Found GGUF model: {gguf_model_path}")

        # Build llama-server path
        llama_server_path = os.path.join(llama_cpp_dir, "build", "bin", "llama-server")
        if platform.system() == "Windows":
            llama_server_path += ".exe"

        # Start llama-server in background
        print(f"Starting llama-server with model: {gguf_model_path}")
        server_cmd = [
            llama_server_path,
            "-m",
            gguf_model_path,
            "-c",
            str(context_length),
        ]
        server_process = subprocess.Popen(
            server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Started llama-server with PID: {server_process.pid}")

        try:
            # Give the server a moment to start up
            print("Please wait...")
            time.sleep(10)
            print("OK you can stop waiting")
            set_progress(
                task_id=task_id,
                progress=1.0,
                message="Pipeline Complete! Chat loop starting; feel free to terminate whenever.",
            )
            # Start chat loop and run until exit
            print("Chat loop started! Feel free to chat with your model")
            if not task_id:
                await simple_chat_loop(
                    system_prompt,
                    correction_prompt_template,
                    context_length,
                    finetune_hub_model_id,
                )
            else: # if we are on the interface, run the server
                documents_dir = os.path.join(output_dir, "documents_all")
                write_text(documents_dir, text_list=all_texts)
                
                if server_type == "rag":
                    await rag_server(
                        prompt_path = os.path.join(output_dir, "prompt.txt"),
                        template_path = os.path.join(output_dir, "template.txt"),
                        gguf_model_path = os.path.join(
                            f"{finished_model_path}",
                            finetune_hub_model_id.split("/")[1] + ".gguf",
                        ),
                        context_length=context_length,
                        documents_dir = documents_dir,
                        questions_jsonl_path = combined_output_file_path,
                        question_chunk_size = 500,
                        top_k = 3,
                        llama_path='./llama.cpp',
                        port=8003,
                        cache_dir = cache_dir,
                        collection_name = 'questions_collection',
                        max_shrink_iterations = 10,
                        task_id=task_id
                    )
                else:
                    await llm_server(
                        prompt_path = os.path.join(output_dir, "prompt.txt"),
                        template_path = os.path.join(output_dir, "template.txt"),
                        gguf_model_path = os.path.join(
                            f"{finished_model_path}",
                            finetune_hub_model_id.split("/")[1] + ".gguf",
                        ),
                        context_length=context_length,
                        llama_path='./llama.cpp',
                        port=8003,
                        task_id=task_id
                    )
        finally:
            # Clean up the llama-server process
            if server_process:
                print("Terminating llama-server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=5)
                    print("llama-server terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("Force killing llama-server...")
                    server_process.kill()
                    server_process.wait()
                    print("llama-server force killed")

        # shared+instruction + ' ' + factual_assistant_prompts_no_rag[0] + ' ' + first input dir's final_system_prompt_additional_context

        # then we need to start a chat loop with it. This part is simple, just a back and forth with the system prompt. Hosting an OAI-compatible API proxy is in the cards -- but for utility function stuff. Like, that one will kick off a lcpp llama-server and then send the proper prompts to it if it is the chat competions route otherwise just redirect the request as a proxy would -- simple thing. Hardly any complexity to it.

    elif do_run and not do_train:
        print(
            "Error! You can only automatically run a model after training, if you train it. (do run True do train false)"
        )

    set_progress(
        task_id=task_id,
        progress=1.0,
        message="FULL FACTUAL DATASET GENERATION PIPELINE, COMPLETE!",
    )


async def _check_if_model_already_trained(repo_id: str, huggingface_token: str) -> bool:
    """
    Check if a Hugging Face model repository already exists and has been trained.

    Args:
        repo_id: The Hugging Face repository ID (e.g., "username/model-name")
        huggingface_token: Hugging Face API token for authentication

    Returns:
        bool: True if the model exists and has been trained, False otherwise
    """
    try:
        api = HfApi(token=huggingface_token)

        # Check if the repository exists
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as e:
            print(f"Repository {repo_id} does not exist or is not accessible: {e}")
            return False

        # List all files in the repository
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        except Exception as e:
            print(f"Could not list files in repository {repo_id}: {e}")
            return False

        # Check if there are any model*.safetensors files
        model_files = [
            f for f in files if f.startswith("model") and f.endswith(".safetensors")
        ]

        if not model_files:
            print(f"No model*.safetensors files found in {repo_id}")
            return False

        print(f"Found model files in {repo_id}: {model_files}")

        # Try to check the last commit message for "end of training"
        try:
            commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="model"))
            if commits:
                last_commit = commits[0]
                commit_message = (
                    last_commit.title.lower() if hasattr(last_commit, "title") else ""
                )
                if "end of training" in commit_message:
                    print(
                        f"Repository {repo_id} has model files AND last commit contains 'end of training'. Model appears to be fully trained."
                    )
                    return True
                else:
                    print(
                        f"Repository {repo_id} has model files but last commit message does not contain 'end of training': '{last_commit.title if hasattr(last_commit, 'title') else 'No title'}'"
                    )
                    # Fall back to just checking for model files
                    print(
                        f"Falling back to simple check: model files exist, assuming training is complete."
                    )
                    return True
            else:
                print(
                    f"No commits found in {repo_id}, but model files exist. Assuming training is complete."
                )
                return True
        except Exception as e:
            print(f"Could not check commit history for {repo_id}: {e}")
            # Fall back to just checking for model files
            print(
                f"Falling back to simple check: model files exist, assuming training is complete."
            )
            return True

    except Exception as e:
        print(f"Error checking if model {repo_id} is already trained: {e}")
        return False


async def _run_automated_training(
    output_dir: str,
    runpod_api_key: str,
    huggingface_token: str,
    wandb_api_key: str,
    pod_id: str | None,
    pretrain_hub_model_id: str,
    finetune_hub_model_id: str,
    pretrain_config_name: str = "axolotl_pretraining_config.yaml",
    sft_config_name: str = "sft_training_config.yaml",
    task_id=None,
):
    """Runs the automated training on RunPod."""
    set_progress(
        task_id=task_id, progress=0.51, message="Starting automated training setup."
    )  # Assuming 0.95 was end of RAG source data gen

    # Check if pretraining model already exists and is trained
    set_progress(
        task_id=task_id,
        progress=0.51,
        message="Checking if pretraining model already exists on Hugging Face...",
    )
    skip_pretraining = await _check_if_model_already_trained(
        pretrain_hub_model_id, huggingface_token
    )

    skip_finetuning = await _check_if_model_already_trained(
        finetune_hub_model_id, huggingface_token=huggingface_token
    )

    if skip_pretraining and skip_finetuning:
        return

    pretraining_run_path = os.path.join(output_dir, "pretraining_run")
    sft_run_path = os.path.join(output_dir, "sft_run")

    public_key_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
    my_public_key = ""
    try:
        with open(public_key_path, "r") as f:
            my_public_key = f.read().strip()
    except FileNotFoundError:
        print(
            f"ERROR: Public key file not found at {public_key_path}. Training cannot proceed without it."
        )
        set_progress(
            task_id=task_id,
            progress=0.51,
            message=f"ERROR: SSH Public key not found at {public_key_path}. Training aborted.",
        )
        raise

    ssh_options = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    import runpod  # import down here so it works even if they have not installed runpod

    runpod.api_key = runpod_api_key

    created_pod_id = pod_id
    if created_pod_id == None:
        print("entered IF")
        env_dict = {"PUBLIC_KEY": my_public_key}

        set_progress(task_id=task_id, progress=0.51, message="Creating RunPod pod...")
        created_pod = runpod.create_pod(
            "trainingtest",
            "axolotlai/axolotl-cloud:main-latest",
            gpu_type_id="NVIDIA H100 80GB HBM3",
            gpu_count=1,
            volume_in_gb=500,
            container_disk_in_gb=500,
            start_ssh=True,
            ports="22/tcp",
            env=env_dict,
            cloud_type="SECURE",
        )  # Will this work even though guessed H100 name?
        created_pod_id = created_pod["id"]

    pods = runpod.get_pods()

    # Function to get pod details and extract IP and port (adapted from the script)
    def get_pod_ip_port(pid):
        data = runpod.get_pods()
        for item in data:
            if item["id"] == pid:
                if item["runtime"] is not None:
                    if (
                        "ports" in item["runtime"]
                        and item["runtime"]["ports"] is not None
                    ):
                        for port in item["runtime"]["ports"]:
                            if port["type"] == "tcp" and port["isIpPublic"]:
                                return port["ip"], port["publicPort"]

        return None, None  # no data for the pod ports yet. It is not set up.

    ip = None
    port = None
    max_retries = 200
    retry_delay_seconds = 15  # Increased delay between retries

    set_progress(
        task_id=task_id,
        progress=0.51,
        message=f"Waiting for pod {created_pod_id} to be ready...",
    )
    for retry_count in range(max_retries):
        ip, port = get_pod_ip_port(created_pod_id)
        if ip and port:
            print(f"Pod {created_pod_id} ready! IP: {ip}, Port: {port}")
            set_progress(
                task_id=task_id,
                progress=0.52,
                message=f"Pod {created_pod_id} ready. IP: {ip}, Port: {port}",
            )
            break
        print(
            f"Retry {retry_count + 1}/{max_retries}: Waiting {retry_delay_seconds}s for pod {created_pod_id}..."
        )
        time.sleep(retry_delay_seconds)

    if not ip or not port:
        print(
            f"Failed to get pod IP and port for pod {created_pod_id} after {max_retries} retries."
        )
        set_progress(
            task_id=task_id,
            progress=0.97,
            message=f"ERROR: Pod {created_pod_id} did not become ready. Training aborted.",
        )
        if created_pod_id and not pod_id:  # Only remove if this script created it
            print(f"Attempting to remove created pod {created_pod_id}...")
            # subprocess.run(f"runpodctl remove pod {created_pod_id}", shell=True, check=False) # Keep commented for now
        return

    try:
        set_progress(
            task_id=task_id, progress=0.983, message="Setting up credentials on pod..."
        )
        setup_commands = [
            f"/root/miniconda3/envs/py3.11/bin/huggingface-cli login --token {huggingface_token} --add-to-git-credential",
            f"/root/miniconda3/envs/py3.11/bin/wandb login {wandb_api_key}",
        ]
        for cmd in setup_commands:
            ssh_cmd = f"ssh {ssh_options} -p {port} root@{ip} '{cmd}'"
            print(f"Executing on pod: {ssh_cmd}")
            subprocess.run(
                ssh_cmd, shell=True, check=True, text=True, capture_output=False
            )
        print("Credentials set up on pod.")

        ## THIS BIT IS THE PRETRAINING
        if not skip_pretraining:
            set_progress(
                task_id=task_id,
                progress=0.51,
                message=f"Copying pretraining data to pod {created_pod_id}...",
            )
            scp_pretrain_cmd = f'scp {ssh_options} -P {port} -r "{pretraining_run_path}" root@{ip}:/workspace/axolotl/pretraining_run'
            print(f"Executing: {scp_pretrain_cmd}")
            subprocess.run(
                scp_pretrain_cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )
            print("Pretraining data copied.")
            set_progress(
                task_id=task_id, progress=0.51, message="Starting pretraining job..."
            )
            pretrain_axolotl_cmd = f"cd /workspace/axolotl/pretraining_run && /root/miniconda3/envs/py3.11/bin/accelerate launch -m axolotl.cli.train {pretrain_config_name}"
            ssh_pretrain_exec_cmd = (
                f"ssh {ssh_options} -p {port} root@{ip} '{pretrain_axolotl_cmd}'"
            )
            print(f"Executing pretraining on pod: {ssh_pretrain_exec_cmd}")
            subprocess.run(
                ssh_pretrain_exec_cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )  # Consider streaming output for long jobs
            print("Pretraining finished.")
        ## END PRETRAINING ##

        ## NOTE SFT
        if not skip_finetuning:
            set_progress(
                task_id=task_id,
                progress=0.75,
                message=f"Copying SFT data to pod {created_pod_id}...",
            )
            scp_sft_cmd = f'scp {ssh_options} -P {port} -r "{sft_run_path}" root@{ip}:/workspace/axolotl/sft_run'
            print(f"Executing: {scp_sft_cmd}")
            subprocess.run(
                scp_sft_cmd, shell=True, check=True, text=True, capture_output=False
            )
            print("SFT data copied.")

            set_progress(task_id=task_id, progress=0.75, message="Starting SFT job...")
            sft_axolotl_cmd = f"cd /workspace/axolotl/sft_run && /root/miniconda3/envs/py3.11/bin/accelerate launch -m axolotl.cli.train {sft_config_name}"
            ssh_sft_exec_cmd = (
                f"ssh {ssh_options} -p {port} root@{ip} '{sft_axolotl_cmd}'"
            )
            print(f"Executing SFT training on pod: {ssh_sft_exec_cmd}")
            subprocess.run(
                ssh_sft_exec_cmd,
                shell=True,
                check=True,
                text=True,
                capture_output=False,
            )  # Consider streaming output
            print("SFT training finished.")
            set_progress(
                task_id=task_id,
                progress=0.99,
                message="Training jobs completed successfully.",
            )
            ## END SFT

    except subprocess.CalledProcessError as e:
        error_message = f"An error occurred during training operations: {e.stderr if e.stderr else e.stdout}"
        print(error_message)
        set_progress(
            task_id=task_id,
            progress=(
                task_id.progress if task_id and hasattr(task_id, "progress") else 0.99
            ),
            message=f"ERROR: Training script failed. Details: {error_message}",
        )
        print(
            "\n\nIf this was an SSH thing, be sure that you have added your computer's public key (found at ~/.ssh/id_ed25519.pub) to your Runpod account's accepted public keys. Then delete any existing pods and rerun."
        )
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        set_progress(
            task_id=task_id,
            progress=(
                task_id.progress if task_id and hasattr(task_id, "progress") else 0.99
            ),
            message=f"ERROR: Unexpected issue in training script. Details: {error_message}",
        )
        print(
            "\n\nIf this was an SSH thing, be sure that you have added your computer's public key (found at ~/.ssh/id_ed25519.pub) to your Runpod account's accepted public keys. Then delete any existing pods and rerun."
        )

        raise
    finally:
        if created_pod_id: # Only remove if this script created it AND it's the one we were using
            print(
                f"Cleaning up - removing pod {created_pod_id}."
            )
            set_progress(
                task_id=task_id,
                progress=0.99,
                message=f"Training finished. Pod {created_pod_id} cleaned up.",
            )
            runpod.terminate_pod(pod_id=created_pod_id)
        else:
            print(
                f"Pod {created_pod_id} was pre-existing or an error occurred before its creation, not removing automatically."
            )
            set_progress(
                task_id=task_id,
                progress=0.99,
                message=f"Training finished or errored. Pod {created_pod_id} not removed by this script, likely because it was passed in earlier.",
            )
