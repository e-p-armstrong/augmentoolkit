import json
import random
import re
import traceback

from jinja2 import Template
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from generation.core_components import filter_chunks
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
    subset_text_list,
)
from generation.core_components.filter_chunks import (
    create_filter_chunks_step,
    filter_out_failed_items,
    filter_out_failed_items_dict,
)
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    setup_semaphore_and_engines,
    make_relative_to_self,
)
import os
from transformers import AutoTokenizer

from redis_config import set_progress


def masked_conversation_processor(text):
    """
    Extracts text from XML tags in a masked conversation and returns as dict.

    Args:
        text (str): Text containing XML tags with conversation components

    Returns:
        dict: Dictionary with extracted text for each conversation component
    """
    result = {}

    # Extract text from each tag
    for tag in [
        "initial_question",
        "flawed_answer",
        "followup_confirmation",
        "correct_answer",
    ]:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[tag] = match.group(1).strip()
        else:
            raise Exception("Critical tag not found! Retry!")

    return result


masked_conversation_creation_step = PipelineStep(
    output_file="correction_data",
    result_key="masked_conversation",
    output_processor=masked_conversation_processor,
    prompt_path="masked_conversation_generation",
    sampling_params={
        "max_tokens": 7000,
        "stop": [],
        "temperature": 0.7,
    },
    max_retries=3,
    details_key="masked_conversation_details",
)


# Modified function signature: template -> model_name_or_path
def create_axolotl_conversations_modelname(
    conversations, output_dir, model_name_or_path: str
):
    """
    Convert masked conversations into Axolotl-compatible format using a tokenizer's chat template.

    Args:
        conversations: List of dicts containing masked_conversation entries
        output_dir: Directory to save the output JSON file
        model_name_or_path: Hugging Face model identifier to load tokenizer and chat template

    Returns:
        List of dicts in Axolotl format with properly masked segments
    """
    axolotl_conversations = []

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Ensure a chat template exists
        if tokenizer.chat_template is None:
            # Fallback or default template if tokenizer doesn't have one
            # Using a common default like ChatML. Adjust if needed.
            print(
                f"Warning: Tokenizer for {model_name_or_path} has no chat_template. Using default ChatML-like template."
            )
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
            # Ensure EOS token is set if needed by template, though apply_chat_template handles it if in template
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "<|im_end|>"  # Example for ChatML
            # Ensure BOS token is set if needed by template
            # apply_chat_template should handle bos if it's part of the template string
            if (
                tokenizer.bos_token is None
                and "{% if add_bos_token %}" in tokenizer.chat_template
            ):  # Heuristic check
                tokenizer.bos_token = "<s>"  # Example default

    except Exception as e:
        print(f"Error loading tokenizer {model_name_or_path}: {e}")
        print("Cannot proceed with Axolotl conversation creation.")
        return  # Or raise an exception

    # Sort conversation keys to ensure deterministic order
    sorted_keys = sorted(conversations.keys())

    for key in sorted_keys:
        conv = conversations[key]
        try:
            masked_conv = conv["masked_conversation"]

            # Create message sequence
            messages = [
                {"role": "user", "content": masked_conv["initial_question"]},
                {"role": "assistant", "content": masked_conv["flawed_answer"]},
                {"role": "user", "content": masked_conv["followup_confirmation"]},
                {"role": "assistant", "content": masked_conv["correct_answer"]},
            ]

            # Apply the chat template using the tokenizer
            # tokenize=False to get the string output for splitting
            # add_generation_prompt=False as Axolotl expects the full conversation including the final assistant turn
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Split into segments and assign masking
            segments = []

            # Find the flawed answer in rendered text
            # NOTE: This assumes the exact flawed_answer string appears in the rendered output.
            # This might break if the chat template adds tokens within or around content.
            rendered_parts = rendered.split(masked_conv["flawed_answer"])

            if len(rendered_parts) != 2:
                print(
                    f"Warning: Could not properly split on flawed answer for key {key}. Skipping."
                )
                print(f"Flawed Answer: {masked_conv['flawed_answer']}")
                print(f"Rendered Text: {rendered}")
                # Optionally save problematic cases for debugging
                # with open(os.path.join(output_dir, f"failed_split_{key}.txt"), "w") as err_f:
                #     err_f.write(f"Flawed Answer:\n{masked_conv['flawed_answer']}\n\nRendered Text:\n{rendered}")
                continue  # Skip this conversation

            # Add pre-flawed answer segment
            if rendered_parts[0]:
                segments.append({"label": True, "text": rendered_parts[0]})

            # Add flawed answer segment
            segments.append({"label": False, "text": masked_conv["flawed_answer"]})

            # Add post-flawed answer segment
            if rendered_parts[1]:
                segments.append({"label": True, "text": rendered_parts[1]})
        except Exception as e:  # Catch broader exceptions during processing
            print(f"Error processing conversation key {key}: {e}")
            traceback.print_exc()
            continue

        axolotl_conversations.append({"segments": segments})

    output_path = os.path.join(output_dir, "axolotl_correction_conversations.json")
    # Check if list is empty before writing
    if not axolotl_conversations:
        print("Warning: No Axolotl conversations were successfully generated.")
        return

    with open(output_path, "w") as f:
        json.dump(axolotl_conversations, f, indent=2)
    print(
        f"Successfully wrote {len(axolotl_conversations)} Axolotl conversations to {output_path}"
    )


def create_axolotl_conversations(conversations, output_dir, template):
    """
    Convert masked conversations into Axolotl-compatible format using a prompt template.

    Args:
        conversations: List of dicts containing masked_conversation entries
        prompt_template: Jinja2 template string for formatting messages

    Returns:
        List of dicts in Axolotl format with properly masked segments
    """
    axolotl_conversations = []

    # Sort conversation keys to ensure deterministic order
    sorted_keys = sorted(conversations.keys())

    for key in sorted_keys:
        conv = conversations[key]
        try:
            masked_conv = conv["masked_conversation"]

            # Create message sequence
            messages = [
                {"role": "user", "content": masked_conv["initial_question"]},
                {"role": "assistant", "content": masked_conv["flawed_answer"]},
                {"role": "user", "content": masked_conv["followup_confirmation"]},
                {"role": "assistant", "content": masked_conv["correct_answer"]},
            ]

            # Fix: Use a different variable name for the Template object
            jinja_template = Template(
                template
            )  # TODO maybe use a template like this in the damned main factual config, hm?
            rendered = jinja_template.render(messages=messages, bos_token="<s>")

            # Split into segments and assign masking
            segments = []

            # Find the flawed answer in rendered text
            rendered_parts = rendered.split(masked_conv["flawed_answer"])

            if len(rendered_parts) != 2:
                raise ValueError("Could not properly split on flawed answer")

            # Add pre-flawed answer segment
            if rendered_parts[0]:
                segments.append({"label": True, "text": rendered_parts[0]})

            # Add flawed answer segment
            segments.append({"label": False, "text": masked_conv["flawed_answer"]})

            # Add post-flawed answer segment
            if rendered_parts[1]:
                segments.append({"label": True, "text": rendered_parts[1]})
        except:
            traceback.print_exc()
            continue

        axolotl_conversations.append({"segments": segments})

    output_path = os.path.join(output_dir, "axolotl_correction_conversations.json")
    with open(output_path, "w") as f:
        json.dump(axolotl_conversations, f, indent=2)


async def correction_pipeline(  # requirement: the node must have the same argument names as the fields in the config
    use_subset: bool,
    subset_size: int,
    chunk_size: int,
    input_dir: str,
    concurrency_limit: int,
    small_model: str,
    small_api_key: str,
    small_base_url: str,
    small_mode: str,
    large_model: str,
    large_api_key: str,
    large_base_url: str,
    large_mode: str,
    output_dir: str,
    default_prompts: str,
    prompts: str,
    completion_mode: bool,
    use_stop: bool,
    prompt_template: str,
    do_meta_datagen: bool = False,
    meta_datagen_keys: list[str] = [],
    meta_datagen_extras: list[str] = [],
    read_files_manually: bool = True,
    text_chunks_passed_in: list[str] = [],
    cost_per_million_small_input: float = 0.0,
    cost_per_million_small_output: float = 0.0,
    cost_per_million_large_input: float = 0.0,
    cost_per_million_large_output: float = 0.0,
    chunking_output_dir=None,
    task_id=None,
    seed=1048596,
    **kwargs,  # All nodes MUST have **kwargs
):
    filter_chunks_step = create_filter_chunks_step(output_file="correction_data")
    # Check if kwargs is not empty and print all keys and values if present
    if kwargs:
        print("Additional arguments provided:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    default_prompts = make_relative_to_self(default_prompts)
    prompts = make_relative_to_self(prompts)

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
                ),
            ],
            engine_output_observers=[
                create_output_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                ),
                create_log_observer(output_dir),
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
                create_output_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                )
            ],
        )
    )

    ### end setup node

    ## Reading node
    set_progress(
        task_id, progress=0.1, message="Pipeline starting; reading and chunking files"
    )

    if read_files_manually:
        sentence_chunks = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=use_subset,
            subset_size=subset_size,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
            seed=seed,
        )
    else:
        sentence_chunks = chunk_text_list(
            text_chunks_passed_in,
            chunk_size=chunk_size,
            keep_folder_structure=True,
            input_dir=input_dir,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
        )
        if use_subset:
            sentence_chunks = subset_text_list(
                sentence_chunks, subset_size=subset_size, seed=seed
            )

    total_tokens = count_total_tokens(sentence_chunks)

    sentence_hashed_dict = hash_input_list(
        input_list=sentence_chunks, key_to_hash_with="text"
    )

    # if use_subset: # good example of what it will look like for cross-node argument passing. Also this stuff is not stored as props but arguments to the call of the class BECAUSE otherwise it's terribly inconvinient to do some typical things.
    #     subset_size = min(subset_size, len(sentence_chunks))
    #     sentence_chunks = random.sample(sentence_chunks, susubset_sizebset_size)
    ### end reading node

    set_progress(
        task_id,
        progress=0.2,
        message="Files read and chunked; proceeding with generation",
    )

    await filter_chunks_step.execute_pipeline(
        input_dict=sentence_hashed_dict,
        engine_wrapper=engine_wrapper,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
        task_id=task_id,
    )
    # Print the value of the first key of the sentence_hashed_dict
    if sentence_hashed_dict:
        first_key = next(iter(sentence_hashed_dict))
        print("First key in sentence_hashed_dict:")
        print(first_key)
        print("Value of first key:")
        print(sentence_hashed_dict[first_key])
    else:
        print("sentence_hashed_dict is empty")
    # the end of the generation node

    # How will this work with recitation?
    # different prompt set
    # should be simple enough to add
    # Print the number of items in sentence_hashed_dict before filtering
    print(
        f"Number of items in sentence_hashed_dict before filtering: {len(sentence_hashed_dict)}"
    )

    filter_out_failed_items_dict(sentence_hashed_dict)
    print(
        f"Number of items in sentence_hashed_dict after filtering: {len(sentence_hashed_dict)}"
    )

    await masked_conversation_creation_step.execute_pipeline(
        input_dict=sentence_hashed_dict,
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
        task_id, progress=0.9, message="Generations done; saving final dataset"
    )  # even though the distance between here and the very end is going to be very slight, we establish a difference because in case it errors during the final saving, we want the progress to express where it failed

    create_axolotl_conversations(
        sentence_hashed_dict, output_dir=output_dir, template=prompt_template
    )

    # the fact of the matter is, we avoid adding things that fail validation because they will have been filtered out by things like filter_out_failed_items_dict beforehand. So, it all works.
    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[sentence_hashed_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete")
