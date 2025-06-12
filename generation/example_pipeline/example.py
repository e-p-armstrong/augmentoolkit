"""This pipeline is mostly boilerplate and exists to walk you through what the fundamental components of an Augmentoolkit pipeline are.

Augmentoolkit pipelines are fundamentally just Python functions that usually use a few shared abstractions, and have some typical arguments.

This pipeline demonstrates how to make a basic Augmentoolkit pipeline. It is useful context to both humans and AI alike. Augmentoolkit pipelines are deliberately close to normal Python code so that it is easy for AI to write them.
"""

import json
import re
import traceback
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.extract_first_words import remove_think_tags
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
    subset_text_list,
)
from generation.core_components.filter_chunks import create_filter_chunks_step
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    setup_semaphore_and_engines,
    make_relative_to_self,
)

import os
from transformers import AutoTokenizer

from redis_config import set_progress


async def example_pipeline(  # NOTE requirement: the pipeline must have the same argument names as the fields in the config.
    use_subset: bool,  # a common parameter that tells us whether to use a subset of the total input (good for testing, dev, cost estimation, and very large input sets)
    subset_size: int,  # a common parameter indicating the number of items from the total read input items to take.
    chunk_size: int,
    input_dir: str,
    concurrency_limit: int,  # how many concurrent requests you want to have active at once. Good for avoiding rate limits.
    small_model: str,
    small_api_key: str,
    small_base_url: str,
    small_mode: str,
    large_model: str,  # NOTE convention: pipeline arguments should be positional, not keyword arguments, when DEFINED (unless they are things like a task ID or seed or do_meta_datagen which is unlikely to appear in the config but we want to take it as an arg and have backward-compatibility with older configs anyway). The reason we use positionals is because that way Python catches us if we miss a critical argument. However for safety/reliability since there are so many args, whenever CALLING pipelines, we use keyword arguments for everything.
    large_api_key: str,
    large_base_url: str,
    large_mode: str,
    output_dir: str,
    default_prompts: str,
    prompts: str,
    completion_mode: bool,
    use_stop: bool,  # Not all APIs (take OpenAI for instance) support more than 4 stop tokens. use_stop is passed to pipeline executions and if it is False, then the number of stop tokens is truncated to 4.
    example_heading,
    key3,
    do_meta_datagen: bool = False,
    meta_datagen_keys: list[str] = [],
    meta_datagen_extras: list[str] = [],
    read_files_manually: bool = True,
    text_chunks_passed_in: list[str] = [],
    cost_per_million_small_input: float = 0.0,
    cost_per_million_small_output: float = 0.0,
    cost_per_million_large_input: float = 0.0,
    cost_per_million_large_output: float = 0.0,
    chunking_output_dir=None,  # Augmentoolkit caches the results of file reading and chunking. If you want your reading/chunking cache dir to be different than your output dir, you can add an option for that.
    task_id=None,  # task_id is a special argument used to help set the progress throughout the pipeline's execution. It's optional for you to add, but if you want a progress bar to show up properly when your pipeline is used with the interface, then you have to add this as well as a few set_progress calls.
    seed=11037,
    **kwargs,  # All pipelines MUST have **kwargs to ensure forward compatibility with new common arguments.
):
    # Check if kwargs is not empty and print all keys and values if present
    if (
        kwargs
    ):  # NOTE standard anti-footgun measure reminding people of excess args they are passing.
        print("Additional arguments provided:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    print("Demonstrating example heading and flattening vs no_flatten")
    print("Example heading:")
    print(example_heading)
    print("key3")
    print(key3)

    # NOTE the make_relative_to_self calls on the prompts is key so that the prompt directories are not looked for relative to the root of the whole project.
    # Prompt files are by convention co-located with their pipeline in the same folder. So we need to adjust the paths to make sure they reflect this.
    default_prompts = make_relative_to_self(default_prompts)
    prompts = make_relative_to_self(prompts)

    # These two things are used for pipeline cost estimation. This is their initialization.
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
        setup_semaphore_and_engines(  # in contrast to previous versions of Augmentoolkit, there is now a function which sets up your standard engine wrappers and semaphore. Hooray for abstraction saving us time!
            concurrency_limit,
            small_model,
            small_api_key,
            small_base_url,
            small_mode,
            large_model,
            large_api_key,
            large_base_url,
            large_mode,  # The most common pattern is for pipelines to have a large powerful model and a small less powerful but cheaper model, so that is what the semaphore and engines setup function creates. If you want more engine wrappers, you can make one with the EngineWrapper() class in engine_wrapper_class.py. pass the model, api key, base url, and mode. If you want just one engine wrapper, you can use this functoin but pass the same settings for both the small and large engine wrapper and assign the large engine wrapper to _
            engine_input_observers=[  # input observers are called on all inputs JUST before they get sent to the LLM. They are useful for things like cost estimation or intermediate output logging for debugging purposes. create_input_token_counter, like most observers, is a higher-order function and returns a function.
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
                create_output_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                ),
                create_log_observer(output_dir),
            ],  # output observers are called on the LLM API responses right after they are received
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
                ),
                create_log_observer(output_dir),
            ],
        )
    )

    set_progress(
        task_id, progress=0.1, message="Pipeline starting; reading and chunking files"
    )  # set_progress controls the progress bar in the interface, and is used to update the redis database that runs when you're not using the CLI about where this pipeline execution is at. task_id is the id of the task, passed in as a kwarg; progress is a float between 0 and 1 indicating % completion (can be as fine or as rough an estimate as you want); message is a descriptive message to be displayed. Just calling it like this is all that is required for compatibility with the interface, and set_progress is the only way in which you might ever have to think of the interface when coding a pipeline.

    if (
        read_files_manually
    ):  # Pipelines can be called as functions. Sometimes when they are being used as functions, it is preferred that already-read-in text chunks be passed in, rather than having the pipeline read and chunk the files itself (take the complete factual datagen pipeline for instance; there, the same common set of read in files is used for most steps). read_files_manually is the arg that is False if we want to read in things from the input dir (in which case we use the read_and_chunk_text helper function); and if this arg is True, then we just chunk the passed-in text_chunks_passed_in and subset it afterwards using the subset_text_list helper. This pattern is more or less common across pipelines, though some use more outdated ways of structuring it.
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

    # We chunk the text to turn large documents into workable pieces that can be used to create data.

    sentence_hashed_dict = hash_input_list(
        input_list=sentence_chunks, key_to_hash_with="text"
    )  # once the text is chunked into a list of chunks, it must be converted to a dict for usage with the Augmentoolkit Pipeline abstractions.
    # The reason we use a dict instead of a list is to avoid difficult completion order bugs and annoyances.
    # The typical pattern is that we start with a list when we chunk our input text; then we convert it to a dict for processing; then we convert it back to a list for saving.

    total_tokens = count_total_tokens(
        sentence_chunks
    )  # counting the total number of tokens is important for later cost estimation.

    set_progress(
        task_id,
        progress=0.2,
        message="Files read and chunked; proceeding with generation",
    )  # another set progress call example. It replaces the previous one completely. Pipeline is now 20% done.

    ## PipelineStep example
    # Here we use a PipelineStep, the core of most Augmentoolkit pipelines.
    # First we create an output processor to extract the part of the LLM output we want.
    # To check the prompt used here, look at ./prompts/write_poem.yaml in the example folder.

    def write_poetry_processor(output):
        # Remove think tags from the output. This is useful for working with reasoning models and has no effect on the outputs of non-reasoning models.
        cleaned_output = remove_think_tags(output)

        if cleaned_output.count("<poem>") > 1:
            raise Exception(
                "Multiple <poem> tags found in the output."
            )  # anything that raises in an output processor will be caught by the pipeline step and will cause a retry. Retries are capped at max_retries number of retries.

        # Use regex to extract content between the last pair of <poem> and </poem> tags. We use the last pair for extra reliability.
        poem_content = re.findall(r"<poem>(.*?)</poem>", cleaned_output, re.DOTALL)
        return poem_content[-1] if poem_content else None

    def validate_poem(output, input_data):
        # validation functions take the processed output. They cause a retry if they return False, and allow the output to save if they return true. Validation can therefore be either handled by raising in an output processor or by returning false here. Which you use depends on what makes more sense -- to validate before or after the output is processed into a more workable, narrow format. Do you lose information you need, or does it become easier to tell if something's broken?

        # the input data is provided as the second argument in case we want to check something in the output against something in the input data that produced it.

        # Here we use a random arbitrary criterion. Does the poem contain the letter "e"?

        if "e" in output:
            return {"result": True, "message": "Success"}
        else:
            return {
                "result": True,
                "message": "Due to hyper-arbitrary (and demonstrative) purposes, all poems must contain the letter 'e' at least once. Validation failed.",
            }
        # Note that even if something fails validation, its intermediate outputs will still be logged by the output observer for debugging purposes (if there is an output observer).

    def process_poem_input(input):
        # input processors modify the input object before it reaches the LLM API call. You're working with the entire input object dictionary here, not just the text, and you can mutate it in a way that will affect how it is saved, so be careful but also be aware that this is useful in a lot of cases.
        input["text"] = input["text"].upper()
        return input

    write_poem_step = PipelineStep(  # Usually pipeline steps are defined outside of a Pipeline's actual function (such as further up the file or in a helpers file) to avoid clutter. I am defining this here so that you learn things in an intuitive order — they are defined as the code needs them.
        prompt_path="write_poem",
        output_processor=write_poetry_processor,  # see how the processors we defined are now called
        validation_function=validate_poem,
        input_processor=process_poem_input,
        sampling_params={
            "max_tokens": 2000,  # max output tokens
            "stop": [
                "\n\n\n\n\n",
            ],
            "temperature": 0.8,  # same as with openai. Any nonstandard args here get passed in as extra body parameters to the API call.
            "top_p": 0.9,
        },
        output_file="demo_file",  # the name of the .json file to which all outputs of this step are saved. Unless absolutely necessary, you should keep the same output file throughout all the steps in your pipeline -- this is far more space efficient in terms of disk space. New keys get added without replacing what was there before.
        result_key="poetry",
        details_key="poetry_details",  # used for meta dataset generation (optional), see the end of this file
        max_retries=3,
        additional_kwarg_example="I will get put in the prompt if the text {additional_kwarg_example} appears in there at all!",
        # sometimes pipeline steps have additional kwrags. Additional kwargs are sometimes deprecated things that have not been deleted since previous versions. Other times they're taking advantage of the fact that any kwargs left over in the pipeline step definition (or the execute pipeline call) get put in the prompt if something of the right name exists. This allows for even further prompt control.
    )

    # NOTE See the Abstractions Primer for some context/explanations of the most useful helpers in Augmentoolkit.
    # See the New Pipeline Primer for a brief text-based overview on when to make a new pipeline as well as some things to keep in mind. It is supporting material for this example.

    await write_poem_step.execute_pipeline(  # execute pipeline calls the pipeline step's run method on each sub-dict inside the input dict. It goes through the input processor, then all the keys and values that appear in the sub-dict are formatted into {curly bracket things} of the same name inside the prompt if they exist (unused things are just passed over), and then this prompt is passed to the LLM API via the engine wrapper. The output is processed by the output processor and validated by the validation function. The input dict is loaded from the preexisting filename (if it exists) at the start and saved at the end (or when interrupted). Concurrency is managed with the semaphore we created earlier (run task with limit). .execute_pipeline() is by far the most useful and powerful abstraction in Augmentoolkit and is worth getting to know.
        input_dict=sentence_hashed_dict,
        engine_wrapper=engine_wrapper,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,  # the difference between that which is passed as an argument, and that which is defined on the pipeline step class, is that if it varies run by run due to config settings, it is usually passed as an argument. There are some exceptions by necessity of the code.
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
        task_id=task_id,
        additional_arg="I will also get put in the prompt if {additional_arg} appears in there!",
    )

    # Print the value of the first key of the sentence_hashed_dict so you can see how it has changed
    if sentence_hashed_dict:
        first_key = next(iter(sentence_hashed_dict))
        print("First key in sentence_hashed_dict:")
        print(first_key)
        print("Value of first key:")
        print(sentence_hashed_dict[first_key])
    else:
        print("sentence_hashed_dict is empty")

    set_progress(
        task_id, progress=0.9, message="Generations done; saving final dataset"
    )  # even though the distance between here and the very end is going to be very slight, we establish a difference because in case it errors during the final saving, we want the progress to express where it failed

    # Final dataset saving just involves taking our big intermediate dict, and saving it in a format that we can train an LLM on. Probably ShareGPT.
    # ShareGPT is a list of items, so we convert our dict back to a list.
    # Literally all we have to do is save our items to a .jsonl in the output dir. AI is capable of writing most such output functions or code as it is very standard.
    sharegpt_format_items = []

    for index, item in sentence_hashed_dict.items():
        if "poetry" in item:
            sharegpt_item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": f"please write a poem inspired by this text {item['text']}",
                    },  # NOTE  you always want different inputs across your data. That's why I included the item's text even though doing so basically made this crude offline distillation. The LLM is learning a relationship between inputs and outputs, so if the same input (e.g., "write a poem" with no inspiration text) produces a bunch of different outputs then you will mess with the model's mind. Don't do that. Make your inputs different from each other and related to the output.
                    {"from": "gpt", "value": item["poetry"]},
                ]
            }
            sharegpt_format_items.append(sharegpt_item)

    sharegpt_output_path = os.path.join(output_dir, "sharegpt_format.jsonl")
    with open(sharegpt_output_path, "w") as f:
        for sharegpt_item in sharegpt_format_items:
            f.write(json.dumps(sharegpt_item) + "\n")

    # meta dataset generation is an option if we want to train models to run the pipelines we specifically build. It makes the data files you save take up a lot more space, but also creates valuable data out of the intermediate execution steps in your pipeline. All you need to do to support meta dataset generation: add the do_meta_datagen: str, meta_datagen_keys: list[str], and meta_datagen_extras: list[str] to your pipeline's arguments, and call the create_meta_dataset function at the end, after all your pipeline steps have executed.
    # Pass in any and all data dicts, as well as:
    # the keys (list of details keys, e.g., here it would be "poetry_details")
    # and your extras (often an empty list, but this is a list of prompt file paths which get formatted like a prompt for a pipeline step, except instead of doing inference we save them as trainable data).
    # (The reason extras exists is: what if you wanted to train am odel to skip all the intermediate steps of a pipeline? To just generate the final output given the initial raw input? Then you could have a prompt that is basically user: {text} AI: {final_output} and specify that as an extra, and you'd get useful training data. If you have intermediate steps because off-the-shelf LLMs lack the capabilities to just zero-shot the datagen task, then extras might come in really handy when you make a custom model)
    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[sentence_hashed_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete")
