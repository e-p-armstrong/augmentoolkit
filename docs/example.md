# Starting Point (Build Your Own Pipeline!)

This pipeline (`generation/example_pipeline/example.py`) serves as a **template, tutorial, and boilerplate** for creating your own custom data generation pipelines within Augmentoolkit. While Augmentoolkit provides several powerful pipelines out-of-the-box (like factual recall, representation variation, etc.), its true strength lies in its extensibility. This example demonstrates the fundamental building blocks and common patterns used in most pipelines, making it easy to get started with your own unique data generation ideas.

The goal here is not necessarily to run this specific pipeline (though it does generate poems inspired by input text), but to understand its structure, use it as a foundation, and reuse its components.

## Anatomy of an Augmentoolkit Pipeline (using `example.py`)

Let's break down the key parts of the `example_pipeline` function and its associated files. This will serve as a sort of written explainer/walkthrough of Augmentoolkit pipelines. If you're an experienced programmer it will probably be easier looking at the code (the code is extensively annotated with comments) but for those who appreciate a more traditional doc/readme, this walkthrough-esq thing is provided.

### 1. Pipeline Function Signature

```python
async def example_pipeline(
    # ... common args like use_subset, subset_size, chunk_size ...
    input_dir: str,
    concurrency_limit: int,
    # ... api details (small/large model, key, base_url, mode) ...
    output_dir: str,
    default_prompts: str,
    prompts: str,
    # ... system settings like completion_mode, use_stop ...
    example_heading, # Custom arg from config
    key3, # Custom arg from config
    # ... optional common args like do_meta_datagen, read_files_manually ...
    cost_per_million_small_input: float = 0.0,
    # ... other cost args ...
    chunking_output_dir=None,
    task_id=None,
    seed=11037,
    **kwargs # MUST include for forward compatibility
):
```

*   **`async def`:** Pipelines are asynchronous to handle concurrent API calls efficiently. They *can* be synchronous but there is rarely a reason to do this.
*   **Arguments Match Config:** Pipeline function arguments generally correspond directly to keys in the pipeline's `config.yaml` file. The `run_augmentoolkit.py` script flattens the YAML config (unless sections are specified in `no_flatten`) and passes the keys/values as keyword arguments to the pipeline function.
*   **Common Arguments:** Many arguments are standard across pipelines:
    *   API details (`small_model`, `large_model`, keys, URLs, modes).
    *   Paths (`input_dir`, `output_dir`, `prompts`, `default_prompts`).
    *   System settings (`chunk_size`, `concurrency_limit`, `use_subset`, `subset_size`).
    *   Optional features (`do_meta_datagen`, `read_files_manually`, cost estimation args).
*   **Custom Arguments:** You can add any specific arguments needed for your pipeline (like `example_heading` and `key3` here).
*   **`task_id`:** Used by the interface for progress tracking via `set_progress`.
*   **`seed`:** For reproducibility in steps involving randomness (like subsetting).
*   **`**kwargs`:** **Essential.** This captures any extra arguments passed from the config, ensuring forward compatibility if new common options are added to Augmentoolkit later.

### 2. Initial Setup

```python
# Check for unused kwargs
if kwargs:
    print("Additional arguments provided:")
    # ... print kwargs ...

default_prompts = make_relative_to_self(default_prompts)
prompts = make_relative_to_self(prompts)

small_token_counter = { ... }
large_token_counter = { ... }

run_task_with_limit, engine_wrapper, engine_wrapper_large, _ = setup_semaphore_and_engines(
    # ... api details, concurrency ...
    engine_input_observers=[...],
    engine_output_observers=[...]
)
```

*   **Check `kwargs`:** A good practice to alert the user if unexpected arguments were passed.
*   **`make_relative_to_self`:** Crucial for ensuring prompt paths are relative to the pipeline's directory, not the project root.
*   **Token Counters:** Initialization for cost estimation (optional but recommended).
*   **`setup_semaphore_and_engines`:** A helper function that creates:
    *   `run_task_with_limit`: An asyncio execution wrapper that respects the `concurrency_limit`.
    *   `engine_wrapper`, `engine_wrapper_large`: Standard `EngineWrapper` instances for making LLM API calls, configured with API details and observers (like cost trackers and loggers).

### 3. Progress Tracking

```python
set_progress(task_id, progress=0.1, message="...")
# ... later ...
set_progress(task_id, progress=0.2, message="...")
# ... etc ...
set_progress(task_id, progress=1.0, message="Pipeline Complete")
```

*   **`set_progress(task_id, progress, message)`:** Updates the pipeline's status in the UI (if used) and the backend database. Call this at logical milestones in your pipeline. `progress` is a float 0.0-1.0. `set_progress` has no effect if the pipeline is run in cli mode.

### 4. Input Data Handling

```python
if read_files_manually:
    sentence_chunks = read_and_chunk_text(...) # Reads from input_dir
else:
    sentence_chunks = chunk_text_list(text_chunks_passed_in, ...)
    if use_subset:
        sentence_chunks = subset_text_list(...)

sentence_hashed_dict = hash_input_list(sentence_chunks, ...)
total_tokens = count_total_tokens(sentence_chunks)
```

*   **Reading/Chunking:** Uses helpers like `read_and_chunk_text` (for reading files) or `chunk_text_list` (for processing passed-in data) to break down input into manageable chunks based on `chunk_size`. Handles subsetting via `use_subset` and `subset_size`.
*   **Caching:** `read_and_chunk_text` automatically caches chunked data (by default in the `output_dir` or optionally in `chunking_output_dir`) to speed up subsequent runs on the same input.
*   **Hashing:** `hash_input_list` converts the list of chunks into a dictionary where keys are deterministic hashes of the content. This is crucial for the stateful, resumable nature of `PipelineStep` execution.
*   **Token Counting:** `count_total_tokens` helps with cost estimation.

### 5. Core Logic: `PipelineStep`

This is the heart of most pipelines. It encapsulates a single LLM interaction step. This toy example pipeline just has one pipeline step, but most pipelines have multiple. **Sequential pipeline steps ought to share the same `output_file` for storage efficiency.**

```python
def write_poetry_processor(output):
    # ... processes LLM output ...
    return poem_content[-1] if poem_content else None

def validate_poem(output, input_data):
    # ... returns True if output is valid, False otherwise ...
    return True

def process_poem_input(input):
    # ... modifies input dict before LLM call ...
    input["text"] = input["text"].upper()
    return input

write_poem_step = PipelineStep(
    prompt_path="write_poem", # Path to prompt file (relative to pipeline)
    output_processor=write_poetry_processor,
    validation_function=validate_poem,
    input_processor=process_poem_input,
    sampling_params={...}, # LLM sampling settings
    output_file="demo_file", # Intermediate data filename
    result_key="poetry", # Key to save processed result under
    details_key="poetry_details", # Key to save raw LLM output under (for meta-datagen)
    max_retries=3,
    additional_kwarg_example="..." # Extra args for prompt interpolation
)
```

*   **Define Processors/Validators:** Create Python functions to:
    *   `output_processor`: Extract the desired information from the raw LLM response string.
    *   `validation_function`: Check if the processed output meets specific criteria (e.g., format, content). Returns `False` to trigger a retry.
    *   `input_processor` (optional): Modify the input dictionary *before* it's formatted into the prompt.
*   **Instantiate `PipelineStep`:** Configure the step with:
    *   `prompt_path`: Name of the YAML prompt file (e.g., `write_poem.yaml`).
    *   Processors and validators.
    *   `sampling_params`: Dictionary of LLM generation settings.
    *   `output_file`: Base filename for saving intermediate state. Should generally be the same across steps in a pipeline for efficiency.
    *   `result_key`: The dictionary key where the *processed* output will be stored.
    *   `details_key`: The dictionary key where the *raw* LLM output will be stored (useful for debugging and meta-datagen).
    *   `max_retries`: How many times to retry if validation fails or an error occurs.
    *   Any other keyword arguments provided here are available for interpolation in the prompt file (e.g., `{additional_kwarg_example}`).

### 6. Executing the Step

```python
await write_poem_step.execute_pipeline(
    input_dict=sentence_hashed_dict,
    engine_wrapper=engine_wrapper, # Which LLM engine to use
    rtwl=run_task_with_limit, # The concurrency wrapper
    default_prompt_folder=default_prompts,
    prompt_folder=prompts, # Override prompt folder
    output_dir=output_dir,
    completion_mode=completion_mode,
    use_stop=use_stop,
    include_details=do_meta_datagen, # Save raw output if True
    task_id=task_id, # Pass for progress updates within the step
    additional_arg="..." # Extra args for prompt interpolation
)
```

*   **`execute_pipeline`:** This asynchronous method orchestrates the execution for all items in the `input_dict`.
*   **Functionality:** It handles loading previous state (if `output_file` exists), applying the `input_processor`, formatting the prompt using data from the input dict item and extra kwargs, making the API call via the `engine_wrapper`, applying the `output_processor` and `validation_function`, handling retries, and saving the results (including intermediate states) back to the `output_file`.
*   **Arguments:** Key arguments include the `input_dict`, the `engine_wrapper` to use, the `rtwl` semaphore, prompt folder paths, and flags like `completion_mode` and `use_stop`.

### 7. Final Output Formatting

```python
sharegpt_format_items = []
for index, item in sentence_hashed_dict.items():
    if "poetry" in item:
        sharegpt_item = {
            "conversations": [
                {"from": "human", "value": f"... {item['text']} ..."},
                {"from": "gpt", "value": item["poetry"]}
            ]
        }
        sharegpt_format_items.append(sharegpt_item)

sharegpt_output_path = os.path.join(output_dir, "sharegpt_format.jsonl")
with open(sharegpt_output_path, "w") as f:
    for sharegpt_item in sharegpt_format_items:
        f.write(json.dumps(sharegpt_item) + "\n")
```

*   After all `PipelineStep`s run, the `sentence_hashed_dict` contains all the generated data.
*   This final step typically involves iterating through the dictionary, selecting the relevant generated keys (like `"poetry"` here), and formatting them into a standard training format (like ShareGPT JSONL).
*   The formatted data is then saved to a final output file.

### 8. Meta Dataset Generation (Optional)

```python
if do_meta_datagen:
    create_meta_dataset(
        data_dicts=[sentence_hashed_dict],
        meta_datagen_keys=meta_datagen_keys, # e.g., ["poetry_details"]
        meta_datagen_extras=meta_datagen_extras,
        input_processors=[],
        output_dir=os.path.join(output_dir, "meta_datagen")
    )
```

*   If enabled via the config and `do_meta_datagen=True` is passed, this step uses the `create_meta_dataset` helper.
*   It gathers the raw LLM outputs stored under the `details_key`s specified in `meta_datagen_keys` and formats them (along with corresponding inputs) into trainable data.
*   `meta_datagen_extras` allows defining custom prompt templates (paths specified in the config list) to create additional training examples from the final data dictionary, potentially skipping intermediate steps.

## Configuration (`config.yaml`)

The `config.yaml` file provides the arguments for the pipeline function. Key things demonstrated:

*   Standard sections (`api`, `path`, `system`, `cost`, `meta_datagen`).
*   Custom arguments (`example_heading`, `key3`). Note how `key3` is flattened from `example_heading_2`, while `example_heading` remains a dictionary because it's listed under `no_flatten`.
*   `!!PLACEHOLDER!!`: Used to mark fields that *must* be filled in by the user before running.

## Prompt File (`prompts/write_poem.yaml`)

*   Demonstrates the YAML structure for defining chat prompts.
*   Uses standard OpenAI chat format (`role`, `content`).
*   Shows how to include few-shot examples (user/assistant pairs) to guide the model.
*   Illustrates f-string style interpolation (`{text}`, `{metadata}`, `{additional_arg}`) using keys from the input data dictionary and extra kwargs passed to the pipeline step.

## How to Use as a Starting Point

1.  **Copy the `generation/example_pipeline` folder** and rename it to something descriptive for your new pipeline.
2.  **Modify `config.yaml`:** Adjust API keys, paths, and system settings. Add any new configuration parameters your pipeline needs.
3.  **Edit `<your_pipeline_name>.py`:**
    *   Change the function name (`example_pipeline` -> `your_pipeline_name`).
    *   Update the function signature to accept your new config parameters.
    *   Define your `PipelineStep`(s): Create new prompt files (in YAML), write corresponding output/validation/input processors, and configure the steps.
    *   Adjust the `execute_pipeline` calls to use the correct engine wrappers and pass necessary arguments.
    *   Modify the final output formatting section to save the data your pipeline generates in the desired format for your use case.
    *   Update `set_progress` messages.
4.  **Create Prompt Files:** Write your prompts in the `prompts/` subdirectory.
5.  **Test:** Run your pipeline using `run_augmentoolkit.py` or the interface.

By following the patterns and using the abstractions demonstrated here, you can leverage Augmentoolkit's framework (concurrency, state management, logging, cost estimation, meta-datagen) to rapidly build powerful custom data generation pipelines. 

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.