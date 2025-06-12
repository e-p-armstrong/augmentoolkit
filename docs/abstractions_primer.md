# Abstractions Primer

This page details and explains the main abstractions (functions and classes, essentially) that Augmentoolkit provides to make developing dataset generation pipelines easier and better. The goal is to empower you to make your own pipelines so that you can create custom models that succeed at your unique usecases. Many people have already built custom prompt sets or have forked the project to make their own customizations. Now, with the 3.0 major update, there has never been a better time to start making custom AI.

To start using any of these abstractions you will of course have to make a new pipeline. As mentioned in the [starting point guide](example.md) and the [new pipeline primer](pipeline_primer.md), Augmentoolkit pipelines are just Python functions following a few optional conventions. They have their arguments supplied via simple config files and they read their prompts in from yaml files as well. So all you need to make a compliant Augmentoolkit pipeline is a python file, a config.yaml file, and a prompt folder, all in the same folder.

`example.py` is a great, annotated template and piece of boilerplate to start from. Its [starting point guide](example.md) also gives a great rundown on all the key parts of that pipeline, most of which will appear in basically any Augmentoolkit pipeline. This primer is more about being a reference *as* you build your new pipeline, helping you use the helpful Augmentoolkit abstractions to make your life easier, and preventing you from getting stuck on expected function inputs and output structures.

## `PipelineStep`

(`augmentoolkit/generation_functions/pipeline_step_class.py`)

This is the **fundamental building block** for most Augmentoolkit pipelines that involve interacting with an LLM. It encapsulates the logic for a single, stateless processing step applied concurrently to many pieces of data.

### Purpose

`PipelineStep` is designed to take an item from an input dictionary, format it into a prompt, send it to an LLM via an `EngineWrapper`, process the LLM's response, validate the result, and save it back to the dictionary. It handles:

*   **State Management:** Loading existing results for items that were already processed in a previous (potentially interrupted) run.
*   **Concurrency:** Works with the `run_task_with_limit` semaphore wrapper provided by `setup_semaphore_and_engines` to manage concurrent API calls.
*   **Retries:** Automatically retries LLM calls if errors occur or if the output fails validation (`validation_function`).
*   **Prompt Formatting:** Injects data from the input item into a prompt template.
*   **Input/Output Processing:** Allows custom functions to modify data before prompting (`input_processor`) and clean/extract data from the LLM response (`output_processor`).
*   **Saving:** Persists the state of the processed items (including intermediate results and potentially raw LLM outputs) to a JSON file.

### Key Interfaces

1.  **`__init__(...)`**
    *   **Configuration:** You initialize a `PipelineStep` with settings that define its behavior for *all* items it processes.
    *   **Key Args:**
        *   `prompt_path`: Base name of the prompt file (e.g., `"write_poem"`). The `.yaml` or `.txt` extension is added automatically based on `completion_mode`.
        *   `sampling_params`: Dict of LLM sampling settings (e.g., `{"max_tokens": 2000, "temperature": 0.8}`).
        *   `output_file`: Base filename (e.g., `"demo_file"`) for the JSON file where intermediate results for this step are stored.
        *   `output_processor`: A function that takes the raw LLM response string and returns the processed/extracted data.
        *   `input_processor` (Optional): A function that takes the input data dict for an item and returns `(modified_input_dict, additional_prompt_args_dict)`. Allows pre-processing data or calculating values needed only for prompt formatting.
        *   `validation_function` (Optional): A function that takes `(processed_output, input_data_dict)` and returns `True` if the output is valid, `False` otherwise (triggers a retry).
        *   `result_key`: The string key under which the *processed* result (from `output_processor`) will be saved in the item's dictionary.
        *   `details_key`: The string key under which the *raw* LLM interaction details (input, response) will be saved if `include_details` is `True` during execution (for meta-datagen).
        *   `max_retries`: How many times to retry on failure.
        *   `regex` (Optional): A regex compiled object used by the underlying `GenerationStep` to extract content from the raw response *before* the `output_processor` is called. Defaults to extracting everything.
        *   `**kwargs`: Any additional keyword arguments are stored and made available for prompt templating.

2.  **`async execute_pipeline(...)`**
    *   **Orchestration:** This is the main method you call to run the step over your entire dataset.
    *   **Key Args:**
        *   `input_dict`: The dictionary containing your dataset items (keys are hashes/IDs, values are data dicts).
        *   `engine_wrapper`: The `EngineWrapper` instance to use for LLM calls.
        *   `rtwl`: The `run_task_with_limit` semaphore wrapper.
        *   `default_prompt_folder`, `prompt_folder`: Paths to prompt directories.
        *   `output_dir`: Path where the `output_file.json` will be saved/loaded.
        *   `completion_mode`, `use_stop`: Flags passed down to the `EngineWrapper`.
        *   `include_details`: Boolean flag to enable saving raw interaction details under `details_key`.
        *   `**kwargs`: Additional keyword arguments passed here are also available for prompt templating during this specific execution.
    *   **Workflow:** Loads existing state from `<output_dir>/<output_file>.json` into `input_dict`, creates an async task for each item in `input_dict` using the internal `run` method, executes tasks concurrently via `rtwl`, waits for completion, filters out items that failed all retries (i.e., missing `result_key`), and finally saves the updated `input_dict` back to the JSON file.

### Internal Methods (Less Commonly Overridden)

*   `async run(...)`: Handles the logic for a *single* item: checks cache, calls `process_input_data`, calls `generate_data`, calls `validation_function`, handles retries, calls `save`.
*   `async generate_data(...)`: Creates the underlying `GenerationStep` and calls its `generate` method to interact with the LLM.
*   `process_input_data(...)`: Calls the provided `input_processor`.
*   `read_previous_output(...)`: Checks if the `result_key` already exists for an item in the `input_dict`.
*   `save(...)`: Updates the `input_dict` *in memory* with the result and details for a single processed item.
*   `load_dataset(...)` / `save_dataset(...)`: Handle loading/saving the entire state dictionary to/from the JSON file, including locking and error handling.

### Usage

Use `PipelineStep` whenever you need to apply the same LLM prompt and processing logic to many independent data items. It's the workhorse for tasks like classification, extraction, rephrasing, and simple generation based on input.

See `example.py` for a concrete usage example.

## `PipelineStep` Subclasses

Augmentoolkit provides several specialized subclasses of `PipelineStep` for common patterns:

### `RandomVariationStep`

(`augmentoolkit/generation_functions/random_variation_step_class.py`)

*   **Purpose:** Generate a specific *number* of diverse variations for each input item using randomly selected prompts.
*   **Interface:**
    *   Adds `variation_generator_count` argument to `__init__`.
    *   Overrides `read_previous_output` to check if *enough* variations already exist (list under `result_key` has length >= `variation_generator_count`).
    *   Overrides `generate_data` to randomly select a `.yaml` prompt file from the *subdirectory* specified by `prompt_path` for each generation call.
    *   Overrides `save` to *append* the new variation to the list stored under `result_key`.
*   **Usage:** Ideal for tasks like the Representation Variation pipeline, where you want multiple different rewrites (e.g., summary, article, list) generated from the same source text chunk.

### `OneToManyStep`

(`augmentoolkit/generation_functions/one_to_many_step.py`)

*   **Purpose:** Transform one input item into *multiple* distinct output items.
*   **Interface:**
    *   Expects `output_processor` to return a *list* of results.
    *   Overrides `save` significantly. Instead of updating the original item's dict, it creates *new* entries in the main dictionary for *each* item in the processed result list. Each new item inherits all data from the original input item, but gets its specific processed result under `result_key`. New keys are generated like `originalKey-index-outputHash`.
    *   Overrides `read_previous_output` to check if *any* keys starting with `originalKey-` exist, indicating the one-to-many transformation has already happened for that input.
    *   Overrides `execute_pipeline` to handle the transformation from the input dictionary structure to the new, expanded output dictionary structure.
*   **Usage:** Essential when a single source should produce multiple independent data points, like generating several distinct Question/Answer pairs from one paragraph ([Multi-Source Facts](multi_source_facts.md)).

### `DepthFirstPipelineStep`

(`augmentoolkit/generation_functions/depth_first_pipeline_step_class.py`)

*   **Purpose:** Support complex, potentially branching pipeline flows where multiple, dependent steps are performed for a single input item before moving to the next.
*   **Interface:**
    *   Subclass of `PipelineStep`.
    *   Overrides `read_previous_output`, `save`, and `run` to work directly with the main dictionary passed around, rather than relying on `execute_pipeline`'s load/save cycle per step.
    *   Designed to be called directly within a custom async orchestrator function (often wrapped by `create_depth_first_executor`).
*   **Usage:** Necessary for pipelines with intricate dependencies or non-linear flows, like [RPToolkit](rptoolkit.md), where generating a story depends on a scene card, which depends on features, etc., all for one initial chunk.
*   **`create_depth_first_executor`:** A helper function that takes your custom orchestrator function (`composition_func`) and wraps it. It handles loading the dataset dict, running your function concurrently for all items, and saving the final dict.

### `MajorityVoteStep`

(`augmentoolkit/generation_functions/majority_vote_step.py`)

*   **Purpose:** Increase the reliability of LLM-based validation or classification by running the same check multiple times and taking a majority vote.
*   **Interface:**
    *   Adds `vote_count_needed`, `percent_true_to_pass`, `final_determination_key` to `__init__`.
    *   Expects `output_processor` to return a boolean-like value.
    *   Overrides `read_previous_output` to check if enough votes (boolean results stored in a list under `result_key`) have been collected.
    *   Overrides `save` to append the boolean result to the list under `result_key`.
    *   Adds `evaluate_final_count_and_save` method to calculate the final boolean outcome (based on `percent_true_to_pass`) and store it under `final_determination_key`. It also filters the raw details saved under `details_key` (if `include_details` is True) to only keep details corresponding to the majority outcome.
    *   Overrides `run` to generate votes until `vote_count_needed` is reached, then call `evaluate_final_count_and_save`.
*   **Usage:** Used in pipelines like [Single-Source Recall](single_source_recall.md) for validating questions and answers with higher confidence than a single LLM call.

## `EngineWrapper`

(`augmentoolkit/generation_functions/engine_wrapper_class.py`)

*   **Purpose:** Provides a unified interface for making asynchronous calls to different LLM API backends.
*   **Interface:**
    *   `__init__(model, api_key, base_url, mode, input_observers=[], output_observers=[])`: Configures the wrapper for a specific model endpoint. `mode` can be `"api"` (for OpenAI-compatible APIs) or `"cohere"`.
    *   `async submit_completion(prompt, sampling_params)`: Sends a request in completion mode.
    *   `async submit_chat(messages, sampling_params)`: Sends a request in chat mode.
*   **Functionality:** Handles the specifics of formatting requests and parsing responses for the configured `mode`. Executes `input_observers` (functions taking `(prompt_or_messages, completion_mode_bool)`) just before sending the request and `output_observers` (functions taking `(prompt_or_messages, completion_string, completion_mode_bool)`) just after receiving the response.
*   **Usage:** Instantiated by `setup_semaphore_and_engines` and passed to `PipelineStep.execute_pipeline`. It's the component that actually talks to the LLM.
*   **Advanced:** Observers are powerful for logging raw interactions (`create_log_observer`), calculating costs (`create_input/output_token_counter`), or potentially modifying requests/responses on the fly (though less common).

## Data Handling Helpers

### Chunking (`generation/core_components/chunking.py`)

*   **Purpose:** Reading source documents and splitting them into appropriately sized chunks for LLM processing.
*   **Key Functions:**
    *   `read_text(input_dir, extensions, output_dir)`: Reads files with specified `extensions` from `input_dir`. Handles `.txt`, `.md`, `.pdf`, `.docx`, `.jsonl`, etc. Caches results in `output_dir` if provided.
    *   `chunk_text_list(text_list, chunk_size, ..., output_dir)`: Takes a list of text items (dicts with `"text"` and `"metadata"`) and chunks each item's text based on `chunk_size` (token limit). Uses sentence tokenization internally. Caches results in `output_dir` if provided.
    *   `read_and_chunk_text(input_dir, ..., chunk_size, ..., output_dir)`: Combines `read_text` and `chunk_text_list`, including caching for both steps.
    *   `count_tokens(text)`: Counts tokens using the default tokenizer.
    *   `count_total_tokens(text_list)`: Counts total tokens in a list of text items.
    *   `subset_text_list(text_list, subset_size, seed)`: Deterministically selects a subset of items from a list.
*   **Usage:** Typically used at the very beginning of a pipeline function to load and prepare the input data before hashing.

### Hashing (`augmentoolkit/generation_functions/hashing_and_ordering.py`)

*   **Purpose:** Convert a list of data items into a dictionary format suitable for `PipelineStep` processing.
*   **Key Function:** `hash_input_list(input_list, key_to_hash_with="text")`
    *   Takes a list of dictionaries.
    *   Sorts the list deterministically based on the value of `key_to_hash_with` to ensure consistent ordering.
    *   Creates a dictionary where keys are the stringified indices (`"0"`, `"1"`, ...) of the items *after sorting*, and values are the original item dictionaries.
    *   (Note: Previous versions used content hashes, but index-based keys proved more robust for resumption).
*   **Usage:** Called immediately after reading/chunking data to create the `input_dict` used by `PipelineStep.execute_pipeline`.

### Setup (`generation/core_components/setup_components.py`)

*   **Purpose:** Standardize the initial setup of engine wrappers, concurrency limits, and path handling.
*   **Key Functions:**
    *   `setup_semaphore_and_engines(...)`: Creates the `run_task_with_limit` async wrapper, `engine_wrapper` (small model), `engine_wrapper_large`, and the underlying `asyncio.Semaphore`. Takes API details, concurrency limit, and observer lists as arguments.
    *   `make_relative_to_self(path)`: Takes a relative path (like a prompt folder path from the config) and makes it absolute relative to the location of the Python script *that called this function*. Essential for finding prompt files co-located with the pipeline code.
*   **Usage:** Called near the beginning of almost every pipeline function.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.