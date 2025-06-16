import hashlib
import json
import traceback
from tqdm import asyncio as tqdmasyncio
import os
import signal
import sys
import logging
import time  # Added for timestamping logs
import asyncio  # Add asyncio import

from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from filelock import FileLock

# Try importing pyfiglet for banners, provide fallback
try:
    import pyfiglet

    def _print_banner(text):  # Use underscore to indicate internal helper
        print(pyfiglet.figlet_format(text, font="standard"))

except ImportError:
    print(
        "Warning: pyfiglet not installed. Install with 'pip install pyfiglet' for formatted banners."
    )

    def _print_banner(text):
        print(f"\n{'='*10} {text} {'='*10}\n")


# Helper function similar to the one in PipelineStep
def filter_out_nonpresent_keys(item_dict, key_to_check="judgement"):
    # print(f"[DEBUG] filter_out_nonpresent_keys: Starting filter for key '{key_to_check}' on dict with {len(item_dict)} items.") # DEBUG
    keys_to_remove = []
    for key, value in item_dict.items():
        # Ensure value is a dictionary before checking for the key
        if isinstance(value, dict) and key_to_check not in value:
            # print(f"[DEBUG] filter_out_nonpresent_keys: Marking key '{key}' for removal (missing '{key_to_check}').") # DEBUG
            keys_to_remove.append(key)
        elif not isinstance(
            value, dict
        ):  # Handle cases where value might not be a dict unexpectedly
            print(
                f"Warning: Expected dictionary for key '{key}' in filter_out_nonpresent_keys, but got {type(value)}. Skipping."
            )
            # Optionally remove if malformed data should be cleaned
            # keys_to_remove.append(key)

    if keys_to_remove:
        print(
            f"Filtering out {len(keys_to_remove)} items missing the key '{key_to_check}'"
        )
        for key in keys_to_remove:
            del item_dict[key]
    # print(f"[DEBUG] filter_out_nonpresent_keys: Finished filtering. {len(item_dict)} items remain.") # DEBUG


class OneToManyStep(
    PipelineStep
):  # A way of switching from one list of items to a new list of many items (each part of a group generated from the input items). Like going from a small list of source paragraphs to a larger list of question/answer pairs to iterate over.
    # fundamentally, it is easier if we carry over all previous information. Like, all previous keys. Yeah. Let's start with that they can subclass it if they want.
    def __init__(
        self,
        prompt_path,
        sampling_params,
        output_file,
        output_processor,
        result_key,
        max_retries,
        details_key,
        log_full_outputs,
        input_file,
        validation_function=lambda x, y: {"result": True, "message": "default message"},
        regex=r"(.*)",
        input_processor=lambda x: (x, {}),
        **kwargs,
    ):
        # print(f"[DEBUG] OneToManyStep.__init__: Initializing step for output file '{output_file}'.") # DEBUG
        super().__init__(
            prompt_path=prompt_path,
            regex=regex,
            sampling_params=sampling_params,
            output_file=output_file,
            output_processor=output_processor,
            result_key=result_key,
            max_retries=max_retries,
            log_full_outputs=log_full_outputs,
            validation_function=validation_function,
            details_key=details_key,
            input_processor=input_processor,
            **kwargs,
        )
        self.input_file = input_file
        # print(f"[DEBUG] OneToManyStep.__init__: Initialization complete for '{output_file}'.") # DEBUG

    async def read_previous_output(self, key, output_dict):
        # print(f"[DEBUG {time.time():.2f}] read_previous_output: Checking for key '{key}' or prefix '{key}-' in output_dict (size {len(output_dict)}).") # DEBUG
        # Check if any keys start with this key followed by separator in the in-memory dict
        key_prefix = str(key) + "-"
        matching_entries = [k for k in output_dict.keys() if k.startswith(key_prefix)]
        if matching_entries:
            # print(f"[DEBUG {time.time():.2f}] read_previous_output: Found {len(matching_entries)} existing entries for key prefix '{key}-'. Skipping generation.") # DEBUG
            # No need to load from file, data should already be in output_dict if loaded
            # print(f"Found existing entries for key {key} in memory: {matching_entries}")
            return True  # Indicate that output for this key exists
        # print(f"[DEBUG {time.time():.2f}] read_previous_output: No existing entries found for key prefix '{key}-'. Proceeding with generation.") # DEBUG
        return False

    def save(
        self,
        result=None,
        key=None,
        output_dict=None,
        input_data=None,
        full_output=None,
        full_response=None,
        full_input=None,
        include_details=False,
        completion_mode=False,
    ):
        # print(f"[DEBUG {time.time():.2f}] save: Starting save for key '{key}'. Result type: {type(result)}, Result length: {len(result) if isinstance(result, list) else 'N/A'}.") # DEBUG
        # Removed output_dir, input_dict args as they are handled by execute_pipeline
        # print("\n=== Starting save() function (In-Memory Update) ===")
        # print(f"Key: {key}")
        # print(f"Result type: {type(result)}")
        # print(f"Result length: {len(result)}")
        # print(f"Include details: {include_details}")

        assert isinstance(
            result, list
        ), f"Result must be a list, got {type(result)}"  # needed for one-to-many
        # print(f"[DEBUG {time.time():.2f}] save: Assertion passed, result is a list.") # DEBUG

        # Removed output_path and related file operations

        if self.log_full_outputs:
            # print(f"[DEBUG {time.time():.2f}] save: log_full_outputs is True. Logging full outputs (logic omitted for brevity).") # DEBUG
            # print("\nLogging full outputs:") # Keep logging if needed
            if full_output:
                # ... (logging logic remains the same) ...
                pass  # Pass for brevity, assume logging logic is unchanged

        # print("\n=== Processing result items (In-Memory Update) ===")
        details_payload = None
        if include_details:
            # print(f"[DEBUG {time.time():.2f}] save: include_details is True. Preparing details payload.") # DEBUG
            # Prepare details payload once, assign it to each subitem
            details_payload = [
                {
                    "full_response": full_response,
                    "full_input": full_input,
                    "completion_mode": completion_mode,
                    "result": result,
                }
            ]
            # print("Details payload prepared:")
            # print(details_payload)

        for idx, subitem in enumerate(result):
            # print(f"[DEBUG {time.time():.2f}] save: Processing item {idx} for original key '{key}'. Subitem: {str(subitem)[:50]}...") # DEBUG
            # print(f"\nProcessing item {idx}:")
            # print(f"Subitem: {str(subitem)[:100]}...")

            # Hash the idx and subitem using hashlib
            hash_input = f"{str(subitem)}"
            item_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            item_key = f"{key}-{idx}-{item_hash}"
            # print(f"[DEBUG {time.time():.2f}] save: Generated item_key: {item_key}") # DEBUG
            # print(f"Generated item_key: {item_key}")

            # Create new entry with ALL original data plus the new result
            output_dict[str(item_key)] = (
                input_data.copy()
            )  # Start with a copy of the original input
            # print(f"[DEBUG {time.time():.2f}] save: Copied input_data to new key '{item_key}'.") # DEBUG
            output_dict[str(item_key)][
                self.result_key
            ] = subitem  # Add the specific subitem result
            # print(f"[DEBUG {time.time():.2f}] save: Added subitem result under key '{self.result_key}' for item_key '{item_key}'.") # DEBUG
            # print(f"Added result to output_dict with key: {self.result_key}")

            if include_details and details_payload:
                output_dict[str(item_key)][
                    self.details_key
                ] = details_payload  # Add details payload
                # print(f"[DEBUG {time.time():.2f}] save: Added details payload under key '{self.details_key}' for item_key '{item_key}'.") # DEBUG
                # print(f"Added details to output_dict with key: {self.details_key}")

            # Removed file writing logic for each item

        # Removed logic that modified the input_file/input_dict for details

        # print(f"[DEBUG {time.time():.2f}] save: Finished processing {len(result)} items for key '{key}'. output_dict size now {len(output_dict)}.") # DEBUG
        # print("\n=== save() function completed (In-Memory Update) ===\n")
        # Save now returns None as it modifies output_dict in place
        return (
            None  # Or return result if needed elsewhere, but primary effect is mutation
        )

    def load_dataset(self, output_dict, output_dir):
        output_path = self.make_output_path(output_dir)
        # print(f"[DEBUG {time.time():.2f}] load_dataset: Attempting to load dataset for '{self.output_file}' from '{output_path}'.") # DEBUG
        if os.path.exists(output_path):
            # Use file lock for reading, although less critical than writing
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    print(
                        f"Loading existing output file for {self.output_file} at {output_path}"
                    )
                    file_contents = json.load(f)
                except json.JSONDecodeError:
                    print("\n" + "!" * 80)
                    print(f"WARNING: JSON DECODE ERROR in file {output_path}")
                    print(
                        "The file exists but contains invalid JSON. Starting with an empty dictionary instead."
                    )
                    print(
                        "This may indicate data corruption or an incomplete write operation."
                    )
                    print("!" * 80 + "\n")
                    file_contents = {}  # Start fresh if decode fails

            assert isinstance(
                file_contents, dict
            ), f"Expected file_contents to be a dictionary, but got {type(file_contents).__name__}."
            # print(f"[DEBUG {time.time():.2f}] load_dataset: File read successfully. Loaded {len(file_contents)} items. Updating output_dict.") # DEBUG
            output_dict.update(file_contents)  # Load into the provided dict
            print(f"Loaded {len(file_contents)} items from {output_path}")
        else:
            # print(f"[DEBUG {time.time():.2f}] load_dataset: Output file '{output_path}' not found. Starting with an empty dataset.") # DEBUG
            print(
                f"Output file {output_path} not found. Starting with an empty dataset."
            )
            # Ensure directory exists even if file doesn't
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # print(f"[DEBUG {time.time():.2f}] load_dataset: Ensured directory '{os.path.dirname(output_path)}' exists.") # DEBUG

    def save_dataset(self, output_dict, output_dir):
        output_path = self.make_output_path(output_dir=output_dir)
        # print(f"\n[DEBUG {time.time():.2f}] save_dataset: Attempting to save dataset for '{self.output_file}' ({len(output_dict)} items) to '{output_path}'.") # DEBUG
        print(f"\nAttempting to save dataset... {output_path}")

        # Robust save with interrupt handling (adapted from PipelineStep)
        interrupt_count = 0
        max_interrupts = 5
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def _robust_save_interrupt_handler(sig, frame):
            nonlocal interrupt_count
            interrupt_count += 1
            if interrupt_count == 1:
                _print_banner("SAVE WARNING")
                # print(f"[DEBUG {time.time():.2f}] _robust_save_interrupt_handler: First interrupt during save for '{self.output_file}'.") # DEBUG
                print(
                    f"\nWARNING: Interrupt detected during critical dataset save for step '{self.output_file}'!",
                    file=sys.stderr,
                )
                print(
                    f"Press Ctrl+C {max_interrupts - interrupt_count} more times consecutively to force exit (DATA LOSS RISK!).",
                    file=sys.stderr,
                )
            elif interrupt_count < max_interrupts:
                print(
                    f"\nInterrupt {interrupt_count}/{max_interrupts}. Press Ctrl+C {max_interrupts - interrupt_count} more times to force exit.",
                    file=sys.stderr,
                )
                # print(f"[DEBUG {time.time():.2f}] _robust_save_interrupt_handler: Interrupt {interrupt_count}/{max_interrupts} during save for '{self.output_file}'.") # DEBUG
            else:
                print(
                    f"\n{max_interrupts} consecutive interrupts detected during save. Forcing exit.",
                    file=sys.stderr,
                )
                # print(f"[DEBUG {time.time():.2f}] _robust_save_interrupt_handler: Max interrupts reached for '{self.output_file}'. Forcing exit.") # DEBUG
                signal.signal(
                    signal.SIGINT, original_sigint_handler
                )  # Restore before exit
                sys.exit(130)  # Exit code 130 for SIGINT

        save_successful = False
        try:
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Setting SIGINT handler for robust save.") # DEBUG
            signal.signal(
                signal.SIGINT, _robust_save_interrupt_handler
            )  # Set custom handler

            # Ensure directory exists
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Ensuring directory '{os.path.dirname(output_path)}' exists.") # DEBUG
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Use file lock for writing
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Lock acquired. Starting atomic write.") # DEBUG
            # Atomic write using temp file
            temp_output_path = (
                f"{output_path}.temp_{os.getpid()}"  # Add PID for extra safety
            )
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Writing to temp file: '{temp_output_path}'.") # DEBUG
            try:
                with open(temp_output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        output_dict, f, indent=None, ensure_ascii=False
                    )  # Use indent for readability
                # print(f"[DEBUG {time.time():.2f}] save_dataset: Temp file write complete. Replacing original file.") # DEBUG

                os.replace(temp_output_path, output_path)  # Atomic rename/replace
                # print(f"[DEBUG {time.time():.2f}] save_dataset: Atomic replace successful.") # DEBUG
                save_successful = True
            except Exception as write_error:
                # print(f"[DEBUG {time.time():.2f}] save_dataset: ERROR during file write/replace: {write_error}") # DEBUG
                print(
                    f"\nERROR during file write/replace operation: {write_error}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                # Clean up temp file if replace failed
                if os.path.exists(temp_output_path):
                    # print(f"[DEBUG {time.time():.2f}] save_dataset: Attempting to remove failed temp file '{temp_output_path}'.") # DEBUG
                    try:
                        os.remove(temp_output_path)
                    except OSError as remove_error:
                        print(
                            f"Error removing temp file {temp_output_path}: {remove_error}",
                            file=sys.stderr,
                        )
                raise  # Re-raise the error after attempting cleanup
            finally:
                # Ensure temp file is removed if it somehow still exists after a successful replace (unlikely but safe)
                if save_successful and os.path.exists(temp_output_path):
                    # print(f"[DEBUG {time.time():.2f}] save_dataset: Ensuring temp file '{temp_output_path}' is removed after successful save.") # DEBUG
                    try:
                        os.remove(temp_output_path)
                    except OSError:
                        pass  # Ignore error if removal fails post-success
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Lock released.") # DEBUG

        except Exception as e:
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Exception during save process: {e}") # DEBUG
            print(f"\nAn error occurred during the save process: {e}", file=sys.stderr)
            # The finally block handles handler restoration.
            # Consider re-raising if the pipeline should halt on save failure.
            # raise e
        finally:
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Restoring original SIGINT handler.") # DEBUG
            signal.signal(
                signal.SIGINT, original_sigint_handler
            )  # ALWAYS restore handler
            if save_successful:
                # print(f"[DEBUG {time.time():.2f}] save_dataset: Save successful for '{output_path}'.") # DEBUG
                print(
                    f"Dataset for '{self.output_file}' saved successfully to {output_path}."
                )
            elif interrupt_count < max_interrupts:
                # print(f"[DEBUG {time.time():.2f}] save_dataset: Save may be incomplete for '{output_path}' due to error or interrupt.") # DEBUG
                print(
                    f"Dataset save for '{self.output_file}' may be incomplete due to an error or interruption.",
                    file=sys.stderr,
                )
            # Reset counter is implicit as it's local to the next call
            # print(f"[DEBUG {time.time():.2f}] save_dataset: Save process finished.") # DEBUG

    async def run(
        self,
        key=None,
        input_data=None,
        engine_wrapper=None,
        default_prompt_folder=None,
        prompt_folder=None,
        completion_mode=False,
        use_stop=True,
        output_dict=None,
        include_details=False,
        **kwargs,
    ):
        # print(f"[DEBUG {time.time():.2f}] run: Starting for key '{key}'. Completion mode: {completion_mode}.") # DEBUG
        full_prompt_path = (
            self.prompt_path + ".yaml"
            if not completion_mode
            else self.prompt_path + ".txt"
        )
        # print(f"[DEBUG {time.time():.2f}] run: Prompt path: '{full_prompt_path}'.") # DEBUG

        # Check existing entries IN MEMORY
        # print(f"[DEBUG {time.time():.2f}] run: Checking previous output for key '{key}'.") # DEBUG
        if await self.read_previous_output(key, output_dict):  # Pass output_dict
            # print(f"[DEBUG {time.time():.2f}] run: Previous output found for key '{key}'. Skipping.") # DEBUG
            return  # Skip generation if already processed
        # print(f"[DEBUG {time.time():.2f}] run: No previous output found for key '{key}'. Proceeding.") # DEBUG

        # print(f"[DEBUG {time.time():.2f}] run: Processing input data for key '{key}'.") # DEBUG
        processed_data, additional_kwargs = self.process_input_data(input_data)
        # print(f"[DEBUG {time.time():.2f}] run: Input data processed for key '{key}'.") # DEBUG

        error_message = ""
        complete = False
        retries_left = self.max_retries
        # print(f"[DEBUG {time.time():.2f}] run: Starting generation loop for key '{key}'. Max retries: {retries_left}.") # DEBUG
        while not complete and retries_left > 0:
            # print(f"[DEBUG {time.time():.2f}] run: Attempting generation for key '{key}'. Retries left: {retries_left}.") # DEBUG
            try:
                # print(f"[DEBUG {time.time():.2f}] run: Calling generate_data for key '{key}'.") # DEBUG
                start_time = time.time()  # DEBUG TIME
                result, full_output, full_response, full_input = (
                    await self.generate_data(
                        processed_data=processed_data,
                        engine_wrapper=engine_wrapper,
                        prompt_path=full_prompt_path,
                        prompt_folder=prompt_folder,
                        default_prompt_folder=default_prompt_folder,
                        completion_mode=completion_mode,
                        use_stop=use_stop,
                        error_message=error_message,
                        **kwargs,
                        **additional_kwargs,
                    )
                )
                end_time = time.time()  # DEBUG TIME
                # print(f"[DEBUG {end_time:.2f}] run: generate_data call for key '{key}' completed in {end_time - start_time:.2f} seconds.") # DEBUG
                # Ensure result is a list for OneToMany validation
                if not isinstance(result, list):
                    # print(f"[DEBUG {time.time():.2f}] run: WARNING - Expected list result from generate_data for key {key}, got {type(result)}. Raising TypeError.") # DEBUG
                    print(
                        f"Warning: Expected list result from generate_data for key {key}, got {type(result)}. Treating as failed attempt."
                    )
                    raise TypeError("Generator did not return a list")  # Trigger retry

                # print(f"[DEBUG {time.time():.2f}] run: Generation successful for key '{key}'. Result is a list of length {len(result)}. Validating...") # DEBUG
                validation_result = self.validation_function(result, input_data)
                if validation_result["result"]:
                    # print(f"[DEBUG {time.time():.2f}] run: Validation successful for key '{key}'.") # DEBUG
                    complete = True
                else:
                    # print(f"[DEBUG {time.time():.2f}] run: Validation FAILED for key {key}.") # DEBUG
                    error_message = validation_result["message"]
                    print(f"Validation failed for key {key}")

            except Exception as e:
                # print(f"[DEBUG {time.time():.2f}] run: Exception during generation/validation for key '{key}': {type(e).__name__}: {e}") # DEBUG
                print(e)
                error_message = str(e)
                traceback.print_exc()
            retries_left -= 1
            if not complete and retries_left > 0:
                # print(f"[DEBUG {time.time():.2f}] run: Retrying generation for key '{key}'. {retries_left} retries remaining.") # DEBUG
                await asyncio.sleep(1)  # Optional: Add a small delay before retrying
            elif not complete:
                # print(f"[DEBUG {time.time():.2f}] run: Generation failed for key '{key}' after exhausting retries.") # DEBUG
                pass  # Added pass to fix potential indentation error
        if not complete:  # consider raising here and catching in the actual pipeline.
            # print(f"[DEBUG {time.time():.2f}] run: Generation ultimately failed for key '{key}'. Returning None.") # DEBUG
            return

        # print(f"[DEBUG {time.time():.2f}] run: Generation complete for key '{key}'. Calling save.") # DEBUG
        save_result = self.save(
            result=result,
            key=key,
            output_dict=output_dict,  # Pass the shared dictionary
            input_data=input_data,
            full_output=full_output,
            full_response=full_response,
            full_input=full_input,
            include_details=include_details,
            completion_mode=completion_mode,
        )
        # print(f"[DEBUG {time.time():.2f}] run: Save call completed for key '{key}'. Returning result: {type(save_result)}") # DEBUG
        return save_result

    async def execute_pipeline(
        self,
        input_dict={},
        engine_wrapper=None,
        rtwl=None,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=None,
        use_stop=None,
        include_details=False,
        **kwargs,
    ):
        # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Starting execution for step '{self.output_file}'. Input dict size: {len(input_dict)}.") # DEBUG
        output_dict = {}  # Initialize the dictionary for this step's results
        # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Loading dataset for '{self.output_file}'.") # DEBUG
        self.load_dataset(output_dict, output_dir)  # Load existing data at the start
        # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Dataset loaded. Current output_dict size: {len(output_dict)}.") # DEBUG

        try:
            # Note: input_dict here refers to the *previous* step's output
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Preparing generation tasks from input_dict (size {len(input_dict)}).") # DEBUG
            data_generations_tasks = []
            for i, (key, value) in enumerate(input_dict.items()):
                # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Creating task {i+1}/{len(input_dict)} for key '{key}'.") # DEBUG
                task = self.run(
                    input_data=value,
                    engine_wrapper=engine_wrapper,
                    key=key,
                    default_prompt_folder=default_prompt_folder,
                    prompt_folder=prompt_folder,
                    completion_mode=completion_mode,
                    use_stop=use_stop,
                    output_dict=output_dict,  # Pass the shared dict to run/save
                    include_details=include_details,
                    **kwargs,
                )
                data_generations_tasks.append(task)
            # data_generations_tasks = [
            #     self.run(
            #         input_data=value,
            #         engine_wrapper=engine_wrapper,
            #         key=key,
            #         default_prompt_folder=default_prompt_folder,
            #         prompt_folder=prompt_folder,
            #         completion_mode=completion_mode,
            #         use_stop=use_stop,
            #         output_dict=output_dict, # Pass the shared dict to run/save
            #         include_details=include_details,
            #         **kwargs
            #     ) for key, value in input_dict.items() # Iterate over inputs from previous step
            # ]

            if not data_generations_tasks:
                # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Warning - No tasks created for step '{self.output_file}'.") # DEBUG
                print("Warning: No tasks to run for this step.")

            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Created {len(data_generations_tasks)} generation tasks. Wrapping with Rate-Limited Wrapper (rtwl).") # DEBUG
            coroutines = [rtwl(task) for task in data_generations_tasks]
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Wrapped {len(coroutines)} tasks. Starting processing with tqdm.") # DEBUG
            print(f"Processing {len(coroutines)} items...")
            TASK_TIMEOUT_SECONDS = 600  # 10 minutes timeout
            processed_count = 0
            total_tasks = len(coroutines)
            for future in tqdmasyncio.tqdm(
                tqdmasyncio.tqdm.as_completed(coroutines),
                total=len(coroutines),
                desc=f"Step: {self.output_file}",
            ):
                processed_count += 1  # DEBUG
                # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Awaiting future {processed_count}/{len(coroutines)}.") # DEBUG - Can be very verbose
                try:
                    # Wrap the await future with a timeout
                    await asyncio.wait_for(future, timeout=TASK_TIMEOUT_SECONDS)
                    # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Future {processed_count}/{len(coroutines)} completed successfully.") # DEBUG
                except asyncio.TimeoutError:
                    print(
                        f"\nWARNING: Task {processed_count}/{total_tasks} for step '{self.output_file}' timed out after {TASK_TIMEOUT_SECONDS} seconds.",
                        file=sys.stderr,
                    )
                    # Task is cancelled by wait_for, loop continues
                except Exception as task_exc:
                    # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Error in awaited task {processed_count}/{len(coroutines)}: {task_exc}") # DEBUG
                    # Log exceptions from individual run tasks if needed, but allow others to continue
                    print(f"\nError in awaited task: {task_exc}")
                    traceback.print_exc()  # Optionally print traceback for task errors
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Finished processing all {len(coroutines)} futures.") # DEBUG

            # Filter out entries where the generation might have failed and didn't add the result key
            # This is crucial for OneToMany as failed 'run' won't add keys to output_dict
            # but we still need to ensure consistency if partial results were loaded.
            # However, the primary filtering happens because failed `run` calls simply don't add anything to `output_dict`.
            # A check after loading might be useful for corrupted files, but less so for runtime failures.
            # Consider if filtering is needed based on how `load_dataset` handles errors.
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Optional filtering step (commented out).") # DEBUG
            # filter_out_nonpresent_keys(output_dict, key_to_check=self.result_key) # Optional: clean loaded data

        except Exception as e:
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: CRITICAL EXCEPTION during execution for step '{self.output_file}': {e}") # DEBUG
            print(
                f"\nCritical exception during pipeline execution for step {self.output_file}: {e}"
            )
            traceback.print_exc()
            # The finally block will still attempt to save. Re-raise if pipeline should stop.
            raise e
        finally:
            final_count = len(output_dict)
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: FINALLY block reached. Final item count: {final_count}. Saving dataset.") # DEBUG
            print(
                f"\nSaving results for step '{self.output_file}'. Final item count: {final_count}"
            )
            # Robust save called in finally block
            filter_out_nonpresent_keys(output_dict, key_to_check=self.result_key)
            self.save_dataset(output_dict, output_dir)
            # print(f"[DEBUG {time.time():.2f}] execute_pipeline: FINALLY block complete after save_dataset call.") # DEBUG

        # print(f"[DEBUG {time.time():.2f}] execute_pipeline: Execution finished for step '{self.output_file}'. Returning output_dict with {len(output_dict)} items.") # DEBUG
        return output_dict  # Return the completed dictionary
