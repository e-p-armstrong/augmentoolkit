import json
import os
import random
import re
import traceback
import sys
import asyncio
from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.generation_functions.pipeline_step_class import (
    PipelineStep,
    filter_out_nonpresent_keys,
)
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from tqdm import asyncio as tqdmasyncio


class RandomVariationStep(
    PipelineStep
):  # pipeline steps store the settings and the prompt, and prevent us from having to repeat the "read previous output" code among other things
    def __init__(
        self,
        sampling_params=None,
        output_file=None,
        intermediate_output_path=None,
        save_path=None,
        use_stop=None,
        completion_mode=None,
        validation_function=lambda x, y: {"result": True, "message": "default message"},
        max_retries=3,
        result_key=None,
        prompt_path=None,
        regex=re.compile(r".*", re.DOTALL),
        variation_generator_count=None,
        output_processor=lambda x: x,
        log_full_outputs=False,
        method_overrides={},
        details_key=None,
    ):
        self.variation_generator_count = variation_generator_count
        super().__init__(
            sampling_params=sampling_params,
            output_file=output_file,  # Add output_file parameter for consolidated storage
            intermediate_output_path=intermediate_output_path,
            save_path=save_path,
            use_stop=use_stop,
            completion_mode=completion_mode,
            validation_function=validation_function,
            max_retries=max_retries,
            result_key=result_key,
            prompt_path=prompt_path,
            output_processor=output_processor,
            log_full_outputs=log_full_outputs,
            method_overrides=method_overrides,
            details_key=details_key,
        )

    async def read_previous_output(self, key, input_dict):
        entry = input_dict.get(str(key), {})
        # if not entry:
        #      print(f"  Key '{str(key)}' not found in input_dict.")
        # else:
        #      print(f"  Found entry. Keys: {list(entry.keys())}")

        current_variations = entry.get(self.result_key, [])

        should_skip = False
        if len(current_variations) >= self.variation_generator_count:
            print(
                f"  Found {len(current_variations)} variations, which meets or exceeds target {self.variation_generator_count}. Will skip generation for key {key}."
            )
            should_skip = True
        # else:

        return should_skip

    async def generate_data(
        self,
        processed_data,
        engine_wrapper,
        prompt_path,
        prompt_folder,
        default_prompt_folder,
        completion_mode,
        use_stop,
        **kwargs,
    ):

        # randomly select a yaml file from the prompt folder
        # prompt_path is the name of the subfolder; prompt_folder is, per usual, the path to the overall PROMPTS folder.
        search_dir = os.path.join(prompt_folder, prompt_path)
        if not os.path.exists(search_dir):
            search_dir = os.path.join(default_prompt_folder, prompt_path)

        if not completion_mode:
            prompt_files = [f for f in os.listdir(search_dir) if f.endswith(".yaml")]
        else:
            prompt_files = [
                f for f in os.listdir(search_dir) if f.endswith(".yaml")
            ]  # NOTE: This looks like a potential bug, it lists .yaml for completion mode too.

        if not prompt_files:
            error_msg = f"  ERROR: No suitable prompt files found in {search_dir} for completion_mode={completion_mode}"
            print(error_msg)
            raise FileNotFoundError(error_msg)

        random_prompt_file = random.choice(prompt_files)
        prompt_filepath = os.path.join(search_dir, random_prompt_file)

        try:
            generator = GenerationStep(
                prompt_path=prompt_filepath,
                default_prompt_folder=default_prompt_folder,
                sampling_params=self.sampling_params,
                completion_mode=completion_mode,
                engine_wrapper=engine_wrapper,
                output_processor=self.output_processor,
                retries=1,
                logging_level=self.logging_level,
                use_stop=use_stop,
                prompt_folder=prompt_folder,
                regex=self.regex,
            )

            # Note: We don't log the actual return values here as they could be large
            result, full_output, full_response, full_input = await generator.generate(
                **processed_data, **self.static_arguments, **kwargs
            )
            return result, full_output, full_response, full_input
        except Exception as e:
            print(f"  ERROR during GenerationStep initialization or execution: {e}")
            traceback.print_exc()
            # Re-raise the exception to be handled by the caller (run method)
            raise

    def save(
        self,
        result=None,
        key=None,
        input_dict=None,
        input_data=None,
        full_response=None,
        full_input=None,
        completion_mode=False,
        include_details=False,
    ):

        # Get existing entry or create a new one. Avoid copying input_data unless necessary.
        entry = input_dict.get(str(key))
        if entry is None:
            # If the key truly doesn't exist, we MUST base it on input_data
            # This case should ideally not happen often if execute_pipeline populates correctly
            entry = input_data.copy() if input_data else {}
        # else:

        # Ensure the result key exists and is a list
        if self.result_key not in entry or not isinstance(
            entry.get(self.result_key), list
        ):  # Use .get for safety
            entry[self.result_key] = []

        try:
            entry[self.result_key].append(result)
        except Exception as e:
            print(f"  ERROR appending result: {e}")
            traceback.print_exc()
            raise

        if include_details:
            # Ensure the details key exists and is a list
            if self.details_key not in entry or not isinstance(
                entry.get(self.details_key), list
            ):  # Use .get for safety
                entry[self.details_key] = []

            detail_entry = {
                "full_response": full_response,
                "full_input": full_input,
                "completion_mode": completion_mode,
            }
            try:
                entry[self.details_key].append(detail_entry)
            except Exception as e:
                print(f"  ERROR appending detail entry: {e}")
                traceback.print_exc()
                raise
        # else:

        try:
            input_dict[str(key)] = entry
        except Exception as e:
            print(f"  ERROR assigning entry to input_dict: {e}")
            traceback.print_exc()
            raise

        return entry

    async def run(
        self,
        key=None,
        input_data=None,
        engine_wrapper=None,
        input_dict=None,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=False,
        use_stop=True,
        include_details=False,
        **kwargs,
    ):

        should_skip = await self.read_previous_output(key, input_dict)
        if should_skip:
            return
        # else:

        processed_data, additional_kwargs = self.process_input_data(input_data)
        # existing_variations = [] # This variable seems unused after the FileLock block was removed

        # Determine current count and number needed based *only* on in-memory input_dict
        current_variations_in_memory = input_dict.get(str(key), {}).get(
            self.result_key, []
        )
        num_existing_in_memory = len(current_variations_in_memory)
        num_variations_needed = self.variation_generator_count - num_existing_in_memory

        if num_variations_needed <= 0:
            # This case should technically be caught by read_previous_output, but added for robustness
            return

        # Generate remaining variations
        for i in range(num_variations_needed):
            error_message = ""
            complete = False
            max_retries = self.max_retries
            while not complete and max_retries > 0:
                try:
                    result, full_output, full_response, full_input = (
                        await self.generate_data(
                            processed_data,
                            engine_wrapper,
                            self.prompt_path,
                            prompt_folder,
                            default_prompt_folder,
                            completion_mode,
                            use_stop,
                            error_message=error_message,
                            **kwargs,
                            **additional_kwargs,
                        )
                    )

                    validation_passed = self.validation_function(result, input_data)
                    if validation_passed["result"]:
                        complete = True
                    else:
                        error_message = validation_passed["message"]
                except Exception as e:
                    print(f"      ERROR during generate_data or validation: {e}")
                    error_message = str(e)
                    traceback.print_exc()
                max_retries -= 1
                if not complete:
                    print(
                        f"      Attempt failed or validation failed. Retries left: {max_retries}"
                    )
            # End of while loop

            if not complete:
                print(
                    f"    ERROR: Failed to generate a valid variation after {self.max_retries} attempts for key {key}, iteration {i+1}."
                )
                # We might want to remove the key from input_dict here or handle failure differently
                return
            # else:

            self.save(
                result=result,
                key=key,
                input_dict=input_dict,
                input_data=input_dict.get(
                    str(key)
                ),  # Pass the current state from input_dict
                full_response=full_response,
                full_input=full_input,
                completion_mode=completion_mode,
                include_details=include_details,
            )
        # End of for loop

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
        self.load_dataset(
            input_dict=input_dict, output_dir=output_dir
        )  # if it is at the start and end of every pipeline execution, we get the same behavior as before, past thing will load and save just fine, without constant reads/writes.
        try:
            data_generations_tasks = [
                self.run(
                    key=key,
                    input_data=value,
                    engine_wrapper=engine_wrapper,
                    input_dict=input_dict,
                    default_prompt_folder=default_prompt_folder,
                    prompt_folder=prompt_folder,
                    output_dir=output_dir,
                    completion_mode=completion_mode,
                    use_stop=use_stop,
                    include_details=include_details,
                    **kwargs,
                )
                for key, value in input_dict.items()
            ]
            coroutines = [rtwl(task) for task in data_generations_tasks]
            TASK_TIMEOUT_SECONDS = 600  # 10 minutes timeout
            processed_count = 0
            total_tasks = len(coroutines)
            for future in tqdmasyncio.tqdm.as_completed(coroutines):
                processed_count += 1
                try:
                    await asyncio.wait_for(future, timeout=TASK_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    print(
                        f"\nWARNING: Task {processed_count}/{total_tasks} for step '{self.output_file}' timed out after {TASK_TIMEOUT_SECONDS} seconds.",
                        file=sys.stderr,
                    )
                    # Task is cancelled by wait_for, loop continues
                except Exception as e_inner:  # Catch exceptions from the future itself
                    print(
                        f"\nError processing task {processed_count}/{total_tasks}: {e_inner}"
                    )
                    traceback.print_exc()  # Optionally log traceback

            filter_out_nonpresent_keys(input_dict, key_to_check=self.result_key)
        except Exception as e:
            print(f"Exception occurred during task execution: {e}")
            # traceback.print_exc()
            raise e
        finally:
            # Robust save with interrupt handling
            self.save_dataset(input_dict=input_dict, output_dir=output_dir)
