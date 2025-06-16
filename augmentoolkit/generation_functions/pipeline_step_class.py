from tqdm import asyncio as tqdmasyncio
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
import asyncio
from augmentoolkit.generation_functions.generation_step_class import GenerationStep

from augmentoolkit.utils.random import noop

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


def load_dataset_func(input_dict, output_path):
    if os.path.exists(output_path):
        # this dataset exists
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                print(f"Loading file at {output_path}")
                file_contents = json.load(f)
                print("Loaded!")
                # print(file_contents)
            except json.JSONDecodeError:
                # Print a prominent warning about the JSON decode error
                print("\n" + "!" * 80)
                print(f"WARNING: JSON DECODE ERROR in file {output_path}")
                print(
                    "The file exists but contains invalid JSON. Creating an empty dictionary instead."
                )
                print(
                    "This may indicate data corruption or an incomplete write operation."
                )
                print("!" * 80 + "\n")
                file_contents = input_dict

        # assert file_contents is a dict
        assert isinstance(
            file_contents, dict
        ), f"Expected file_contents to be a dictionary, but got {type(file_contents).__name__} instead. This may indicate data corruption in {output_path}."

        # Mutate input_dict to contain file_contents
        input_dict.update(file_contents)


def save_dataset(input_dict, output_path):
    interrupt_count = 0
    max_interrupts = 5
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def _robust_save_interrupt_handler(sig, frame):
        nonlocal interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            _print_banner("SAVE WARNING")
            print(
                f"\nWARNING: Interrupt detected during critical dataset save operation! Unless you really want to lose the in-progress dataset generation for the {output_path} step, please wait a few seconds!",
                file=sys.stderr,
            )
            print(
                f"Press Ctrl+C {max_interrupts - interrupt_count} more times consecutively to force exit.",
                file=sys.stderr,
            )
        elif interrupt_count < max_interrupts:
            print(
                f"\nInterrupt {interrupt_count}/{max_interrupts}. Press Ctrl+C {max_interrupts - interrupt_count} more times to force exit.",
                file=sys.stderr,
            )
        else:
            print(
                f"\n{max_interrupts} consecutive interrupts detected during save. Forcing exit.",
                file=sys.stderr,
            )
            # Restore original handler *before* exiting
            signal.signal(signal.SIGINT, original_sigint_handler)
            sys.exit(130)  # Exit code 130 for SIGINT

    # Outer try...finally ensures original handler is always restored
    save_successful = False
    try:
        # Install custom handler ONLY for the duration of save_dataset
        signal.signal(signal.SIGINT, _robust_save_interrupt_handler)

        # --- Perform the critical operation ---
        print("\nAttempting to save dataset...")  # Indicate save start
        # Create empty structure if file doesn't exist
        if not os.path.exists(output_path):
            # Ensure directory exists before creating file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump({}, f)

        # Create a temporary file with a similar name
        temp_output_path = f"{output_path}.temp"

        try:
            # Write to the temporary file first
            with open(temp_output_path, "w", encoding='utf-8') as f:
                json.dump(input_dict, f)

            # Replace the original file with the temporary file
            # This ensures atomic write operation
            if os.path.exists(output_path):
                os.replace(temp_output_path, output_path)
            else:
                os.rename(temp_output_path, output_path)
        finally:
            # If something goes wrong, try to clean up the temp file
            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except:
                    pass
        # --- Critical operation finished ---
        save_successful = True
    except Exception as e:
        print(f"\nAn error occurred during dataset save: {e}", file=sys.stderr)
        traceback.print_exc()
        # The finally block below will still run to restore the handler.
        # Re-raise the exception if you want the pipeline execution to fail clearly
        raise e
    finally:
        # ALWAYS restore the original handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        if save_successful:
            print(
                f"Dataset saved successfully to {output_path}"
            )  # Optional success message
        elif (
            interrupt_count < max_interrupts
        ):  # Only print if not exited due to interrupts
            print(
                "Dataset save may be incomplete due to an error or interruption (but not forced exit).",
                file=sys.stderr,
            )
        # Reset counter (mostly for clarity, as it's reset next time execute_pipeline runs)
        interrupt_count = 0


def filter_out_nonpresent_keys(item_dict, key_to_check="judgement"):
    keys_to_remove = []
    for key, value in item_dict.items():
        if not key_to_check in value:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del item_dict[key]


# just realized that I could've done process input dta as higher order func smh


class PipelineStep:
    def __init__(
        self,
        method_overrides=None,
        prompt_path=None,
        sampling_params=None,
        output_file=None,
        output_processor=lambda x: x,
        input_processor=lambda x: (x, {}),
        logging_level=logging.INFO,
        result_key="placeholder_result_key",  # this is the key that the result will be saved under in the output dictionary.
        details_key="placeholder_key_details",
        regex=re.compile(r".*", re.DOTALL),
        validation_function=lambda x, y: {"result": True, "message": "default message"},
        max_retries=3,
        log_full_outputs=False,
        **kwargs,  # Anything run time gets passed into .run() instead of the class at initialization. The only other thing I may have to add to that list are the static arguments.
    ):  # things that are args here are things that would be in the code. Some of these will be live-tweakable.
        self.prompt_path = prompt_path
        self.sampling_params = sampling_params
        self.output_processor = output_processor
        self.logging_level = logging_level
        self.result_key = result_key
        self.regex = regex
        self.output_file = output_file
        self.validation_function = validation_function
        self.max_retries = max_retries
        self.log_full_outputs = log_full_outputs
        self.static_arguments = kwargs  # any additional arguments are passed in during generation time. Fits the role of stuff read from the config, like special instructions.
        self.details_key = details_key
        self.input_processor = input_processor

        # Handle method overrides
        if method_overrides:
            for method_name, method in method_overrides.items():
                setattr(
                    self, method_name, method.__get__(self)
                )  # Bind methods to instance

    def process_input_data(self, input_data):
        return self.input_processor(
            input_data
        )  # this should be a dictionary with the keys being the same as the interpolation spots in the prompt. This function in particular will basically always be overridden in subclasses.

    def make_output_path(self, output_dir):
        """Returns full path to the consolidated JSON file"""
        return os.path.join(output_dir, f"{self.output_file}.json")

    def read_previous_output(self, key, output_dict):
        entry = output_dict.get(str(key), {})
        # Check if result key exists in existing entry
        if self.result_key in entry:
            # print(f"Found key {self.result_key}!")
            # print(entry[self.result_key])
            # output_dict[key] = entry
            # print("successfully loaded item")
            # print(output_dict[key])
            return True
        return False

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
        try:

            generator = GenerationStep(
                prompt_path=prompt_path,
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

            # print(processed_data)

            result, full_output, full_response, full_input = await generator.generate(
                **processed_data, **self.static_arguments, **kwargs
            )

            return result, full_output, full_response, full_input
        except Exception as e:
            print(e)
            traceback.print_exc()

    def save(
        self,
        result=None,
        key=None,
        input_dict=None,
        input_data=None,
        full_response=None,
        include_details=False,
        full_input=None,
        completion_mode=False,
    ):

        input_data[self.result_key] = result
        if include_details:
            input_data[self.details_key] = [
                {
                    "full_response": full_response,
                    "full_input": full_input,
                    "completion_mode": completion_mode,  # needed to determine whether full input is a list of messages or a string
                    # did it pass validation is determined by, well, our config and the whole-dataset postprocessors that check for things missing keys. what keys were used is just whatever keys are in the input data since they should not change, they should not be mutated.
                    # the prompt and input are essentially inseparable?
                }
            ]

        input_dict[str(key)] = input_data
        return input_data

    async def run(
        self,
        key=None,
        input_data=None,
        engine_wrapper=None,
        input_dict=None,
        default_prompt_folder=None,
        prompt_folder=None,
        completion_mode=False,
        use_stop=True,
        include_details=False,
        **kwargs,
    ):  # things that are args here are produced during inference time. Including config settings.
        full_prompt_path = (
            self.prompt_path + ".yaml"
            if not completion_mode
            else self.prompt_path + ".txt"
        )

        # Check existing entries
        if self.read_previous_output(key, input_dict):
            return

        processed_data, additional_args_dict = self.process_input_data(
            input_data
        )  # additional args dict is a dict that will go away and die, but is present information for the generation

        error_message = ""  # TODO reserved for when we

        complete = False
        max_retries = self.max_retries
        while not complete and max_retries > 0:
            try:
                result, full_output, full_response, full_input = (
                    await self.generate_data(
                        processed_data,
                        engine_wrapper,
                        full_prompt_path,
                        prompt_folder,
                        default_prompt_folder,
                        completion_mode,
                        use_stop,
                        error_message=error_message,
                        **kwargs,
                        **additional_args_dict,
                    )
                )
                validation_result = self.validation_function(
                    result, input_data
                )  # What I should do here -- have it return a string as well. That string gets passed in as error message t the prompt, allowing for the model to self-correct over repeat generations basd on failed generations. If it excepts then the exception is turned into the error string, handling output processor errors too. It would all work. I just have to change the usage throughout all the pipelines. This would be great and immensely useful.
                if validation_result["result"]:
                    complete = True
                else:
                    error_message = validation_result["message"]
            except Exception as e:
                print(e)
                error_message = str(e)
                traceback.print_exc()
            max_retries -= 1
        if not complete:  # consider raising here and catching in the actual pipeline.
            return

        return self.save(
            result=result,
            key=key,
            input_dict=input_dict,
            input_data=input_data,
            full_response=full_response,
            include_details=include_details,
            completion_mode=completion_mode,
            full_input=full_input,
        )

    def load_dataset(
        self, input_dict, output_dir
    ):  # NOTE footgun here where if the output file path
        output_path = self.make_output_path(output_dir)
        load_dataset_func(input_dict=input_dict, output_path=output_path)

    def save_dataset(self, input_dict, output_dir):
        output_path = self.make_output_path(output_dir=output_dir)
        save_dataset(input_dict=input_dict, output_path=output_path)

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
        loop_callback=noop,
        **kwargs,
    ):
        self.load_dataset(
            input_dict=input_dict, output_dir=output_dir
        )  # if it is at the start and end of every pipeline execution, we get the same behavior as before, past thing will load and save just fine, without constant reads/writes.
        try:
            data_generations_tasks = [
                self.run(
                    input_data=value,
                    engine_wrapper=engine_wrapper,
                    key=key,
                    input_dict=input_dict,
                    default_prompt_folder=default_prompt_folder,
                    prompt_folder=prompt_folder,
                    completion_mode=completion_mode,
                    use_stop=use_stop,
                    include_details=include_details,
                    **kwargs,
                )
                for key, value in input_dict.items()
            ]
            coroutines = [rtwl(task) for task in data_generations_tasks]
            TASK_TIMEOUT_SECONDS = 600  # 10 minutes timeout
            processed_count = 0  # Keep track for logging
            total_tasks = len(coroutines)
            for future in tqdmasyncio.tqdm.as_completed(coroutines):
                processed_count += 1
                loop_callback(input_dict=input_dict, **kwargs)
                try:
                    await asyncio.wait_for(future, timeout=TASK_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    print(
                        f"\nWARNING: Task {processed_count}/{total_tasks} timed out after {TASK_TIMEOUT_SECONDS} seconds.",
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
            traceback.print_exc()
            raise e
        finally:
            # Robust save with interrupt handling
            filter_out_nonpresent_keys(input_dict, key_to_check=self.result_key)
            self.save_dataset(input_dict=input_dict, output_dir=output_dir)


# we don't want failed items biting us later. Mind, this may not be a complete solution, since other things reading from the file might screw us. Hmm there is clearly a bug case, where on the next step as it iterates through keys it will also iterate through keys that failed and supply things with insufficient context. And you know what the solution is? A filter out failed items dict baked into every execute pipeline. Since if the output key does not exist in the item at this key, then it failed, and it should be dropped.
