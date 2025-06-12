import hashlib
import json
import random
import traceback
from tqdm import asyncio as tqdmasyncio
import os
import sys  # Add sys import
import asyncio  # Add asyncio import

from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.generation_functions.pipeline_step_class import (
    PipelineStep,
    filter_out_nonpresent_keys,
)

# TODO make this stop checking if/when it becomes impossible for either failing or passing to occur. I.e., early stopping in the event of the result being determined already.
# Also note that working with abstractions makes the logic around skipping steps NICE AND EASY
# we just put a majority vote and its "remove failed items" check into a conditional
# and that's the branching
# all nice and easy to look at
# and easier to learn from too


class MajorityVoteStep(
    PipelineStep
):  # Basic idea is that, result will always be a boolean, and we run this multiple times until we have a result list as long as we expect (runtime arg), and then we take the majority vote of that. Also, runtime arg of % positive required to pass to allow for easy supermajority.
    def __init__(
        self,
        prompt_path,
        regex,
        sampling_params,
        output_file,
        output_processor,
        result_key,
        max_retries=3,
        log_full_outputs=False,
        final_determination_key=None,
        vote_count_needed=1,
        percent_true_to_pass=0.5,
        validation_function=lambda x, y: {"result": True, "message": "default message"},
        **kwargs,
    ):
        # print(f"[DEBUG MajorityVoteStep __init__] Initializing MajorityVoteStep.")
        # print(f"[DEBUG MajorityVoteStep __init__] vote_count_needed={vote_count_needed}, percent_true_to_pass={percent_true_to_pass}, result_key='{result_key}', final_determination_key='{final_determination_key}'")
        self.vote_count_needed = vote_count_needed
        self.percent_true_to_pass = percent_true_to_pass
        self.final_determination_key = final_determination_key
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
            **kwargs,
        )
        # print(f"[DEBUG MajorityVoteStep __init__] Initialization complete.")

    async def read_previous_output(self, key, input_dict):
        # print(f"[DEBUG MajorityVoteStep read_previous_output] Reading previous output for key='{key}'.")
        entry = input_dict.get(str(key), {})
        current_votes = entry.get(self.result_key, [])
        # print(f"[DEBUG MajorityVoteStep read_previous_output] Found {len(current_votes)} existing votes for key='{key}'.")

        if len(current_votes) >= self.vote_count_needed:
            # print(f"[DEBUG MajorityVoteStep read_previous_output] Vote count needed ({self.vote_count_needed}) met for key='{key}'. Returning True.")
            return True, current_votes
        # print(f"[DEBUG MajorityVoteStep read_previous_output] Vote count needed ({self.vote_count_needed}) not met for key='{key}'. Returning False.")
        return False, current_votes  # get the votes so far

    def save(
        self,
        result,
        key,
        input_dict,
        input_data,
        full_response,
        full_input,
        include_details=False,
        completion_mode=False,
    ):
        # print(f"[DEBUG MajorityVoteStep save] Saving result for key='{key}'. Result={result}, include_details={include_details}")
        # Get existing entry from input_dict first, avoid copying input_data if key exists
        entry = input_dict.get(str(key))
        if entry is None:
            # Only copy if the key is genuinely missing from input_dict
            # print(f"[DEBUG MajorityVoteStep save] Key '{key}' not found in input_dict, creating new entry.")
            entry = {}

        if self.result_key not in entry:
            # print(f"[DEBUG MajorityVoteStep save] Initializing result list for key='{key}'.")
            entry[self.result_key] = []
        entry[self.result_key].append(result)  # result is a boolean?
        # print(f"[DEBUG MajorityVoteStep save] Appended result. New vote count for key='{key}': {len(entry[self.result_key])}.")

        # we hold off on saving the details possibly until the end, because we only want to save those that match the majority vote result
        if (
            include_details
        ):  # well actually, we save them all but then at the end we filter... no wait what's actually the point. The logic can be shoved in the end datagen pipeline, if we keep all the boolean results together then... wait but we will not do that. That's not what happens. They all get smushed into the asme list. OK so we should filter.
            # print(f"[DEBUG MajorityVoteStep save] Including details for key='{key}'.")
            if not entry.get(self.details_key):
                # print(f"[DEBUG MajorityVoteStep save] Initializing details list for key='{key}'.")
                entry[self.details_key] = []
            entry[self.details_key].append(
                {
                    "full_response": "...",  # Avoid printing large objects
                    "full_input": "...",  # Avoid printing large objects
                    "completion_mode": completion_mode,
                    "bool_result": result,
                }
            )
            # print(f"[DEBUG MajorityVoteStep save] Appended details. New details count for key='{key}': {len(entry[self.details_key])}.")

        input_dict[str(key)] = entry
        # print(f"[DEBUG MajorityVoteStep save] Saved entry for key='{key}'.")

        return entry

    def evaluate_final_count_and_save(
        self, current_votes, input_dict, input_data, output_dir, key, include_details
    ):
        # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Evaluating final count for key='{key}'. Current vote count: {len(current_votes)}.")
        true_count = sum(1 for vote in current_votes if vote)
        total_votes = len(current_votes)
        # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] True counts: {true_count}, Total votes: {total_votes}.")
        if total_votes == 0:
            # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] No votes found for key='{key}'. Returning False.")
            return False
        judgement = (true_count / total_votes) >= self.percent_true_to_pass
        # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Judgement for key='{key}': {judgement} (threshold={self.percent_true_to_pass}).")

        # Entry should exist in input_dict at this point, fetch directly
        entry = input_dict.get(str(key))
        if entry is None:
            # Fallback if key unexpectedly missing, log or handle error?
            # print(f"  WARNING: Key {key} unexpectedly missing from input_dict in evaluate_final_count_and_save. Creating new entry.")
            entry = {}  # Fallback to original logic if absolutely necessary
        else:
            # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Found existing entry for key='{key}'.")
            pass  # Explicitly do nothing if entry exists

        entry[self.final_determination_key] = judgement
        # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Set final determination '{self.final_determination_key}' for key='{key}' to {judgement}.")

        # filtered detail list
        if include_details:
            # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Filtering details for key='{key}' based on judgement ({judgement}).")
            # print("INCLUDING DETAILS -- MAJORITY VOTE")
            original_details_count = len(entry.get(self.details_key, []))
            filtered_details = [
                detail
                for detail in entry.get(self.details_key, [])
                if detail["bool_result"] == judgement
            ]  # the beauty is, it does not matter if we re-run. Because then all the details will still be of the same bool result that passed the first time. So the details list is not changed.
            # print([det["bool_result"] for det in filtered_details])
            # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Original details count: {original_details_count}, Filtered details count: {len(filtered_details)} for key='{key}'.")
            entry[self.details_key] = filtered_details

        input_dict[str(key)] = (
            entry  # where before we kept this up to date with the xisting data, now it IS the existing data. Much simpler eh?
        )
        # print(f"[DEBUG MajorityVoteStep evaluate_final_count_and_save] Updated entry in input_dict for key='{key}'.")

        return judgement

    async def run(
        self,
        key,
        input_data,
        engine_wrapper,
        input_dict,
        default_prompt_folder,
        prompt_folder,
        output_dir,
        include_details,
        completion_mode,
        use_stop,
        **kwargs,
    ):

        # print(f"[DEBUG MajorityVoteStep run] Starting run for key='{key}'.")
        full_prompt_path = (
            self.prompt_path + ".yaml"
            if not completion_mode
            else self.prompt_path + ".txt"
        )
        # print(f"[DEBUG MajorityVoteStep run] Prompt path: '{full_prompt_path}'")

        res, current_votes = await self.read_previous_output(key, input_dict)
        # print(f"[DEBUG MajorityVoteStep run] Read previous output for key='{key}'. Result: {res}, Current votes: {len(current_votes)}.")
        if res:
            # print(f"[DEBUG MajorityVoteStep run] Vote count already met for key='{key}'. Evaluating final count.")
            self.evaluate_final_count_and_save(
                current_votes=current_votes,
                input_dict=input_dict,
                input_data=input_data,
                output_dir=output_dir,
                key=key,
                include_details=include_details,
            )
            # print(f"[DEBUG MajorityVoteStep run] Finished run early for key='{key}' as vote count was met.")
            return  # Skip generation if votes are already sufficient

        processed_data, additional_kwargs = self.process_input_data(input_data)
        # print(f"[DEBUG MajorityVoteStep run] Processed input data for key='{key}'.")
        existing_variations = []

        # Check existing variations using key
        output_path = self.make_output_path(output_dir)
        # print(f"[DEBUG MajorityVoteStep run] Checking for existing variations in file: '{output_path}' for key='{key}'.")
        try:
            # print(f"[DEBUG MajorityVoteStep run] Acquired lock for '{output_path}.lock'. Reading file.")
            try:
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
                    existing_variations = existing_data.get(str(key), {}).get(
                        self.result_key, []
                    )
                    # print(f"[DEBUG MajorityVoteStep run] Loaded {len(existing_variations)} existing variations from file for key='{key}'.")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # print(f"[DEBUG MajorityVoteStep run] File '{output_path}' not found or JSON error for key='{key}': {e}. Starting with 0 existing variations.")
                pass  # No existing data is fine
        except Exception as e:
            print(
                f"[DEBUG MajorityVoteStep run] Error during file lock/read for '{output_path}': {e}"
            )
            # Decide if we should continue or raise

        # Generate remaining variations (it is in a loop we generate like random variations does)
        needed_generations = self.vote_count_needed - len(existing_variations)
        # print(f"[DEBUG MajorityVoteStep run] Need to generate {needed_generations} more votes for key='{key}' (Have {len(existing_variations)}, Need {self.vote_count_needed}).")
        for i in range(len(existing_variations), self.vote_count_needed):
            # print(f"[DEBUG MajorityVoteStep run] Starting generation loop iteration {i+1}/{self.vote_count_needed} for key='{key}'.")
            error_message = ""
            complete = False
            attempt = 0
            current_max_retries = self.max_retries
            while not complete and attempt < current_max_retries:
                attempt += 1
                # print(f"[DEBUG MajorityVoteStep run] Generation attempt {attempt}/{current_max_retries} for key='{key}', iteration {i+1}.")
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
                            **additional_kwargs,
                        )
                    )
                    # print(f"[DEBUG MajorityVoteStep run] generate_data returned for key='{key}'. Result: {result}.")
                    validation_result = self.validation_function(result, input_data)
                    if validation_result["result"]:
                        # print(f"[DEBUG MajorityVoteStep run] Validation function passed for key='{key}'.")
                        complete = True
                    else:
                        # print(f"[DEBUG MajorityVoteStep run] Validation function failed for key='{key}'.")
                        error_message = validation_result["message"]
                        pass  # Explicitly do nothing if validation fails
                except Exception as e:
                    # print(f"[DEBUG MajorityVoteStep run] Exception during generate_data for key='{key}', attempt {attempt}: {e}")
                    error_message = str(e)
                    traceback.print_exc()

            if not complete:
                # print(f"[DEBUG MajorityVoteStep run] Failed to generate a valid result after {current_max_retries} attempts for key='{key}'. Returning.")
                return  # Stop processing this key if generation failed

            # print(f"[DEBUG MajorityVoteStep run] Generation successful for key='{key}', iteration {i+1}. Saving result.")
            self.save(
                result=result,
                key=key,
                input_dict=input_dict,
                input_data=input_data,  # Passing input_data, but save doesn't seem to use it directly
                full_response=full_response,  # Passing potentially large object
                full_input=full_input,  # Passing potentially large object
                include_details=include_details,
                completion_mode=completion_mode,
            )
            # print(f"[DEBUG MajorityVoteStep run] Saved result for key='{key}', iteration {i+1}.")

        # print(f"[DEBUG MajorityVoteStep run] Finished generation loop for key='{key}'.")
        # Re-check current votes after potential generation
        res, current_votes = await self.read_previous_output(key, input_dict)
        # print(f"[DEBUG MajorityVoteStep run] Re-read previous output for key='{key}' after generation loop. Result: {res}, Current votes: {len(current_votes)}.")
        if res:  # Should be true now if loop completed successfully
            # print(f"[DEBUG MajorityVoteStep run] Evaluating final count after generation for key='{key}'.")
            self.evaluate_final_count_and_save(
                current_votes=current_votes,
                input_dict=input_dict,
                input_data=input_data,
                output_dir=output_dir,
                key=key,
                include_details=include_details,
            )
        else:
            # print(f"[DEBUG MajorityVoteStep run] WARNING: Vote count still not met after generation loop for key='{key}'. This shouldn't happen if loop completed.")
            pass  # Explicitly do nothing if vote count not met
        # print(f"[DEBUG MajorityVoteStep run] Finished run for key='{key}'.")

    # This way it makes it even easierto code pipelinesteps too! Just need to manipulte the dict, no files.

    async def execute_pipeline(
        self,
        engine_wrapper,
        rtwl,
        default_prompt_folder,
        prompt_folder,
        output_dir,
        completion_mode,
        use_stop,
        include_details,
        input_dict={},
        **kwargs,
    ):
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Starting execute_pipeline. Output dir: '{output_dir}'. Initial input_dict size: {len(input_dict)}.")
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Loading dataset...")
        self.load_dataset(input_dict=input_dict, output_dir=output_dir)
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Dataset loaded. input_dict size: {len(input_dict)}.")
        num_tasks = len(input_dict)
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Creating {num_tasks} run tasks.")
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
            # print(f"[DEBUG MajorityVoteStep execute_pipeline] Submitting {len(coroutines)} tasks to rate limited wrapper.")
            # Adding progress tracking
            TASK_TIMEOUT_SECONDS = 600  # 10 minutes timeout
            completed_tasks = 0
            num_tasks = len(coroutines)  # Use pre-calculated num_tasks
            for future in tqdmasyncio.tqdm.as_completed(coroutines):
                try:
                    # Wrap await future with timeout
                    await asyncio.wait_for(future, timeout=TASK_TIMEOUT_SECONDS)
                    completed_tasks += 1
                    # if completed_tasks % 100 == 0 or completed_tasks == num_tasks: # Print every 100 tasks or on the last task
                    #      print(f"[DEBUG MajorityVoteStep execute_pipeline] Completed {completed_tasks}/{num_tasks} tasks.")
                except asyncio.TimeoutError:
                    # completed_tasks might not be perfectly accurate here if timeout happens before increment
                    print(
                        f"\nWARNING: Task (approx {completed_tasks+1}/{num_tasks}) for step '{self.output_file}' timed out after {TASK_TIMEOUT_SECONDS} seconds.",
                        file=sys.stderr,
                    )
                    # Task is cancelled by wait_for, loop continues
                except Exception as task_exception:
                    # print(f"[DEBUG MajorityVoteStep execute_pipeline] Exception occurred in awaited task: {task_exception}")
                    traceback.print_exc()  # Print traceback for individual task failure

            # print(f"[DEBUG MajorityVoteStep execute_pipeline] All tasks completed.")
        except Exception as e:
            # print(f"[DEBUG MajorityVoteStep execute_pipeline] Exception occurred during task execution loop: {e}")
            traceback.print_exc()
            # raise e # Decide if outer exception should halt everything
        finally:
            # print(f"[DEBUG MajorityVoteStep execute_pipeline] Entering finally block.")
            # print(f"[DEBUG MajorityVoteStep execute_pipeline] Saving dataset. Current input_dict size: {len(input_dict)}.")
            self.save_dataset(input_dict=input_dict, output_dir=output_dir)
            # print(f"[DEBUG MajorityVoteStep execute_pipeline] Dataset saved.")

        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Filtering out non-present keys using key '{self.result_key}'. Current input_dict size: {len(input_dict)}.")
        filter_out_nonpresent_keys(input_dict, key_to_check=self.result_key)
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Filtering complete. Final input_dict size: {len(input_dict)}.")
        # print(f"[DEBUG MajorityVoteStep execute_pipeline] Finished execute_pipeline.")
