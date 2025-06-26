from tqdm import asyncio as tqdmasyncio
import json
import os
import traceback
from augmentoolkit.generation_functions.one_to_many_step import (
    filter_out_nonpresent_keys,
)
from augmentoolkit.generation_functions.pipeline_step_class import (
    PipelineStep,
    load_dataset_func,
    save_dataset,
)
from redis_config import set_progress


# TODO this has no execute pipeline step so how TF do I do save once?
class DepthFirstPipelineStep(PipelineStep):
    def read_previous_output(self, key, input_dict):
        entry = input_dict.get(str(key), {})
        if self.result_key in entry:
            return entry
        return False

    def save(
        self,
        result,
        key,
        input_data,
        output_dict,
        include_details,
        full_response,
        full_input,
        completion_mode,
        **kwargs,
    ):

        entry = input_data
        entry[self.result_key] = result
        if include_details:
            # print("INCLUDING DETAILS")
            entry[self.details_key] = [
                {
                    "full_response": full_response,
                    "full_input": full_input,
                    "completion_mode": completion_mode,
                }
            ]

        output_dict[str(key)] = entry
        return entry

    async def run(
        self,
        key,
        input_data,
        engine_wrapper,
        default_prompt_folder,
        prompt_folder,
        completion_mode,
        use_stop,
        include_details,
        input_dict,
        **kwargs,
    ):
        full_prompt_path = (
            self.prompt_path + ".yaml"
            if not completion_mode
            else self.prompt_path + ".txt"
        )

        previous = self.read_previous_output(key, input_dict)
        if previous:
            return previous

        processed_data, additional_kwargs = self.process_input_data(input_data)

        error_message = ""

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
                        **additional_kwargs,
                    )
                )

                validation_result = self.validation_function(result, input_data)
                if validation_result["result"]:
                    complete = True
                else:
                    max_retries -= 1
                    error_message = validation_result["message"]
            except Exception as e:
                print(e)
                error_message = str(e)
                traceback.print_exc()
                max_retries -= 1
        if not complete:
            return

        return self.save(
            result=result,
            key=key,
            input_data=input_data,
            full_output=full_output,
            include_details=include_details,
            full_response=full_response,
            full_input=full_input,
            completion_mode=completion_mode,
            output_dict=input_dict,
        )  # we want the custom model to produce truly profound works when doing creative data. A good cold start. And something skilled.
        # maybe increase the size?
        # Nah... right?
        # Well yes having size variations would be nice
        # need multiple hyperparams but
        # Well to be fair to get writing like I was thinking of I'd have to pretrain on a bunch of absolute classics then SFT+RL from there
        # might be a bit antiquated
        # But possible as a fun project


# but how to guarntee that all the right things will be writen to?
# you pass the input dict in. The full thing. And it goes into the pipeline step .runs and then gets drilled and we do all the right things.

# args ot main func are code time-args to executor are runtime

# This is a handy wrapper for your function that calls all your depth first things. Think of it like execute pipeline except useful in this way.


# takes a task id and total number of items. If you have multiple depth first executors in the same pipeline and want to track progress not just in that one executor, then set task id to None or don't pass it in.
def create_depth_first_executor(
    composition_func, output_dir, output_file, final_result_key
):  # takes an async function that chains all your depth first pipeline step calls together. And wraps it in a try-except-finally that saves and loads the dataset at start and exit. Viola?
    assert isinstance(output_file, str), "output file must be str"
    assert output_file.endswith(".json"), "output file mus be json"
    assert isinstance(final_result_key, str), "final result key must be str"

    counter = 0

    output_path = os.path.join(output_dir, output_file)  # output file must be .json

    async def executor(input_dict, rtwl, task_id=None, total_items=0, **kwargs):
        load_dataset_func(
            input_dict=input_dict, output_path=output_path
        )  # if it is at the start and end of every pipeline execution, we get the same behavior as before, past thing will load and save just fine, without constant reads/writes.
        # depth first funcs must also now take key as an arg
        nonlocal counter # Ensure we modify the outer counter
        # Initialize counter based on already completed items
        completed_keys = {key for key, value in input_dict.items() if isinstance(value, dict) and final_result_key in value}
        counter = len(completed_keys)

        try:
            data_generations_tasks = [
                composition_func(
                    key=key,
                    input_data=value,
                    output_path=output_path,  # not actually used in the pipeline steps themselves, but in case we need it, here it is.
                    input_dict=input_dict,
                    **kwargs,
                )
                for key, value in input_dict.items()
            ]
            coroutines = [rtwl(task) for task in data_generations_tasks]
            
            # Use tqdm for progress bar
            progress_bar = tqdmasyncio.tqdm(total=total_items, initial=counter, desc=f"Executing {output_file}")

            for future in tqdmasyncio.tqdm.as_completed(coroutines):
                result = await future
                # Increment counter only if the task was successful and added the final key
                if result and isinstance(result, dict) and final_result_key in result:
                    if not any(isinstance(v, dict) and v.get(final_result_key) == result[final_result_key] for k, v in completed_keys):
                        counter += 1
                        progress_bar.update(1)
                        if task_id:
                            set_progress(
                                task_id,
                                progress=counter / total_items if total_items > 0 else 0,
                                message=f"{counter} out of {total_items} processed!",
                            )

            progress_bar.close()

        except Exception as e:
            print(f"Exception occurred during task execution: {e}")
            traceback.print_exc()
            # We still want to save progress in the finally block, so we don't re-raise here
            # but you could if you want the whole pipeline to stop.
        finally:
            # Print count before filtering
            print(f"Items in dictionary before final save: {len(input_dict)}")

            # FIX: Filter out incomplete entries BEFORE the final save
            filter_out_nonpresent_keys(input_dict, key_to_check=final_result_key)
            
            # Print count after filtering
            print(f"Items in dictionary after filtering (to be saved): {len(input_dict)}")

            # Robust save with interrupt handling
            save_dataset(input_dict=input_dict, output_path=output_path)

        return input_dict
        
    return executor