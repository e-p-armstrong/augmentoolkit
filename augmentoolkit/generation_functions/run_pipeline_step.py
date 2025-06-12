from tqdm import asyncio as tqdmasyncio

from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.generation_functions.random_variation_step_class import (
    RandomVariationStep,
)
from augmentoolkit.generation_functions.depth_first_pipeline_step_class import (
    DepthFirstPipelineStep,
)


def create_random_variation_step_function(pipeline_kwargs={}):
    # Extract special parameters first
    method_overrides = pipeline_kwargs.pop("method_overrides", {})

    # Create pipeline step with remaining kwargs
    pipeline_step = RandomVariationStep(
        method_overrides=method_overrides, **pipeline_kwargs
    )
    print("entered this function")

    async def execute_variation_step(
        chunk,
        engine_wrapper,
        idx,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=None,
        use_stop=None,
        output_list=None,
        variation_generator_count=None,
    ):
        for i in range(variation_generator_count):
            await pipeline_step.run(
                input_data=chunk,
                engine_wrapper=engine_wrapper,
                idx=idx,
                output_list=output_list,
                default_prompt_folder=default_prompt_folder,
                prompt_folder=prompt_folder,
                output_dir=output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
            )

    async def execute_variation_pipeline(
        input_list,
        engine_wrapper,
        rtwl,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=None,
        use_stop=None,
        variation_generator_count=None,
    ):
        print("running")
        output_list = []
        data_generations_tasks = [
            execute_variation_step(
                chunk=chunk,
                engine_wrapper=engine_wrapper,
                idx=idx,
                output_list=output_list,
                default_prompt_folder=default_prompt_folder,
                prompt_folder=prompt_folder,
                output_dir=output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
                variation_generator_count=variation_generator_count,
            )
            for idx, chunk in enumerate(input_list)
        ]
        coroutines = [rtwl(task) for task in data_generations_tasks]
        for future in tqdmasyncio.tqdm.as_completed(coroutines):
            print("awaiting future")
            await future
        return output_list

    return execute_variation_pipeline


def create_depth_first_step_function(pipeline_kwargs={}):
    # Extract method overrides first
    method_overrides = pipeline_kwargs.pop("method_overrides", {})

    # Create pipeline step with remaining kwargs
    pipeline_step = DepthFirstPipelineStep(
        method_overrides=method_overrides, **pipeline_kwargs
    )

    # The value add of this is 1. it is a function 2. it has already loaded the proper things into the class and well no actually that part is pointless BUT it lets us wrap things around the "run" if we want to and maintains similarity with the rest

    async def execute_pipeline(
        data,
        engine_wrapper,
        idx,
        default_prompt_folder=None,
        prompt_folder=None,
        output_dir=None,
        completion_mode=None,
        use_stop=None,
        **kwargs
    ):
        ret_val = await pipeline_step.run(
            data=data,
            engine_wrapper=engine_wrapper,
            idx=idx,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            **kwargs
        )
        return ret_val

    return execute_pipeline  # basically .run() with data, engine_wrapper, idx, default prompt folder, prompt folder, output dir, completion mode, and use stop


def wrap_async_func_in_awaited_execution(async_generation_func):
    # Extract method overrides first
    async def execute_pipeline(input_list, engine_wrapper, rtwl, **kwargs):
        output_list = []
        data_generations_tasks = [
            async_generation_func(
                input_data=chunk,
                engine_wrapper=engine_wrapper,
                idx=idx,
                output_list=output_list,
                **kwargs
            )
            for idx, chunk in enumerate(input_list)
        ]
        coroutines = [rtwl(task) for task in data_generations_tasks]
        for future in tqdmasyncio.tqdm.as_completed(coroutines):
            await future
        return output_list

    return execute_pipeline
