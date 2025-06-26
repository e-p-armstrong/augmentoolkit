import random
import traceback
from augmentoolkit.generation_functions.depth_first_pipeline_step_class import (
    DepthFirstPipelineStep,
    create_depth_first_executor,
)
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.generalized_parsing_and_writing_formats import (
    extract_structured_data,
)
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.run_pipeline_step import (
    create_depth_first_step_function,
)
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from generation.core_components.chunking import count_total_tokens, read_and_chunk_text
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from generation.core_pipelines.rptoolkit.rptoolkit_helpers import (
    count_tokens,
    create_generate_constrained_emotion_step,
    create_generate_story_pipeline_step,
    dict_to_string,
    extract_charname,
    generate_scene_card,
    is_story_awesome,
    is_story_ok,
    parse_string_to_dict,
    write_final_dataset_files,
    generate_emotion_pipeline_step,
    rate_story_step,
    create_archetype_step,
)
from tqdm import tqdm
from augmentoolkit.utils.observers import *

import nltk
from tqdm import asyncio as tqdmasyncio
import os
import sys
import time
import yaml

from redis_config import set_progress


async def generate_data(
    input_data,
    engine_wrapper: EngineWrapper,
    engine_wrapper_large: EngineWrapper,
    default_prompt_folder,
    prompt_folder,
    completion_mode,
    use_stop,
    key,
    pick_emotion,
    phase_index,
    work_in_phases,
    emotions,
    include_chunk_in_prompt,
    use_min_p,
    large_mode,
    input_dict,
    include_details=False,
    generate_archetype=False,
    archetypes=[""],
    extract_features_pipeline_step=None,
    output_path="",
):  # This is a PRIME example of a node. This'll let us test the node and composition plan for this whole thing.
    # NOTE Generate emotions, or pick
    # print("Started datagen")
    # print("DEBUG: INCLUDE DETAILS")
    # print(include_details)
    try:
        if pick_emotion:
            emotion_data = await generate_emotion_pipeline_step.run(
                input_data=input_data,
                engine_wrapper=engine_wrapper,
                default_prompt_folder=default_prompt_folder,
                prompt_folder=prompt_folder,
                input_dict=input_dict,
                key=key,
                completion_mode=completion_mode,
                use_stop=use_stop,
                include_details=include_details,
            )
            if emotion_data:
                emotion_data["emotion"] = emotion_data["emotion"].split("\n")[0]
            if not emotion_data:
                print(f"Emotion {key} failed checks.")
                return None, None, None
        else:
            constrained_emotion_step = create_generate_constrained_emotion_step(
                emotions=emotions
            )

            emotion_data = await constrained_emotion_step.run(
                input_data=input_data,
                engine_wrapper=engine_wrapper,
                input_dict=input_dict,
                key=key,
                default_prompt_folder=default_prompt_folder,
                prompt_folder=prompt_folder,
                completion_mode=completion_mode,
                use_stop=use_stop,
                include_details=include_details,
            )
            # print(data)
        archetype = emotion_data
        if generate_archetype:
            archetype = await create_archetype_step.run(
                input_data=emotion_data,
                engine_wrapper=engine_wrapper,
                input_dict=input_dict,
                key=key,
                default_prompt_folder=default_prompt_folder,
                prompt_folder=prompt_folder,
                completion_mode=completion_mode,
                use_stop=use_stop,
                include_details=include_details,
            )
        else:
            emotion_data["archetype"] = random.choice(archetypes)
            archetype = emotion_data

        features_data = await extract_features_pipeline_step.run(
            input_data=archetype,
            engine_wrapper=engine_wrapper,
            input_dict=input_dict,
            key=key,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=include_details,
        )

        scene_data = await generate_scene_card.run(
            input_data=features_data,
            engine_wrapper=engine_wrapper,
            input_dict=input_dict,
            key=key,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=include_details,
        )
        charname = extract_charname(scene_data["scene_card"])
        print("DEBUG: CHARNAME")
        print(charname)

        if phase_index == 0 and work_in_phases:
            return

        generate_story_pipeline_step = create_generate_story_pipeline_step(
            include_chunk_in_prompt=include_chunk_in_prompt,
            use_min_p=use_min_p,
            large_mode=large_mode,
        )

        outs = await generate_story_pipeline_step.run(
            input_data=scene_data,
            engine_wrapper=engine_wrapper_large,
            input_dict=input_dict,
            key=key,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=include_details,
        )

        if phase_index == 1 and work_in_phases:
            return

        out_story_ratings = await rate_story_step.run(
            input_data=outs,
            engine_wrapper=engine_wrapper,
            input_dict=input_dict,
            key=key,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=include_details,
        )
        # data["story_ratings"] = out_story_ratings
        out_story_ratings["id"] = key
        out_story_ratings["charname"] = charname
        # print("DEBUG--OUT STORY RATINGS")
        # print(out_story_ratings)
    except Exception as e:
        print(f"\n\n\nTHIS CHUNK STORYGEN FAILED -- EXCEPTION ENCOUNTERED: {e}")
        print("Cutting losses and moving on to the next chunk.")
        traceback.print_exc()


async def rptoolkit_inner_function(
    paragraphs_processed,
    engine_wrapper: EngineWrapper,
    engine_wrapper_large: EngineWrapper,
    pick_emotion,
    phase_index,
    work_in_phases,
    run_task_with_limit,
    default_prompt_folder,
    prompt_folder,
    output_dir,
    completion_mode,
    use_stop,
    emotions,
    include_chunk_in_prompt,
    use_min_p,
    large_mode,
    include_details=False,
    archetypes=[""],
    generate_archetype=False,
    extract_features_pipeline_step=None,
):
    stories = []
    data_generations_tasks = [
        generate_data(
            data=chunk,
            engine_wrapper=engine_wrapper,
            engine_wrapper_large=engine_wrapper_large,
            stories=stories,
            idx=idx,
            pick_emotion=pick_emotion,
            phase_index=phase_index,
            work_in_phases=work_in_phases,
            default_prompt_folder=default_prompt_folder,
            prompt_folder=prompt_folder,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            emotions=emotions,
            include_chunk_in_prompt=include_chunk_in_prompt,
            use_min_p=use_min_p,
            large_mode=large_mode,
            include_details=include_details,
            archetypes=archetypes,
            generate_archetype=generate_archetype,
            extract_features_pipeline_step=extract_features_pipeline_step,
        )
        for idx, chunk in enumerate(paragraphs_processed)
    ]
    coroutines = [run_task_with_limit(task) for task in data_generations_tasks]
    for future in tqdmasyncio.tqdm.as_completed(coroutines):
        await future
    return stories


# pick_emotion = bool(config["SYSTEM"]["pick_emotion"])
# work_in_phases = bool(config["PHASES"]["work_in_phases"])
# phase_index = int(config["PHASES"]["phase_index"])
# USE_SUBSET = bool(config["SYSTEM"]["USE_SUBSET"])
# SUBSET_SIZE = int(config["SYSTEM"]["SUBSET_SIZE"])
# CHUNK_SIZE = int(config["SYSTEM"]["CHUNK_SIZE"])
# USE_LIGHTNOVELCO = bool(config["SCRAPING"]["USE_LIGHTNOVELCO"])
# LNCO_BASE_URL = config["SCRAPING"]["LNCO_BASE_URL"]
# LNCO_RANKING_URL = config["SCRAPING"]["LNCO_RANKING_URL"]
# LNCO_CHAPTER_COUNT = int(config["SCRAPING"]["LNCO_CHAPTER_COUNT"])
# LNCO_NOVEL_COUNT = int(config["SCRAPING"]["LNCO_NOVEL_COUNT"])
# LNCO_WAIT_TIME = int(config["SCRAPING"]["LNCO_WAIT_TIME"])
# LNCO_MAX_WORKERS = int(config["SCRAPING"]["LNCO_MAX_WORKERS"])
# output_dir = os.path.abspath(obj_conf["PATH"]["OUTPUT"])
# INPUT = os.path.abspath(obj_conf["PATH"]["INPUT"])
# DEFAULT_PROMPT_PATH = os.path.abspath(obj_conf["PATH"]["DEFAULT_PROMPTS"])
# PROMPTS = os.path.abspath(obj_conf["PATH"]["PROMPTS"])
# COMPLETION_MODE = parse_bool(obj_conf["SYSTEM"]["COMPLETION_MODE"])
# LOGGING_LEVEL = logging.INFO
# LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
# LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
# API_KEY_A = obj_conf["API"]["API_KEY_A"]
# API_KEY_B = obj_conf["API"]["API_KEY_B"]
# BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
# BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
# MODE_A = obj_conf["SYSTEM"]["MODE_A"]
# MODE_B = obj_conf["SYSTEM"]["MODE_B"]
# CONCURRENCY_LIMIT = int(obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"])
# USE_STOP = parse_bool(obj_conf["SYSTEM"]["STOP"])
# EMOTIONS = parse_string_list.parse_string_list(obj_conf["SYSTEM"]["EMOTIONS"])
# INCLUDE_CHUNK_IN_PROMPT = parse_bool(obj_conf["SYSTEM"]["INCLUDE_CHUNK_IN_PROMPT"])
# USE_MIN_P = parse_bool(obj_conf["SYSTEM"]["USE_MIN_P"])


async def rptoolkit_pipeline(
    archetypes,
    generate_archetype,
    pick_emotion,
    work_in_phases,
    phase_index,
    use_subset,
    subset_size,
    chunk_size,
    output_dir,
    input_dir,
    default_prompts,
    prompts,
    completion_mode,
    small_model,
    large_model,
    small_api_key,
    large_api_key,
    small_base_url,
    large_base_url,
    small_mode,
    large_mode,
    concurrency_limit,
    use_stop,
    emotions,
    include_chunk_in_prompt,
    use_min_p,
    rp_prompt_start,
    rp_prompt_end,
    cost_per_million_small_input,
    cost_per_million_small_output,
    cost_per_million_large_input,
    cost_per_million_large_output,
    do_meta_datagen,
    meta_datagen_keys,
    meta_datagen_extras,
    to_include_features,
    chunking_output_dir=None,
    task_id=None,
    seed=1048596,
    **kwargs,
):
    # in the final datagen for the meta thing, I will go over each of the outputs with R1 and classify whether it actually follows the instructions to the letter. And only take the best of them.  This additioanl quality control will ensure only the best gets through.

    set_progress(
        task_id=task_id,
        progress=0.0,
        message="Pipeline starting; reading and chunking files",
    )

    # I will train the model for 2 epochs
    def parse_features(output):
        features_list_extraction_dict = {
            h: {"name": "list"} for h in to_include_features
        }
        try:
            features_obj = extract_structured_data(
                output, features_list_extraction_dict
            )
            # to_include = to_include_features
            # features_obj = parse_string_to_dict(features, to_include)
            # # remove keys that are not in the given to-include list
            # features_obj = {key: value for key, value in features_obj.items() if key in to_include}

            # print("\n\nFEATURES OBJECT POSTPROCESSING DEBUGGING:")
            # print(features_obj)
            if "genre tags" in features_obj:
                features_obj["genre tags"] = features_obj["genre tags"][
                    :10
                ]  # postprocessing to fix run-on generations
            # print(features_obj)
            return dict_to_string(features_obj)
        except Exception as e:
            print("\n\n!!!ERROR IN EXTRACTING FEATURES!")
            print(output)
            print(e)
            print("----------------")
            traceback.print_exc()
            return None

    extract_features_pipeline_step = DepthFirstPipelineStep(
        prompt_path="extract_features",
        output_processor=parse_features,
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "\n\n\n\n\n",
            ],
            "temperature": 0.8,
        },
        output_file="story_generation",
        result_key="features",
        details_key="features_details",
    )

    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

    # Create counter dictionaries instead of using integers
    # good point, I accidentally created a well-functioning thing because the dicts allow many different things in one variable
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
                )
            ],
            engine_output_observers=[
                create_log_observer(output_dir, do_meta_datagen),
                create_output_token_counter(
                    counter=small_token_counter,
                    cost_per_million=cost_per_million_small_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "small_model_tokens.json"
                    ),
                ),
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
                create_log_observer(output_dir, do_meta_datagen),
                create_output_token_counter(
                    counter=large_token_counter,
                    cost_per_million=cost_per_million_large_output,
                    count_tokens_fn=count_tokens,
                    persistence_path=os.path.join(
                        output_dir, "large_model_tokens.json"
                    ),
                ),
            ],
        )
    )

    if chunking_output_dir:
        paragraphs_processed = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=use_subset,
            subset_size=subset_size,
            output_dir=chunking_output_dir,
            seed=seed,
        )  # todo make this func take jsonl with "text" and json with "text" too!
    else:
        paragraphs_processed = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=use_subset,
            subset_size=subset_size,
            output_dir=output_dir,
            seed=seed,
        )  # todo make this func take jsonl with "text" and json with "text" too!
    print("Counting tokens...")
    total_tokens = count_total_tokens(paragraphs_processed)
    print(f"Tokens counted! We have {total_tokens} of them.")

    print("\n\nParagraphs have been processed and chunked.\n\n")
    if len(paragraphs_processed) > 0:
        print(f"First chunk: {paragraphs_processed[0]}\n\n")
    else:
        print("No paragraphs found.")
        sys.exit(1)

    print(
        "\n\n\n================== RPToolkit: Running the pipeline ==================\n\n\n"
    )
    print(
        "NOTE: if there are previously-existing generations, then they will be read. **The progress bar below shows how many complete stories have been generated**, not how far we are along the current step, because this pipeline is depth first so it does multiple different steps at once for different chunks. Basically: don't panic if the progress bar isn't moving. If you see more requests being made, and files being saved, then you're moving along just fine.\n\n\n"
    )

    print("Converting to dict...")
    hashed_paragraphs_processed = hash_input_list(paragraphs_processed)
    print("Converted!")

    total_items = len(paragraphs_processed)
    set_progress(
        task_id,
        progress=0.0,
        message=f"{total_items} files read and chunked! proceeding with LLM chunk filtering",
    )

    # NOTE Generate the data

    rptoolkit_executor = create_depth_first_executor(
        generate_data,
        output_dir=output_dir,
        output_file="story_generation.json",
        final_result_key="story",
    )

    story_data = await rptoolkit_executor(
        input_dict=hashed_paragraphs_processed,
        rtwl=run_task_with_limit,  # kwargs start
        engine_wrapper=engine_wrapper,
        engine_wrapper_large=engine_wrapper_large,
        completion_mode=completion_mode,
        use_stop=use_stop,
        emotions=emotions,
        include_chunk_in_prompt=include_chunk_in_prompt,
        use_min_p=use_min_p,
        phase_index=phase_index,
        pick_emotion=pick_emotion,
        work_in_phases=work_in_phases,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        large_mode=large_mode,
        include_details=do_meta_datagen,
        archetypes=archetypes,
        generate_archetype=generate_archetype,
        extract_features_pipeline_step=extract_features_pipeline_step,
        total_items=total_items,
        task_id=task_id,
    )

    # turn the dict into a list so that we can do things with it that are list operations
    story_data = [
        {**item, "key": key} for key, item, in hashed_paragraphs_processed.items()
    ]

    # story_data = []
    # data_generations_tasks = [generate_data(chunk=chunk, engine_wrapper=engine_wrapper, engine_wrapper_large=engine_wrapper_large, stories=story_data, idx=idx) for idx, chunk in enumerate(paragraphs_processed)]
    # coroutines = [run_task_with_limit(task) for task in data_generations_tasks]
    # for future in tqdmasyncio.tqdm.as_completed(coroutines):
    #     await future

    # TOTAL COSTS
    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    # filter out all those items with null story_ratings
    story_data = [story for story in story_data if story.get("story_ratings")]

    set_progress(
        task_id=task_id,
        progress=1.0,
        message="Saving final dataset to files based on grading",
    )

    if (
        phase_index == 2 and work_in_phases
    ) or not work_in_phases:  # TODO fix final saving
        # overall thing works
        minimally_ok_stories = [story for story in story_data if is_story_ok(story)]
        highly_rated_stories = [
            story for story in story_data if is_story_awesome(story)
        ]

        # NOTE Write the output to file using JSON
        os.makedirs(f"{output_dir}/final_outputs", exist_ok=True)
        write_final_dataset_files(
            story_data, "full_stories_list", output_dir, rp_prompt_start, rp_prompt_end
        )
        write_final_dataset_files(
            minimally_ok_stories,
            "good_and_above_stories_list",
            output_dir,
            rp_prompt_start,
            rp_prompt_end,
        )
        write_final_dataset_files(
            highly_rated_stories,
            "incredible_stories_list",
            output_dir,
            rp_prompt_start,
            rp_prompt_end,
        )

        print(
            "\n\n\n================== ALL DATA WRITTEN!! HERE ARE YOUR STATS: ==================\n"
        )
        print(f"Total stories generated: {len(story_data)}")
        print(
            f"Stories that are at least OK across the board, but might slightly flawed ('good' and above, according to the AI rater): {len(minimally_ok_stories)}"
        )
        print(
            f"Stories that are highly rated by the AI across the board ('incredible' and above, according to the AI rater.): {len(highly_rated_stories)}"
        )
        total_tokens_of_stories = sum(
            [count_tokens(story["story"]) for story in story_data]
        )
        print(
            "Total tokens of all stories (roughly equivalent to the number of training tokens): ",
            total_tokens_of_stories,
        )
        print(
            "ShareGPT-format .json export is created, and the full dataset is also available in the final_outputs folder."
        )
        if len(story_data) == 0:
            print(
                "Hmm... No stories were generated. Check the logs for more information, and consider creating an issue if this is unexpected. If you do make an issue, please include your input data and the logs!"
            )
        else:
            print("Enjoy training your model!")
        print(
            "\n\n\n=============================================================================\n\n\n"
        )

    # make a dict out of story data. Key = index of each story. Value = story data.
    story_data_dict = {
        str(idx): story for idx, story in enumerate(highly_rated_stories)
    }
    if do_meta_datagen:
        print("entered here")
        create_meta_dataset(
            data_dicts=[
                story_data_dict
            ],  # is the filtering something that should be done by the input processor, or by the meta pipeline?
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete")
