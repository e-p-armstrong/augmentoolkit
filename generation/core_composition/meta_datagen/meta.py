# the whole composition

import json
import os
import random
import shutil
import traceback
import tracemalloc  # Added import
import gc  # Added import
import yaml  # Added import
import glob  # Added import
from generation.core_components.chunking import read_jsonl_completions, read_text
from generation.core_components.data_prep_operations import (
    completionify_sharegpt,
    count_tokens_glob,
    create_subset,
    route_template_to_preset,
    save_hf_dataset,
)
from generation.core_components.setup_components import make_relative_to_self
from generation.core_components.sharegpt_operations import combine_single_turn_convs
from generation.core_pipelines.clean_and_convert_pdf.clean_and_convert_pdf import (
    pdf_clean_and_convert_pipeline,
)
from generation.core_pipelines.correction_pipeline.corrections import (
    correction_pipeline,
)
from generation.core_pipelines.factual_generation_individual.factual_generation import (
    generate_factual_qa_dataset,
)
from generation.core_pipelines.generic_data_rephrase.transform_generic_data import (
    transform_generic_data_pipeline,
)
from generation.core_pipelines.recall_multiple_sources.multi_source_recall import (
    generate_multi_source_dataset,
)
from generation.core_pipelines.representation_variation.repvar import (
    representation_variation_pipeline,
)
from generation.core_pipelines.rptoolkit.rptoolkit import rptoolkit_pipeline
from generation.core_pipelines.train_a_model_to_do_rag.rag_data_pipeline import (
    rag_data_pipeline,
)
from generation.core_components.write_config_files import (
    write_training_config,
    create_completion_dataset,
    create_input_output_dataset,
)
from transformers import AutoTokenizer
from generation.core_components.chunking import read_jsonl_completions, read_text
from redis_config import set_progress


def create_meta_segment_convs(obj, tokenizer):
    """
    Takes an input like
    ```
    {
        "full_input": [ ... ], // List of message dicts (system, user, assistant)
        "full_response": "..." // String for the final assistant response
    }
    ```
    and makes a new output list, segments, with a label key and a text key.
    label is true if the text is visible during training and false if not.
    Here are the rules for being visible:
    1. if there is a system message it is label: true
    2. every user-assistant message pair after that point is label: false
    3. the final user message is label: true
    4. the full_response (the text is known; the role is assumed to be assistant) is label: true
    5. the text has to be formatted using tokenizer.apply_chat_template.

    Hint: you need to apply the chat template to get all the prompt template bits of the actual response.
    But once you apply a chat template obviously you have a string, which is difficult to create segments with labels from.
    However you can get structure by splitting at the message contents themselves (or using iterative rendering).

    Returns:
    {
        "segments": [
            {
                "label": true|false
                "text": str
            },
            ...
        ]
    }
    or None if processing fails.
    """

    # Helper function to truncate strings for cleaner logging
    def truncate_str(s, max_len=150):
        if isinstance(s, str) and len(s) > max_len:
            return s[:max_len] + "..."
        return s

    # Helper function to truncate strings within nested structures (lists/dicts) for printing
    def truncate_obj_repr(item, max_len=150):
        if isinstance(item, dict):
            return {k: truncate_obj_repr(v, max_len) for k, v in item.items()}
        elif isinstance(item, list):
            # Truncate the list representation itself if too long, plus its elements
            list_repr = [truncate_obj_repr(elem, max_len) for elem in item]
            str_repr = str(list_repr)
            if len(str_repr) > max_len * 2:  # Allow more space for list structure
                # Represent very long lists concisely
                return f"[{truncate_obj_repr(item[0], max_len)}, ..., {truncate_obj_repr(item[-1], max_len)}] (len={len(item)})"
            return list_repr
        elif isinstance(item, str):
            return truncate_str(item, max_len)
        else:
            return item

    # print(f"\n--- Entering create_meta_segment_convs ---")
    # Use json.dumps for potentially large/nested objects for controlled, truncated output
    try:
        import json

        obj_repr = json.dumps(
            truncate_obj_repr(obj), default=str
        )  # Use default=str for non-serializable
    except:  # Fallback if json fails
        obj_repr = truncate_str(str(obj), 500)
    # print(f"Input obj: {obj_repr}")

    # print(f"Tokenizer type: {type(tokenizer)}")
    # print(f"Tokenizer name/path: {getattr(tokenizer, 'name_or_path', 'N/A')}")

    try:
        messages = obj.get("full_input", [])
        response = obj.get("full_response", "")
        # Use helper for potentially long lists/dicts
        # print(f"Extracted messages (len={len(messages)}): {truncate_str(str(truncate_obj_repr(messages)), 500)}")
        # Use helper for potentially long strings
        # print(f"Extracted response: '{truncate_str(response)}'")

        if not messages or not response:
            # print(">>> Warning: Missing 'full_input' or 'full_response' in object. Returning None.")
            # print(f"Problematic obj: {truncate_str(str(truncate_obj_repr(obj)), 500)}")
            return None

        # Construct the full conversation including the final response
        full_conversation = messages + [{"role": "assistant", "content": response}]
        # print(f"Constructed full_conversation (len={len(full_conversation)}): {truncate_str(str(truncate_obj_repr(full_conversation)), 500)}")

        segments = []
        last_rendered = ""
        bos_segment = ""

        # Check for potential BOS token by rendering an empty conversation
        # print("\nAttempting BOS token check...")
        if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
            bos_segment = tokenizer.bos_token
            segments.append({"label": True, "text": bos_segment})
            last_rendered = bos_segment
        else:
            # print("No BOS token found in tokenizer.")
            bos_segment = ""

        # Use helper for potentially long strings
        # print(f"Initial last_rendered: '{truncate_str(last_rendered)}'")

        has_system = False
        if full_conversation and full_conversation[0]["role"] == "system":
            has_system = True
        # print(f"Conversation has system message: {has_system}")

        for i, message in enumerate(full_conversation):
            # print(f"\n--- Processing message index {i} ---")
            # Use helper for potentially large dicts
            # print(f"Current message: {truncate_str(str(truncate_obj_repr(message)), 500)}")

            # Render the conversation up to the current message
            conversation_slice = full_conversation[: i + 1]
            # Use helper for potentially large lists
            # print(f"Applying template to conversation slice (len={len(conversation_slice)}): {truncate_str(str(truncate_obj_repr(conversation_slice)), 500)}")
            current_rendered = ""
            try:
                current_rendered = tokenizer.apply_chat_template(
                    conversation_slice, tokenize=False, add_generation_prompt=False
                )
                # Use helper for potentially long strings
                # print(f"Rendered template output (type={type(current_rendered)}): '{truncate_str(current_rendered)}'")
            except Exception as e:
                # print(f">>> Error applying chat template at step {i}. Error type: {type(e).__name__}, Error: {e}")
                # import traceback # Removed
                # print("Traceback for template application error:")
                traceback.print_exc()
                # Use helper for potentially large lists
                # print(f"Failed conversation slice was: {truncate_str(str(truncate_obj_repr(conversation_slice)), 500)}")
                return None  # Cannot proceed if template application fails

            # Extract the text added by this message
            # Use helper for potentially long strings in the debug print
            # print(f"Calculating segment text. current_rendered='{truncate_str(current_rendered)}', last_rendered='{truncate_str(last_rendered)}', bos_segment='{truncate_str(bos_segment)}', i={i}")
            segment_text = ""
            # Handle potential prefix overlap if BOS wasn't rendered initially
            if (
                i == 0
                and not bos_segment
                and current_rendered.startswith(last_rendered)
            ):
                # Calculate with *untruncated* strings
                segment_text = current_rendered[len(last_rendered) :]
                # Use helper for potentially long strings
                # print(f"Segment text calculation (case 1): '{truncate_str(segment_text)}'")
            elif current_rendered.startswith(last_rendered):
                # Calculate with *untruncated* strings
                segment_text = current_rendered[len(last_rendered) :]
                # Use helper for potentially long strings
                # print(f"Segment text calculation (case 2): '{truncate_str(segment_text)}'")
            else:
                # This might happen if the template fundamentally changes with more messages
                # Or if last_rendered was empty and current_rendered is not
                # print(f"!!! Warning: Unexpected rendering change at step {i}. Using full current render as fallback.")
                segment_text = current_rendered  # Less ideal, but fallback
                # Use helper for potentially long strings
                # print(f"Segment text calculation (case 3 - fallback): '{truncate_str(segment_text)}'")

            if not segment_text:
                # Skip empty segments which might occur due to template logic
                # print(f"Segment text is empty. Skipping append. Updating last_rendered.")
                # Update with *untruncated* string
                last_rendered = current_rendered
                continue

            # Determine the label based on the rules
            is_final_assistant = i == len(full_conversation) - 1
            is_final_user = i == len(full_conversation) - 2
            # Corrected system check: checks role of the *current* message if it's the *first* message
            is_system = i == 0 and message["role"] == "system"
            # print(f"Label calculation: is_final_assistant={is_final_assistant}, is_final_user={is_final_user}, is_system={is_system}, has_system_flag={has_system}")

            if is_system or is_final_user or is_final_assistant:
                label = True
                # print("Label set to True.")
            else:
                # All other messages (user-assistant pairs acting as examples)
                label = False
                # print("Label set to False.")

            # Use helper for potentially long strings in the debug print
            # print(f"Appending segment: {{'label': {label}, 'text': '{truncate_str(segment_text)}'}}")
            # Append the *untruncated* string
            segments.append({"label": label, "text": segment_text})
            # Update with *untruncated* string
            last_rendered = current_rendered
            # Use helper for potentially long lists in the debug print
            # print(f"Updated last_rendered: '{truncate_str(last_rendered)}'")
            # print(f"Current segments list (len={len(segments)}): {truncate_str(str(truncate_obj_repr(segments)), 500)}")

        if not segments:
            # print(">>> Warning: No segments generated for conversation. Returning None.")
            # Use helper for potentially large lists
            # print(f"Full conversation that yielded no segments: {truncate_str(str(truncate_obj_repr(full_conversation)), 500)}")
            return None

        # print(f"--- Successfully generated segments (count={len(segments)}) ---")
        # Use helper for potentially large lists
        # print(f"Final segments: {truncate_str(str(truncate_obj_repr(segments)), 500)}")
        return {"segments": segments}

    except Exception as e:
        # print(f">>> Error creating meta segments (outer catch block). Error type: {type(e).__name__}, Error: {e}")
        # import traceback # Removed
        # print("Traceback for outer error:")
        traceback.print_exc()
        # Use helper for potentially large objects
        # print(f"Input obj at time of error: {truncate_str(str(truncate_obj_repr(obj)), 500)}")
        return None


# ... existing code ...
# rp_input = _sample_tracked_item(this_pipeline_random, cfg["inputs"], rp_sampling_state, cfg_input_key)


def _sample_tracked_item(rng, item_list, tracking_dict, list_key):
    """
    Samples an item from item_list using tracking to ensure all items are
    sampled once before repeats within a 'round'.

    Args:
        rng: The random.Random instance to use.
        item_list: The list to sample from.
        tracking_dict: A dictionary holding the state. Must contain list_key.
                       tracking_dict[list_key] is the list of remaining indices.
        list_key: The key in tracking_dict for the current list's state.

    Returns:
        A randomly selected item from item_list, respecting the tracking.
    """
    if not item_list:
        return None  # Or raise error, depending on desired behavior

    if not tracking_dict.get(list_key):
        # Refill indices if empty or key doesn't exist
        tracking_dict[list_key] = list(range(len(item_list)))
        # Shuffle to ensure randomness even when refilling mid-run
        rng.shuffle(tracking_dict[list_key])

    # Pop a random index from the remaining list
    chosen_index = tracking_dict[list_key].pop(
        rng.randrange(len(tracking_dict[list_key]))
    )

    return item_list[chosen_index]


async def meta_datagen_compositionpipeline(
    use_subset: bool,
    chunk_size: int,
    output_dir: str,
    concurrency_limit: int,
    seed: int,
    base_model_name: str,
    prob_zero_pairs: float,
    shared_chunking_output_dir: str,
    pdf_number_of_times_to_sample: int,
    pdf_cleaning_valid_input_paths: list[str],
    pdf_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    pdf_cleaning_model: list[dict[str, str]],
    pdf_cleaning_prompts: list[str],
    pdf_cleaning_chunk_size: int,
    pdf_cleaning_subset: int,
    repvar_times_to_sample: int,
    repvar_valid_input_paths: list[str],
    repvar_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    repvar_model: list[dict[str, str]],
    repvar_chunk_size: list[int],
    repvar_subset: list[dict[str, bool]],
    repvar_variation_count: list[int],
    factual_generation_times_to_sample: int,
    factual_generation_inputs_and_conversation_instructions: list[dict[str, str]],
    factual_generation_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    factual_generation_model: list[dict[str, str]],
    double_check_counter: int,
    factual_prompts_and_skips: list[dict[str, bool]],
    factual_chunk_size: list[int],
    factual_subset: list[int],
    correction_time_to_sample: int,
    correction_model: list[dict[str, str]],
    correction_prompts: list[str],
    correction_chunk_sizes: list[int],
    correction_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    correction_inputs: list[str],
    correction_subset_size: int,
    rag_data_time_to_sample: int,
    rag_data_model: list[dict[str, str]],
    rag_data_prompts: list[str],
    rag_data_chunk_sizes: list[int],
    rag_data_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    rag_failure_percentage: list[float],
    rag_max_chunks: list[int],
    rag_data_inputs: list[str],
    rag_subset_size: int,
    multi_source_times_to_sample: int,
    multi_source_inputs_and_conversation_instructions: list[str],
    multi_source_cleaning_completion_mode_and_use_stop: list[dict[str, bool]],
    multi_source_model: list[dict[str, str]],
    multi_source_prompts: list[str],
    multi_source_chunk_size: list[int],
    multi_source_subset: list[int],
    # TODO add multi-source factual generation and generic data transformation
    generic_thoughts_times_to_sample: int,
    generic_thoughts_model: list[dict[str, str]],
    generic_thoughts_inputs: list[str],
    generic_thoughts_prompts: list[str],
    generic_thoughts_completion_and_use_stop: list[dict[str, bool]],
    generic_thoughts_subset_size: int,
    roleplaying_times_to_sample: int,
    prompt_input_settings_config: list[dict],
    rp_chunk_sizes: list[int],
    rp_completion_mode_and_use_stop: list[dict[str, bool]],
    rp_subset_size: int,
    to_obfuscate_prompts_dir,
    task_id=None,  # task ID must always be specified as a pipeline arg. And it must be a kwarg, not an arg.
    **kwargs,
):
    tracemalloc.start()  # Start memory tracking

    to_obfuscate_prompts_dir = make_relative_to_self(to_obfuscate_prompts_dir)

    # Load obfuscation prompts
    obfuscation_map = {}
    if os.path.isdir(to_obfuscate_prompts_dir):
        print(f"Loading obfuscation prompts from: {to_obfuscate_prompts_dir}")
        yaml_files = glob.glob(os.path.join(to_obfuscate_prompts_dir, "*.yaml"))
        print(f"Found {len(yaml_files)} YAML files for obfuscation.")
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                    if (
                        isinstance(data, dict)
                        and "prompt" in data
                        and "replacement" in data
                    ):
                        obfuscation_map[data["prompt"].strip()] = data[
                            "replacement"
                        ].strip()
                        print(
                            f"  Loaded obfuscation rule from {os.path.basename(yaml_file)}"
                        )
                    else:
                        print(
                            f"  WARNING: Invalid format in {os.path.basename(yaml_file)}. Skipping."
                        )
            except Exception as e:
                print(f"  ERROR loading {os.path.basename(yaml_file)}: {e}")
        print(f"Total obfuscation rules loaded: {len(obfuscation_map)}")
    else:
        print(
            f"WARNING: Obfuscation directory not found or not a directory: {to_obfuscate_prompts_dir}"
        )

    print("\n\nOBFUSCATION MAP")
    print(obfuscation_map)

    import traceback  # Explicit import for scope clarity

    # We want to call each pipeline with varying settings. So the natural thing to do is to have a bunch of functions that create argument lists for these things, and these functions wilkl take kwargs that are a list of valid inputs for each arg that can actually vary. Maybe some of these will depend on each other...
    # Oh wait. This is literally. Schema datagen.
    # well no because.. it's a similar thing but it is not quite that, and we will build our own thing.

    # First pipeline we're making data from: the pdf cleaning pipeline.

    # we don't need to define any fancy inter-field relation rules here. If two things rely on each other ,group them in a selection list and that's it. Like a list of dicts

    # TODO we'll eventually want CoT on the single-turn prompts... maybe on the others too... hmmm idk. WEll what I am saying is, maybe cold start on some CoT things on pipelines that are otherwise multiturn, but the CoT is done in a single turn setting, if tha makes snse. Poorly prased but whatever I am not spending the mental energy to refine that statement.

    this_pipeline_random = random.Random()
    this_pipeline_random.seed(seed)

    # first, we randomly selectthings for each field that is a list
    set_progress(
        task_id=task_id, progress=1 / 8, message="Starting RP Dataset Generation!"
    )
    # --- Roleplaying Pipeline State ---
    rp_sampling_state = {}
    rp_list_keys = {
        "configs": "prompt_input_settings_config",
        "inputs": "rp_inputs",  # We'll track inputs inside the chosen config
        "models": "rp_models",  # We'll track models inside the chosen config
        "chunks": "rp_chunk_sizes",
        "completion": "rp_completion_mode_and_use_stop",
    }

    for i in range(roleplaying_times_to_sample):
        # --- Sampling (MUST happen before skip check) ---

        # Sample top-level config
        cfg = _sample_tracked_item(
            this_pipeline_random,
            prompt_input_settings_config,
            rp_sampling_state,
            rp_list_keys["configs"],
        )

        # Initialize tracking for nested lists if this config is new in this round
        # Note: This nested tracking might reset more often than strictly necessary
        # if the same cfg is picked again *after* a full round, but maintains correctness.
        # A more complex state could track indices *per* cfg, but adds complexity.
        cfg_input_key = (
            f"{rp_list_keys['inputs']}_{prompt_input_settings_config.index(cfg)}"
        )
        cfg_model_key = (
            f"{rp_list_keys['models']}_{prompt_input_settings_config.index(cfg)}"
        )

        if (
            cfg_input_key not in rp_sampling_state
            or not rp_sampling_state[cfg_input_key]
        ):
            rp_sampling_state[cfg_input_key] = list(range(len(cfg["inputs"])))
            this_pipeline_random.shuffle(rp_sampling_state[cfg_input_key])

        if (
            cfg_model_key not in rp_sampling_state
            or not rp_sampling_state[cfg_model_key]
        ):
            rp_sampling_state[cfg_model_key] = list(range(len(cfg["models"])))
            this_pipeline_random.shuffle(rp_sampling_state[cfg_model_key])

        # Sample nested items using their specific keys
        rp_input = _sample_tracked_item(
            this_pipeline_random, cfg["inputs"], rp_sampling_state, cfg_input_key
        )
        model_config = _sample_tracked_item(
            this_pipeline_random, cfg["models"], rp_sampling_state, cfg_model_key
        )

        # Sample other parameters using tracked sampling
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            rp_chunk_sizes,
            rp_sampling_state,
            rp_list_keys["chunks"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            rp_completion_mode_and_use_stop,
            rp_sampling_state,
            rp_list_keys["completion"],
        )

        # Extract values from sampled dicts
        generate_archetype = cfg["generate_archetype"]
        archetypes = cfg["archetypes"]
        prompts_path = cfg["prompts"]
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        to_include_features = cfg["to_include_features"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]

        # --- Calculate Output Dir (after sampling, before potential skip) ---
        this_run_output_dir = os.path.join(output_dir, "roleplaying", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Roleplaying Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Roleplaying Pipeline Run {i}")
        print("=" * 80)
        print(f"Input Path:         {rp_input}")
        print(f"Prompts Path:       {prompts_path}")
        print(f"Generate Archetype: {generate_archetype}")
        print(f"Completion Mode:    {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Chunk Size:         {chunk_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print(f"Features to Include: {to_include_features}")
        print
        print(f"Output Directory:   {this_run_output_dir}")
        print("=" * 80 + "\n")

        # TODO: Implement the roleplaying pipeline call here
        print(f"RP Run {i}: Taking snapshot before pipeline call...")
        snap_before_rp = tracemalloc.take_snapshot()
        # try:
        #     _ = await rptoolkit_pipeline(
        #         input_dir=rp_input,
        #         default_prompts=prompts_path,
        #         output_dir=this_run_output_dir,
        #         completion_mode=completion_mode,
        #         use_stop=use_stop,
        #         small_model=small_model,
        #         small_api_key=small_api_key,
        #         small_base_url=small_base_url,
        #         small_mode=small_mode,
        #         large_model=large_model,
        #         large_api_key=large_api_key,
        #         large_base_url=large_base_url,
        #         large_mode=large_mode,
        #         cost_per_million_small_input=cost_per_million_small_input,
        #         cost_per_million_small_output=cost_per_million_small_output,
        #         cost_per_million_large_input=cost_per_million_large_input,
        #         cost_per_million_large_output=cost_per_million_large_output,
        #         prompts=prompts_path,
        #         generate_archetype=generate_archetype,
        #         archetypes=archetypes,
        #         chunk_size=chunk_size,
        #         use_subset=True,
        #         subset_size=rp_subset_size,
        #         concurrency_limit=concurrency_limit,
        #         pick_emotion=True,
        #         work_in_phases=False,
        #         to_include_features=to_include_features,
        #         do_meta_datagen=True,

        #         phase_index=3,
        #         use_lightnovelco=False,
        #         lnco_base_url="not used",
        #         lnco_ranking_url="not used",
        #         lnco_chapter_count=11037,
        #         lnco_novel_count=1048596, # if you get both of those references, I respect you
        #         lnco_wait_time=1337,
        #         lnco_max_workers=1337,
        #         emotions=["No defaults used"],
        #         include_chunk_in_prompt=False,
        #         use_min_p=False, # TODO maybe make this configurable
        #         rp_prompt_start="",
        #         rp_prompt_end="",
        #         meta_datagen_keys=["emotion_details", "features_details", "scene_card_details", "story_ratings_details", "story_details"],
        #         meta_datagen_extras=None,
        #         chunking_output_dir=shared_chunking_output_dir,
        #         seed=i
        #     )
        #     # Optional: Force garbage collection
        #     # gc.collect()

        #     print(f"RP Run {i}: Taking snapshot after pipeline call...")
        #     snap_after_rp = tracemalloc.take_snapshot()

        #     # Compare the snapshots
        #     top_stats = snap_after_rp.compare_to(snap_before_rp, 'lineno')
        #     print(f"\n[Memory Diff - RP Run {i}] Top 15 lines:")
        #     for stat in top_stats[:15]:
        #         print(stat)

        #     # Total allocated difference
        #     total_allocated_before = sum(stat.size for stat in snap_before_rp.statistics('lineno'))
        #     total_allocated_after = sum(stat.size for stat in snap_after_rp.statistics('lineno'))
        #     print(f"Total memory allocated increase during RP run {i}: {(total_allocated_after - total_allocated_before) / 1024**2:.2f} MiB")
        #     print("-" * 30 + "\n")

        # except Exception as e:
        #     print(f"ERROR running roleplaying pipeline (run {i}): {e}")
        #     traceback.print_exc()
        #     # Optionally continue to the next iteration or handle the error differently
        #     continue # Example: skip to the next RP sample if one fails

    set_progress(
        task_id=task_id, progress=2 / 8, message="Starting PDF Dataset Generation!"
    )
    # --- PDF Cleaning Pipeline State ---
    pdf_sampling_state = {}
    pdf_list_keys = {
        "inputs": "pdf_cleaning_valid_input_paths",
        "completion": "pdf_cleaning_completion_mode_and_use_stop",
        "models": "pdf_cleaning_model",
        "prompts": "pdf_cleaning_prompts",
        "chunks": "pdf_cleaning_chunk_size",
        "subsets": "pdf_cleaning_subset",
    }

    for i in range(pdf_number_of_times_to_sample):
        # --- Sampling (MUST happen before skip check) ---
        input_path = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_valid_input_paths,
            pdf_sampling_state,
            pdf_list_keys["inputs"],
        )
        completion_mode_and_use_stop = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_completion_mode_and_use_stop,
            pdf_sampling_state,
            pdf_list_keys["completion"],
        )
        model_details = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_model,
            pdf_sampling_state,
            pdf_list_keys["models"],
        )
        prompts = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_prompts,
            pdf_sampling_state,
            pdf_list_keys["prompts"],
        )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_chunk_size,
            pdf_sampling_state,
            pdf_list_keys["chunks"],
        )
        subset = _sample_tracked_item(
            this_pipeline_random,
            pdf_cleaning_subset,
            pdf_sampling_state,
            pdf_list_keys["subsets"],
        )
        subset_size = subset["subset_size"]
        use_subset = subset["use_subset"]

        # Extract values
        completion_mode = completion_mode_and_use_stop["completion_mode"]
        use_stop = completion_mode_and_use_stop["use_stop"]
        small_model = model_details["small_model"]
        small_mode = model_details["small_mode"]
        small_base_url = model_details["small_base_url"]
        small_api_key = model_details["small_api_key"]
        # Assuming large model details might be used later or were intended
        large_model = model_details.get(
            "large_model", "default_large_model_if_needed"
        )  # Added .get for safety
        large_mode = model_details.get("large_mode", "default_large_mode_if_needed")
        large_base_url = model_details.get("large_base_url", None)
        large_api_key = model_details.get("large_api_key", None)
        cost_per_million_large_input = model_details["cost_per_million_large_input"]
        cost_per_million_large_output = model_details["cost_per_million_large_output"]
        cost_per_million_small_input = model_details["cost_per_million_small_input"]
        cost_per_million_small_output = model_details["cost_per_million_small_output"]

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "pdf_cleaning", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping PDF Cleaning Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"PDF Cleaning Pipeline Run {i}")
        print("=" * 80)
        print(f"Input Path:          {input_path}")
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(f"Chunk Size:         {chunk_size}")
        print(f"Use Subset:         {use_subset}")
        print(f"Subset Size:        {subset_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        print(f"PDF Run {i}: Taking snapshot before pipeline call...")
        snap_before_pdf = tracemalloc.take_snapshot()
        # try:
        #     _ = await pdf_clean_and_convert_pipeline(
        #         use_subset=use_subset,
        #         subset_size=subset_size,
        #         chunk_size=chunk_size,
        #         input_dir=input_path,
        #         output_dir=this_run_output_dir,
        #         completion_mode=completion_mode,
        #         small_model=small_model,
        #         small_api_key=small_api_key,
        #         small_base_url=small_base_url,
        #         small_mode=small_mode,
        #         large_model=large_model,
        #         large_api_key=large_api_key,
        #         large_base_url=large_base_url,
        #         large_mode=large_mode,
        #         concurrency_limit=concurrency_limit,
        #         use_stop=use_stop,
        #         prompts=prompts,
        #         default_prompts="./prompts", # standard fallback
        #         cost_per_million_small_input=cost_per_million_small_input,
        #         cost_per_million_small_output=cost_per_million_small_output,
        #         cost_per_million_large_input=cost_per_million_large_input,
        #         cost_per_million_large_output=cost_per_million_large_output,
        #         read_files_manually=True,
        #         do_meta_datagen=True,
        #         meta_datagen_keys=["clean_pdf_details"],
        #         meta_datagen_extras=None,
        #         chunking_output_dir=shared_chunking_output_dir,
        #         seed=i
        #     )
        #     # Optional: Force garbage collection
        #     # gc.collect()

        #     print(f"PDF Run {i}: Taking snapshot after pipeline call...")
        #     snap_after_pdf = tracemalloc.take_snapshot()

        #     # Compare the snapshots
        #     top_stats = snap_after_pdf.compare_to(snap_before_pdf, 'lineno')
        #     print(f"\n[Memory Diff - PDF Run {i}] Top 15 lines:")
        #     for stat in top_stats[:15]:
        #         print(stat)

        #     # Total allocated difference
        #     total_allocated_before = sum(stat.size for stat in snap_before_pdf.statistics('lineno'))
        #     total_allocated_after = sum(stat.size for stat in snap_after_pdf.statistics('lineno'))
        #     print(f"Total memory allocated increase during PDF run {i}: {(total_allocated_after - total_allocated_before) / 1024**2:.2f} MiB")
        #     print("-" * 30 + "\n")

        # except Exception as e:
        #     print(f"ERROR running pdf_cleaning pipeline (run {i}): {e}")
        #     traceback.print_exc()
        #     continue # Example: skip to the next PDF sample if one fails

    set_progress(
        task_id=task_id, progress=3 / 8, message="Starting Repvar Dataset Generation!"
    )
    # --- Repvar Pipeline State ---
    repvar_sampling_state = {}
    repvar_list_keys = {
        "inputs": "repvar_valid_input_paths",
        "completion": "repvar_cleaning_completion_mode_and_use_stop",
        "variations": "repvar_variation_count",
        "models": "repvar_model",
        "chunks": "repvar_chunk_size",
        "subsets": "repvar_subset",
    }
    for i in range(repvar_times_to_sample):
        # --- Sampling ---
        input_path_group = _sample_tracked_item(
            this_pipeline_random,
            repvar_valid_input_paths,
            repvar_sampling_state,
            repvar_list_keys["inputs"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            repvar_cleaning_completion_mode_and_use_stop,
            repvar_sampling_state,
            repvar_list_keys["completion"],
        )
        variation_count = _sample_tracked_item(
            this_pipeline_random,
            repvar_variation_count,
            repvar_sampling_state,
            repvar_list_keys["variations"],
        )
        model_config = _sample_tracked_item(
            this_pipeline_random,
            repvar_model,
            repvar_sampling_state,
            repvar_list_keys["models"],
        )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            repvar_chunk_size,
            repvar_sampling_state,
            repvar_list_keys["chunks"],
        )
        subset_config = _sample_tracked_item(
            this_pipeline_random,
            repvar_subset,
            repvar_sampling_state,
            repvar_list_keys["subsets"],
        )

        # Extract values
        input_path = input_path_group["path"]
        context = input_path_group["additional_context"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        prompts = model_config["prompt_path"]
        inferred_facts = model_config["inferred_facts"]
        use_subset = subset_config["use_subset"]
        subset_size = subset_config["subset_size"]

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "repvar", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Representation Variation Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Representation Variation Pipeline Run {i}")
        print("=" * 80)
        print(f"Input Path:          {input_path}")
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Variation Count:     {variation_count}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(f"Inferred Facts:     {inferred_facts}")
        print(f"Chunk Size:         {chunk_size}")
        print(f"Use Subset:         {use_subset}")
        print(f"Subset Size:        {subset_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        print(f"RepVar Run {i}: Taking snapshot before pipeline call...")
        snap_before_repvar = tracemalloc.take_snapshot()
        # try:
        #     _ = await representation_variation_pipeline(
        #         input_dir=input_path,
        #         output_dir=this_run_output_dir,
        #         completion_mode=completion_mode,
        #         use_stop=use_stop,
        #         small_model=small_model,
        #         small_api_key=small_api_key,
        #         small_base_url=small_base_url,
        #         small_mode=small_mode,
        #         large_model=large_model,
        #         large_api_key=large_api_key,
        #         large_base_url=large_base_url,
        #         large_mode=large_mode,
        #         cost_per_million_small_input=cost_per_million_small_input,
        #         cost_per_million_small_output=cost_per_million_small_output,
        #         cost_per_million_large_input=cost_per_million_large_input,
        #         cost_per_million_large_output=cost_per_million_large_output,
        #         prompts=prompts,
        #         default_prompts="./prompts",
        #         chunk_size=chunk_size,
        #         use_subset=use_subset,
        #         subset_size=subset_size,
        #         concurrency_limit=concurrency_limit,
        #         variation_generator_count=variation_count,
        #         include_context_in_dataset=False, # this setting is meaningless for meta-datagen, but turning it off will save space
        #         dataset_context="Meta Dataset Generation",
        #         code_variation_functions=[],
        #         make_inferred_facts=inferred_facts,
        #         do_meta_datagen=True,
        #         meta_datagen_keys=["judgement_details", "atomic_facts_details", "variations_details", "inferred_facts_details"],
        #         meta_datagen_extras=None,
        #         chunking_output_dir=shared_chunking_output_dir,
        #         additional_dataset_context=context,
        #         seed=i
        #     )
        #     # Optional: Force garbage collection
        #     # gc.collect()

        #     print(f"RepVar Run {i}: Taking snapshot after pipeline call...")
        #     snap_after_repvar = tracemalloc.take_snapshot()

        #     # Compare the snapshots
        #     top_stats = snap_after_repvar.compare_to(snap_before_repvar, 'lineno')
        #     print(f"\n[Memory Diff - RepVar Run {i}] Top 15 lines:")
        #     for stat in top_stats[:15]:
        #         print(stat)

        #     # Total allocated difference
        #     total_allocated_before = sum(stat.size for stat in snap_before_repvar.statistics('lineno'))
        #     total_allocated_after = sum(stat.size for stat in snap_after_repvar.statistics('lineno'))
        #     print(f"Total memory allocated increase during RepVar run {i}: {(total_allocated_after - total_allocated_before) / 1024**2:.2f} MiB")
        #     print("-" * 30 + "\n")

        # except Exception as e:
        #     print(f"ERROR running repvar pipeline (run {i}): {e}")
        #     traceback.print_exc()
        #     continue # Example: skip to the next RepVar sample if one fails

    set_progress(
        task_id=task_id,
        progress=0.5,
        message="Starting Factual (single) Dataset Generation!",
    )
    # --- Factual (Single) Pipeline State ---
    factual_sampling_state = {}
    factual_list_keys = {
        "inputs": "factual_generation_inputs_and_conversation_instructions",
        "completion": "factual_generation_cleaning_completion_mode_and_use_stop",
        "models": "factual_generation_model",
        "chunks": "factual_chunk_size",
        "prompts": "factual_prompts_and_skips",
        "subsets": "factual_subset",
    }
    for i in range(factual_generation_times_to_sample):
        # --- Sampling ---
        input_and_instructions = _sample_tracked_item(
            this_pipeline_random,
            factual_generation_inputs_and_conversation_instructions,
            factual_sampling_state,
            factual_list_keys["inputs"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            factual_generation_cleaning_completion_mode_and_use_stop,
            factual_sampling_state,
            factual_list_keys["completion"],
        )
        model_config = _sample_tracked_item(
            this_pipeline_random,
            factual_generation_model,
            factual_sampling_state,
            factual_list_keys["models"],
        )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            factual_chunk_size,
            factual_sampling_state,
            factual_list_keys["chunks"],
        )
        prompts_config = _sample_tracked_item(
            this_pipeline_random,
            factual_prompts_and_skips,
            factual_sampling_state,
            factual_list_keys["prompts"],
        )
        factual_subset_size = _sample_tracked_item(
            this_pipeline_random,
            factual_subset,
            factual_sampling_state,
            factual_list_keys["subsets"],
        )

        # Extract values
        input_path = input_and_instructions["input_path"]
        conversation_instructions = input_and_instructions["conversation_instructions"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        prompts = prompts_config["prompt_path"]
        skip_answer_relevancy_check = prompts_config["skip_answer_relevancy_check"]
        skip_repair_qa_tuples = prompts_config["skip_repair_qa_tuples"]
        skip_filter_chunks = prompts_config["skip_filter_chunks"]
        skip_question_check = prompts_config["skip_question_check"]
        skip_conversation_generation = prompts_config["skip_conversation_generation"]
        skip_answer_accuracy_check = prompts_config["skip_answer_accuracy_check"]

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "factual_generation", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Factual Generation Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Factual Generation Pipeline Run {i}")
        print("=" * 80)
        print(f"Input Path:          {input_path}")
        print(f"Conversation Instructions: {conversation_instructions}")
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(
            f"Skip Checks:        answer_relevancy={skip_answer_relevancy_check}, repair={skip_repair_qa_tuples}"
        )
        print(
            f"                    filter={skip_filter_chunks}, question={skip_question_check}"
        )
        print(f"                    conversation={skip_conversation_generation}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        # TODO might be a good idea to replace read_files_manually with something like checking if text chunks passed in is truthy and if so using them, else reading. That way we have one option instead of two.

        try:
            facts, _ = await generate_factual_qa_dataset(
                input_dir=input_path,
                output_dir=this_run_output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
                small_model=small_model,
                small_api_key=small_api_key,
                small_base_url=small_base_url,
                small_mode=small_mode,
                large_model=large_model,
                large_api_key=large_api_key,
                large_base_url=large_base_url,
                large_mode=large_mode,
                cost_per_million_small_input=cost_per_million_small_input,
                cost_per_million_small_output=cost_per_million_small_output,
                cost_per_million_large_input=cost_per_million_large_input,
                cost_per_million_large_output=cost_per_million_large_output,
                prompts=prompts,
                default_prompts="./prompts",
                chunk_size=chunk_size,
                use_subset=True,
                subset_size=factual_subset_size,
                concurrency_limit=concurrency_limit,
                conversation_instructions=conversation_instructions,
                double_check_counter=double_check_counter,
                skip_answer_relevancy_check=skip_answer_relevancy_check,
                skip_repair_qa_tuples=skip_repair_qa_tuples,
                skip_filter_chunks=skip_filter_chunks,
                skip_question_check=skip_question_check,
                skip_conversation_generation=skip_conversation_generation,
                read_files_manually=True,
                do_meta_datagen=True,
                meta_datagen_keys=[
                    "judgement_details",
                    "factual_questions_details",
                    "question_validation_details",
                    "answer_relevancy_validation_details",
                    "answer_accuracy_validation_details",
                    "context_repair_details",
                    "conversation_details",
                ],
                meta_datagen_extras=[],  # TODO this shall have to be filled out. We will do so... let's try a different approach and do this breadth first. You know maybe when we train agents on this, we can teach it to do tasks breadth first or depth first (latter being small batch sizes way). The goal for me is to reduce teh amount I Have to hold in my battered head.
                # from here to end of call is meaningless args included to prevent errors. Maybe it would be good for me to include dfaults for some things that are not as esssential
                phase_index=3,
                work_in_phases=False,
                use_gutenberg=False,
                start_url="meaningless",
                max_books=1,
                max_failures=1,
                hub_path="Nope",
                private=True,
                push_to_hub=False,
                use_filenames=True,
                skip_answer_accuracy_check=skip_answer_accuracy_check,
                do_not_use_system_prompts=True,  # mostly meaningless, though it does somewhat impact the conv start pipeline later I suppose
                final_assistant_prompts_no_rag=[""],
                final_assistant_prompts_rag=[""],
                rag_failure_percentage=0,
                items_per_conversation=1,
                chunking_output_dir=shared_chunking_output_dir,
                seed=i,
            )

        except Exception as e:
            print(f"ERROR running factual_generation (single) pipeline (run {i}): {e}")
            traceback.print_exc()
            continue  # Example: skip to the next factual sample if one fails

    set_progress(
        task_id=task_id,
        progress=5 / 8,
        message="Starting Factual (multi) Dataset Generation!",
    )
    # --- Factual (Multi) Pipeline State ---
    multi_source_sampling_state = {}
    multi_source_list_keys = {
        "inputs": "multi_source_inputs_and_conversation_instructions",  # Note: variable name implies instructions, but seems used only for path
        "completion": "multi_source_cleaning_completion_mode_and_use_stop",
        "models": "multi_source_model",
        "prompts": "multi_source_prompts",
        "chunks": "multi_source_chunk_size",
        "subsets": "multi_source_subset",
    }
    for i in range(multi_source_times_to_sample):
        # --- Sampling ---
        input_path = _sample_tracked_item(
            this_pipeline_random,
            multi_source_inputs_and_conversation_instructions,
            multi_source_sampling_state,
            multi_source_list_keys["inputs"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            multi_source_cleaning_completion_mode_and_use_stop,
            multi_source_sampling_state,
            multi_source_list_keys["completion"],
        )
        model_config = _sample_tracked_item(
            this_pipeline_random,
            multi_source_model,
            multi_source_sampling_state,
            multi_source_list_keys["models"],
        )
        # Handle optional prompts list
        prompts = "./prompts"  # Default
        if multi_source_prompts:
            prompts = _sample_tracked_item(
                this_pipeline_random,
                multi_source_prompts,
                multi_source_sampling_state,
                multi_source_list_keys["prompts"],
            )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            multi_source_chunk_size,
            multi_source_sampling_state,
            multi_source_list_keys["chunks"],
        )
        subset_size = _sample_tracked_item(
            this_pipeline_random,
            multi_source_subset,
            multi_source_sampling_state,
            multi_source_list_keys["subsets"],
        )

        # Extract values
        conversation_instructions = ""  # As noted in original code
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]

        # Skip parameters hardcoded as False in original
        skip_answer_relevancy_check = False
        skip_repair_qa_tuples = False
        skip_filter_chunks = False
        skip_question_check = False
        skip_conversation_generation = False
        skip_answer_accuracy_check = False

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "multi_source", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Multi-Source Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Multi-Source Pipeline Run {i}")
        print("=" * 80)
        print(f"Input Path:          {input_path}")
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:            {use_stop}")
        print(f"Small Model:         {small_model}")
        print(f"Small Mode:          {small_mode}")
        print(f"Large Model:         {large_model}")
        print(f"Large Mode:          {large_mode}")
        print(f"Prompts Path:        {prompts}")
        print(f"Chunk Size:          {chunk_size}")
        print(f"Subset Size:         {subset_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        # TODO might be a good idea to replace read_files_manually with something like checking if text chunks passed in is truthy and if so using them, else reading. That way we have one option instead of two.

        try:
            multi_source_facts, _ = await generate_multi_source_dataset(
                input_dir=input_path,
                output_dir=this_run_output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
                small_model=small_model,
                small_api_key=small_api_key,
                small_base_url=small_base_url,
                small_mode=small_mode,
                large_model=large_model,
                large_api_key=large_api_key,
                large_base_url=large_base_url,
                large_mode=large_mode,
                cost_per_million_small_input=cost_per_million_small_input,
                cost_per_million_small_output=cost_per_million_small_output,
                cost_per_million_large_input=cost_per_million_large_input,
                cost_per_million_large_output=cost_per_million_large_output,
                prompts=prompts,
                default_prompts="./prompts",
                chunk_size=chunk_size,
                use_subset=True,
                subset_size=subset_size,
                concurrency_limit=concurrency_limit,
                conversation_instructions=conversation_instructions,
                double_check_counter=double_check_counter,
                skip_answer_relevancy_check=skip_answer_relevancy_check,
                skip_repair_qa_tuples=skip_repair_qa_tuples,
                skip_filter_chunks=skip_filter_chunks,
                skip_question_check=skip_question_check,
                skip_conversation_generation=skip_conversation_generation,
                read_files_manually=True,
                do_meta_datagen=True,
                meta_datagen_keys=[
                    "judgement_details",
                    "factual_questions_details",
                    "question_validation_details",
                    "answer_relevancy_validation_details",
                    "answer_accuracy_validation_details",
                    "context_repair_details",
                    "conversation_details",
                ],
                meta_datagen_extras=[],  # TODO this shall have to be filled out. We will do so... let's try a different approach and do this breadth first. You know maybe when we train agents on this, we can teach it to do tasks breadth first or depth first (latter being small batch sizes way). The goal for me is to reduce teh amount I Have to hold in my battered head.
                # from here to end of call is meaningless args included to prevent errors. Maybe it would be good for me to include dfaults for some things that are not as esssential
                phase_index=3,
                work_in_phases=False,
                use_gutenberg=False,
                start_url="meaningless",
                max_books=1,
                max_failures=1,
                hub_path="Nope",
                private=True,
                push_to_hub=False,
                use_filenames=True,
                skip_answer_accuracy_check=skip_answer_accuracy_check,
                do_not_use_system_prompts=True,  # mostly meaningless, though it does somewhat impact the conv start pipeline later I suppose
                final_assistant_prompts_no_rag=[""],
                final_assistant_prompts_rag=[""],
                rag_failure_percentage=0,
                items_per_conversation=1,
                chunking_output_dir=shared_chunking_output_dir,
                cleanup_embedding_dir=False,
                seed=i,
            )

        except Exception as e:
            print(f"ERROR running multi_source pipeline (run {i}): {e}")
            traceback.print_exc()
            continue  # Example: skip to the next multi-source sample if one fails

    # REMINDER meta_datagen_extras is a list of paths to prompts inside that pipeline's folder which can be used for data. they are tested and work but I don't have any official ones yet.

    # epiphany: if you want your python to be better, use it in the way that it is meant to be. You won't make good python by writing it like a Java. Write it like good python. Lean into the style of the tool.
    # I need to save all these not ecomments to my self.
    set_progress(
        task_id=task_id,
        progress=6 / 8,
        message="Starting Correction Dataset Generation!",
    )
    # --- Correction Pipeline State ---
    correction_sampling_state = {}
    correction_list_keys = {
        "models": "correction_model",
        "inputs": "correction_inputs",
        "prompts": "correction_prompts",
        "chunks": "correction_chunk_sizes",
        "completion": "correction_cleaning_completion_mode_and_use_stop",
    }
    for i in range(correction_time_to_sample):
        # --- Sampling ---
        model_config = _sample_tracked_item(
            this_pipeline_random,
            correction_model,
            correction_sampling_state,
            correction_list_keys["models"],
        )
        inp = _sample_tracked_item(
            this_pipeline_random,
            correction_inputs,
            correction_sampling_state,
            correction_list_keys["inputs"],
        )
        prompts = _sample_tracked_item(
            this_pipeline_random,
            correction_prompts,
            correction_sampling_state,
            correction_list_keys["prompts"],
        )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            correction_chunk_sizes,
            correction_sampling_state,
            correction_list_keys["chunks"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            correction_cleaning_completion_mode_and_use_stop,
            correction_sampling_state,
            correction_list_keys["completion"],
        )

        # Extract values
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "corrections", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Correction Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Correction Pipeline Run {i}")
        print("=" * 80)
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(f"Chunk Size:         {chunk_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        try:
            _ = await correction_pipeline(
                output_dir=this_run_output_dir,
                input_dir=inp,
                completion_mode=completion_mode,
                use_stop=use_stop,
                small_model=small_model,
                small_api_key=small_api_key,
                small_base_url=small_base_url,
                small_mode=small_mode,
                large_model=large_model,
                large_api_key=large_api_key,
                large_base_url=large_base_url,
                large_mode=large_mode,
                cost_per_million_small_input=cost_per_million_small_input,
                cost_per_million_small_output=cost_per_million_small_output,
                cost_per_million_large_input=cost_per_million_large_input,
                cost_per_million_large_output=cost_per_million_large_output,
                prompts=prompts,
                default_prompts="./prompts",
                chunk_size=chunk_size,
                use_subset=True,
                subset_size=correction_subset_size,
                concurrency_limit=concurrency_limit,
                read_files_manually=True,
                do_meta_datagen=True,
                prompt_template="{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{message['content'] + '\n'}}{% elif (message['role'] == 'user') %}{{'Human: ' + message['content'] + ' **Finished.**' + '\n'}}{% elif message['role'] == 'assistant' %}{{'AI: ' + message['content'] + ' **Finished.**' + '\n'}}{% endif %}{% endfor %}",
                meta_datagen_keys=["masked_conversation_details", "judgement_details"],
                meta_datagen_extras=[],
                chunking_output_dir=shared_chunking_output_dir,
                seed=i,
            )
        except Exception as e:
            print(f"ERROR running correction pipeline (run {i}): {e}")
            traceback.print_exc()
            continue  # Example: skip to the next correction sample if one fails

    # --- Generic Thoughts Pipeline State ---
    generic_sampling_state = {}
    generic_list_keys = {
        "models": "generic_thoughts_model",
        "inputs": "generic_thoughts_inputs",
        "prompts": "generic_thoughts_prompts",
        "completion": "generic_thoughts_completion_and_use_stop",
        # Note: chunk_size seems reused from correction loop, assuming it's intended
    }
    for i in range(generic_thoughts_times_to_sample):
        # --- Sampling ---
        model_config = _sample_tracked_item(
            this_pipeline_random,
            generic_thoughts_model,
            generic_sampling_state,
            generic_list_keys["models"],
        )
        inp = _sample_tracked_item(
            this_pipeline_random,
            generic_thoughts_inputs,
            generic_sampling_state,
            generic_list_keys["inputs"],
        )
        prompts = _sample_tracked_item(
            this_pipeline_random,
            generic_thoughts_prompts,
            generic_sampling_state,
            generic_list_keys["prompts"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            generic_thoughts_completion_and_use_stop,
            generic_sampling_state,
            generic_list_keys["completion"],
        )
        # Assuming chunk_size from the correction loop is intentionally reused, otherwise it needs its own sampling

        # Extract values
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]

        # --- Calculate Output Dir ---
        this_run_output_dir = os.path.join(output_dir, "transform_generics", str(i))

        # --- Skip Check ---
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping Generic Transformation Pipeline Run {i} as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"Generic Transformation Pipeline Run {i}")
        print("=" * 80)
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(f"Chunk Size:         {chunk_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print("=" * 80 + "\n")

        try:
            _ = await transform_generic_data_pipeline(
                output_dir=this_run_output_dir,
                input_dir=inp,
                completion_mode=completion_mode,
                use_stop=use_stop,
                small_model=small_model,
                small_api_key=small_api_key,
                small_base_url=small_base_url,
                small_mode=small_mode,
                large_model=large_model,
                large_api_key=large_api_key,
                large_base_url=large_base_url,
                large_mode=large_mode,
                cost_per_million_small_input=cost_per_million_small_input,
                cost_per_million_small_output=cost_per_million_small_output,
                cost_per_million_large_input=cost_per_million_large_input,
                cost_per_million_large_output=cost_per_million_large_output,
                prompts=prompts,
                default_prompts="./prompts",
                use_subset=True,
                subset_size=generic_thoughts_subset_size,
                concurrency_limit=concurrency_limit,
                cot_preface="Not used!",
                cot_suffix="Also not used!",
                read_files_manually=True,
                do_meta_datagen=True,
                meta_datagen_keys=["thought_process_addition_details"],
                meta_datagen_extras=[],
                seed=i,
            )
        except Exception as e:
            print(f"ERROR running transform_generics pipeline (run {i}): {e}")
            traceback.print_exc()
            continue  # Example: skip to the next generic transform sample if one fails

    # TODO have a create_node function or something that takes a function as an arg and wraps it in some stuff that gives us some "inheritence" behavior that is actually currying --
    # we want kwargs. We want to log unused arguments (kwargs that are not in the main function's thing); we want a few other bits of boilerplate I think?

    set_progress(
        task_id=task_id, progress=7 / 8, message="Starting RAG Dataset Generation!"
    )
    # --- RAG Data Pipeline State ---
    rag_sampling_state = {}
    rag_list_keys = {
        "models": "rag_data_model",
        "prompts": "rag_data_prompts",
        "chunks": "rag_data_chunk_sizes",
        "completion": "rag_data_cleaning_completion_mode_and_use_stop",
        "failure": "rag_failure_percentage",
        "inputs": "rag_data_inputs",
        "max_chunks": "rag_max_chunks",  # Assuming this should be sampled too
    }

    # First RAG Loop
    for i in range(rag_data_time_to_sample):
        # --- Sampling ---
        model_config = _sample_tracked_item(
            this_pipeline_random,
            rag_data_model,
            rag_sampling_state,
            rag_list_keys["models"],
        )
        prompts = _sample_tracked_item(
            this_pipeline_random,
            rag_data_prompts,
            rag_sampling_state,
            rag_list_keys["prompts"],
        )
        chunk_size = _sample_tracked_item(
            this_pipeline_random,
            rag_data_chunk_sizes,
            rag_sampling_state,
            rag_list_keys["chunks"],
        )
        completion_mode_use_stop = _sample_tracked_item(
            this_pipeline_random,
            rag_data_cleaning_completion_mode_and_use_stop,
            rag_sampling_state,
            rag_list_keys["completion"],
        )
        rag_failure_rate = _sample_tracked_item(
            this_pipeline_random,
            rag_failure_percentage,
            rag_sampling_state,
            rag_list_keys["failure"],
        )
        rag_inp = _sample_tracked_item(
            this_pipeline_random,
            rag_data_inputs,
            rag_sampling_state,
            rag_list_keys["inputs"],
        )
        current_rag_max_chunks = rag_max_chunks  # Sampled max chunks

        # Extract values
        small_model = model_config["small_model"]
        small_api_key = model_config["small_api_key"]
        small_base_url = model_config["small_base_url"]
        small_mode = model_config["small_mode"]
        large_model = model_config["large_model"]
        large_api_key = model_config["large_api_key"]
        large_base_url = model_config["large_base_url"]
        large_mode = model_config["large_mode"]
        cost_per_million_small_input = model_config["cost_per_million_small_input"]
        cost_per_million_small_output = model_config["cost_per_million_small_output"]
        cost_per_million_large_input = model_config["cost_per_million_large_input"]
        cost_per_million_large_output = model_config["cost_per_million_large_output"]
        completion_mode = completion_mode_use_stop["completion_mode"]
        use_stop = completion_mode_use_stop["use_stop"]
        input_path_rag = rag_inp["path"]
        skip_filter_chunks = rag_inp["skip_filter_chunks"]

        # --- Calculate Output Dir ---
        # Note: Both RAG loops write to the same output structure. This might be intentional,
        # but if they represent different phases, they might need distinct output paths.
        # Assuming current logic is correct for now.
        this_run_output_dir = os.path.join(output_dir, "rag_data", str(i))

        # --- Skip Check ---
        # This check applies to the output dir path shared by both loops for a given 'i'.
        # If the first loop creates it, the second loop might skip unnecessarily if not careful.
        # However, the current structure runs loop 1 fully, then loop 2 fully.
        # So, skipping based on existence before the call is okay within each loop.
        # if os.path.exists(this_run_output_dir):
        #     print(f"Skipping RAG Data Pipeline Run {i} (first loop) as output directory already exists: {this_run_output_dir}")
        #     continue

        # --- Log and Execute ---
        print("\n" + "=" * 80)
        print(f"rag_data Pipeline Run {i} (First Loop)")  # Identifier added
        print("=" * 80)
        print(f"Completion Mode:     {completion_mode}")
        print(f"Use Stop:           {use_stop}")
        print(f"Small Model:        {small_model}")
        print(f"Small Mode:         {small_mode}")
        print(f"Large Model:        {large_model}")
        print(f"Large Mode:         {large_mode}")
        print(f"Prompts Path:       {prompts}")
        print(f"Chunk Size:         {chunk_size}")
        print(
            f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}"
        )
        print(
            f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}"
        )
        print(
            f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}"
        )
        print(
            f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}"
        )
        print(f"RAG Failure Rate:   {rag_failure_rate}")
        print(f"RAG Max Chunks:     {current_rag_max_chunks}")  # Use sampled value
        print(f"Output Directory:   {this_run_output_dir}")
        print("=" * 80 + "\n")

        try:
            _ = await rag_data_pipeline(
                input_dir=input_path_rag,
                output_dir=this_run_output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
                small_model=small_model,
                small_api_key=small_api_key,
                small_base_url=small_base_url,
                small_mode=small_mode,
                large_model=large_model,
                large_api_key=large_api_key,
                large_base_url=large_base_url,
                large_mode=large_mode,
                cost_per_million_small_input=cost_per_million_small_input,
                cost_per_million_small_output=cost_per_million_small_output,
                cost_per_million_large_input=cost_per_million_large_input,
                cost_per_million_large_output=cost_per_million_large_output,
                prompts=prompts,
                default_prompts="./prompts",
                chunk_size=chunk_size,
                use_subset=True,
                subset_size=rag_subset_size,
                rag_failure_percentage=rag_failure_rate,
                rag_max_chunks=current_rag_max_chunks,  # Pass sampled value
                concurrency_limit=concurrency_limit,
                do_meta_datagen=True,
                meta_datagen_keys=["masked_conversation_details", "judgement_details"],
                meta_datagen_extras=[],
                read_files_manually=True,
                user_format="doesnotmatter",
                assistant_format="also does not matter",
                system_format="what do you know, does not matter",
                final_assistant_prompts=["{data}"],
                bos="",
                num_items_per_group=1,
                skip_filter_chunks=skip_filter_chunks,
                chunking_output_dir=shared_chunking_output_dir,
                seed=i,
            )
        except Exception as e:
            print(f"ERROR running rag_data pipeline (first loop) (run {i}): {e}")
            traceback.print_exc()
            continue  # Example: skip to the next RAG sample if one fails

    # # Reset RAG state for the second loop if needed, or reuse. Reusing seems appropriate here.
    # # If the intention was independent sampling rounds for loop 1 and 2, state should be reset.
    # # Assuming reuse for now.
    # for i in range(rag_data_time_to_sample):
    #     # --- Sampling ---
    #     # Uses the same state `rag_sampling_state`
    #     model_config = _sample_tracked_item(this_pipeline_random, rag_data_model, rag_sampling_state, rag_list_keys["models"])
    #     prompts = _sample_tracked_item(this_pipeline_random, rag_data_prompts, rag_sampling_state, rag_list_keys["prompts"])
    #     chunk_size = _sample_tracked_item(this_pipeline_random, rag_data_chunk_sizes, rag_sampling_state, rag_list_keys["chunks"])
    #     completion_mode_use_stop = _sample_tracked_item(this_pipeline_random, rag_data_cleaning_completion_mode_and_use_stop, rag_sampling_state, rag_list_keys["completion"])
    #     rag_failure_rate = _sample_tracked_item(this_pipeline_random, rag_failure_percentage, rag_sampling_state, rag_list_keys["failure"])
    #     rag_inp = _sample_tracked_item(this_pipeline_random, rag_data_inputs, rag_sampling_state, rag_list_keys["inputs"])
    #     current_rag_max_chunks = _sample_tracked_item(this_pipeline_random, rag_max_chunks, rag_sampling_state, rag_list_keys["max_chunks"]) # Sampled max chunks

    #     # Extract values
    #     small_model = model_config["small_model"]
    #     small_api_key = model_config["small_api_key"]
    #     small_base_url = model_config["small_base_url"]
    #     small_mode = model_config["small_mode"]
    #     large_model = model_config["large_model"]
    #     large_api_key = model_config["large_api_key"]
    #     large_base_url = model_config["large_base_url"]
    #     large_mode = model_config["large_mode"]
    #     cost_per_million_small_input = model_config["cost_per_million_small_input"]
    #     cost_per_million_small_output = model_config["cost_per_million_small_output"]
    #     cost_per_million_large_input = model_config["cost_per_million_large_input"]
    #     cost_per_million_large_output = model_config["cost_per_million_large_output"]
    #     completion_mode = completion_mode_use_stop["completion_mode"]
    #     use_stop = completion_mode_use_stop["use_stop"]
    #     input_path_rag = rag_inp["path"]
    #     skip_filter_chunks = rag_inp["skip_filter_chunks"]

    #     # --- Calculate Output Dir ---
    #     this_run_output_dir = os.path.join(output_dir, "rag_data", str(i))

    #     # --- Skip Check ---
    #     # if os.path.exists(this_run_output_dir):
    #     #      print(f"Skipping RAG Data Pipeline Run {i} (second loop) as output directory already exists: {this_run_output_dir}")
    #     #      continue

    #     # --- Log and Execute ---
    #     print("\n" + "="*80)
    #     print(f"rag_data Pipeline Run {i} (Second Loop)") # Identifier added
    #     print("="*80)
    #     print(f"Completion Mode:     {completion_mode}")
    #     print(f"Use Stop:           {use_stop}")
    #     print(f"Small Model:        {small_model}")
    #     print(f"Small Mode:         {small_mode}")
    #     print(f"Large Model:        {large_model}")
    #     print(f"Large Mode:         {large_mode}")
    #     print(f"Prompts Path:       {prompts}")
    #     print(f"Chunk Size:         {chunk_size}")
    #     print(f"Small Model Input Cost (per 1M tokens):  ${cost_per_million_small_input}")
    #     print(f"Small Model Output Cost (per 1M tokens): ${cost_per_million_small_output}")
    #     print(f"Large Model Input Cost (per 1M tokens):  ${cost_per_million_large_input}")
    #     print(f"Large Model Output Cost (per 1M tokens): ${cost_per_million_large_output}")
    #     print(f"RAG Failure Rate:   {rag_failure_rate}")
    #     print(f"RAG Max Chunks:     {current_rag_max_chunks}") # Use sampled value
    #     print(f"Output Directory:   {this_run_output_dir}")
    #     print("="*80 + "\n")

    #     try:
    #         _ = await rag_data_pipeline(
    #             input_dir=input_path_rag,
    #             output_dir=this_run_output_dir,
    #             completion_mode=completion_mode,
    #             use_stop=use_stop,
    #             small_model=small_model,
    #             small_api_key=small_api_key,
    #             small_base_url=small_base_url,
    #             small_mode=small_mode,
    #             large_model=large_model,
    #             large_api_key=large_api_key,
    #             large_base_url=large_base_url,
    #             large_mode=large_mode,
    #             cost_per_million_small_input=cost_per_million_small_input,
    #             cost_per_million_small_output=cost_per_million_small_output,
    #             cost_per_million_large_input=cost_per_million_large_input,
    #             cost_per_million_large_output=cost_per_million_large_output,
    #             prompts=prompts,
    #             default_prompts="./prompts",
    #             chunk_size=chunk_size,
    #             use_subset=True,
    #             subset_size=rag_subset_size,
    #             rag_failure_percentage=rag_failure_rate,
    #             rag_max_chunks=current_rag_max_chunks, # Pass sampled value
    #             concurrency_limit=concurrency_limit,
    #             do_meta_datagen=True,
    #             meta_datagen_keys=["masked_conversation_details", "judgement_details"],
    #             meta_datagen_extras=[],
    #             read_files_manually=True,
    #             user_format="doesnotmatter",
    #             assistant_format="also does not matter",
    #             system_format="what do you know, does not matter",
    #             final_assistant_prompts=["{data}"],
    #             bos="",
    #             num_items_per_group=1,
    #             skip_filter_chunks=skip_filter_chunks,
    #             chunking_output_dir=shared_chunking_output_dir,
    #             seed=i
    #         )
    #     except Exception as e:
    #         print(f"ERROR running rag_data pipeline (second loop) (run {i}): {e}") # Added identifier for second loop
    #         traceback.print_exc()
    #         continue # Example: skip to the next RAG sample if one fails

    # next step after we collect all the things, first thing we do is load them
    # why did we not return them?
    # too intrusive into the normal pipeline flow
    # so they are at the hardcoded output paths within this script:
    # os.path.join(output_dir, ONEOFTHETHINGSBELOW)
    # "conv_starts", "corrections", "factual_generation", "pdf_cleaning", "rag_data", "repvar", "roleplaying"
    # within each of these is a number of subfolders named for the index of that generation.
    # within each subfolder is a meta_datagen folder.
    # within each of these are chat_lists and completion_lists folders
    # For each of the pipelines ("oneofthethingsbelow") we want to load all their chat lists into one big list
    # we want to create a dict with pipelinename: chat list for each pipeline that we have made data from
    pipeline_chat_lists = {}
    pipeline_names = [
        "corrections",
        "factual_generation",
        "pdf_cleaning",
        "rag_data",
        "repvar",
        "roleplaying",
        "transform_generics",
    ]

    print(f"\nLoading chat lists from {output_dir}...")

    for pipeline_name in pipeline_names:
        print(f"\n{'='*50}")
        print(f"PROCESSING PIPELINE: {pipeline_name}")
        print(f"{'='*50}")

        pipeline_dir = os.path.join(output_dir, pipeline_name)
        print(f"Looking for pipeline directory at: {pipeline_dir}")

        if not os.path.exists(pipeline_dir):
            print(
                f"WARNING: Pipeline directory {pipeline_dir} does not exist. Skipping."
            )
            continue
        else:
            print(f"Found pipeline directory: {pipeline_dir}")

        all_chats = []

        # Get all numbered subdirectories
        print(f"Scanning for numbered run directories in {pipeline_dir}...")
        run_dirs = [
            d
            for d in os.listdir(pipeline_dir)
            if os.path.isdir(os.path.join(pipeline_dir, d)) and d.isdigit()
        ]
        print(f"Found {len(run_dirs)} run directories: {run_dirs}")

        for run_dir in run_dirs:
            print(f"\n{'-'*40}")
            print(f"Processing run directory: {run_dir}")

            chat_lists_dir = os.path.join(
                pipeline_dir, run_dir, "meta_datagen", "chat_lists"
            )
            print(f"Looking for chat lists at: {chat_lists_dir}")

            if not os.path.exists(chat_lists_dir):
                print(
                    f"WARNING: Chat lists directory {chat_lists_dir} does not exist. Skipping."
                )
                continue
            else:
                print(f"Found chat lists directory: {chat_lists_dir}")

            # Load all chat list files in this directory
            print(f"Scanning for JSON and JSONL files in {chat_lists_dir}...")
            chat_files = [
                f for f in os.listdir(chat_lists_dir) if f.endswith((".jsonl", ".json"))
            ]
            print(f"Found {len(chat_files)} chat files: {chat_files}")

            for chat_file in chat_files:
                file_path = os.path.join(chat_lists_dir, chat_file)
                print(f"\nAttempting to load: {file_path}")

                try:
                    with open(file_path, "r") as f:
                        print(f"File opened successfully, determining format...")

                        chats = []
                        if file_path.endswith(".jsonl"):
                            print("Detected .jsonl format, parsing line by line...")
                            for line_num, line in enumerate(f):
                                line = line.strip()
                                if line:  # Skip empty lines
                                    try:
                                        chat_item = json.loads(line)
                                        chats.append(chat_item)
                                    except json.JSONDecodeError as je:
                                        print(
                                            f"ERROR: JSON parsing failed for line {line_num + 1} in {file_path}: {str(je)}"
                                        )
                                        print(
                                            f"Line content: {line[:100]}..."
                                        )  # Print beginning of the problematic line
                                        # Optionally, decide whether to continue or break/raise
                        elif file_path.endswith(".json"):
                            print("Detected .json format, parsing entire file...")
                            chats = json.load(f)
                            # Ensure chats is a list, even if the JSON file contains a single object (though it should be a list based on prior logic)
                            if not isinstance(chats, list):
                                print(
                                    f"WARNING: Expected a list in {file_path}, but found {type(chats)}. Wrapping in a list."
                                )
                                chats = [chats]
                        else:
                            # This case should not be reached due to the file filter, but included for robustness
                            print(
                                f"WARNING: Unsupported file format for {file_path}. Skipping."
                            )
                            continue

                        print(f"Parsing successful. Found {len(chats)} chat items.")

                        # Print the first item of chats if it exists
                        if chats:
                            print(f"First chat item from {file_path}:")
                            print(f"Chat type: {type(chats[0])}")
                            print(
                                f"Chat keys: {list(chats[0].keys()) if isinstance(chats[0], dict) else 'Not a dictionary'}"
                            )
                            print(json.dumps(chats[0]))
                        else:
                            print(f"WARNING: No chat items found in {file_path}")

                        print(
                            f"Adding {len(chats)} chats to all_chats (current size: {len(all_chats)})"
                        )
                        all_chats.extend(chats)
                        print(f"all_chats size after extension: {len(all_chats)}")

                    print(f"Successfully loaded {len(chats)} chats from {file_path}")

                except json.JSONDecodeError as je:
                    print(f"ERROR: JSON parsing failed for {file_path}: {str(je)}")
                    print(
                        f"First 100 characters of file: {open(file_path, 'r').read(100)}"
                    )
                except Exception as e:
                    print(f"ERROR loading {file_path}: {str(e)}")
                    print(f"Exception type: {type(e).__name__}")
                    import traceback

                    print(f"Traceback: {traceback.format_exc()}")

        print(f"\n{'-'*40}")
        print(f"Finished processing all runs for pipeline: {pipeline_name}")

        if all_chats:
            print(
                f"Adding {len(all_chats)} chats to pipeline_chat_lists[{pipeline_name}]"
            )
            pipeline_chat_lists[pipeline_name] = all_chats
            print(f"Total chats loaded for {pipeline_name}: {len(all_chats)}")
        else:
            print(f"WARNING: No chat data found for {pipeline_name}")

    # also, tomorrow, I Do RL model training, not api-fication
    # no to scope creep. I mean I will do the api thing but, I can do it while I wait for RL runs. We're time efficient in this house.

    # Print summary of loaded data
    print("\nSummary of loaded chat lists:")
    total_chats = 0
    for pipeline_name, chats in pipeline_chat_lists.items():
        print(f"{pipeline_name}: {len(chats)} chats")
        total_chats += len(chats)
    print(f"Total chats across all pipelines: {total_chats}")

    # Print the first item from each pipeline chat list to inspect the data structure
    print("\nSample of first chat from each pipeline:")
    for pipeline_name, chats in pipeline_chat_lists.items():
        if chats:
            print(f"\n{'-'*40}\n{pipeline_name} first chat:\n{'-'*40}")
            try:
                # Pretty print the first chat with indentation for readability
                print(json.dumps(chats[0]))
            except Exception as e:
                print(f"Error displaying chat: {str(e)}")
                print(f"Raw data: {chats[0]}")
        else:
            print(f"\n{pipeline_name}: No chats available")

    # what we need is the sharegpt from each.

    # assume it's sharegpt
    # because ifit is not it will be.

    all_singleturn_chats = []  # all true
    all_multiturn_segment_chats = (
        []
    )  # system: true, examples: label false,  input and output: label true
    for pipeline_name, chats in pipeline_chat_lists.items():
        for chat in chats:
            # --- Start Obfuscation Check ---
            if obfuscation_map and chat.get("full_input"):
                messages = chat["full_input"]
                if messages and messages[0]["role"] == "system":
                    original_prompt = messages[0]["content"].strip()
                    if original_prompt in obfuscation_map:
                        # print("Caught original prompt!")
                        if this_pipeline_random.random() < 0.5:  # 50% chance
                            messages[0]["content"] = obfuscation_map[original_prompt]
                            # print(f"Obfuscated system prompt for a chat from pipeline '{pipeline_name}'.") # Optional logging
            # --- End Obfuscation Check ---

            if len(chat["full_input"]) <= 3:  # single-turn.
                all_singleturn_chats.append(chat)
            else:
                # take some random number of the user-assistant pairs from start (or after system message if there is one) to the end.
                starting_index = 0
                conv = chat["full_input"]
                if conv[0]["role"] == "system":
                    starting_index = 1

                # Identify all available user-assistant pairs with their original indices
                available_pairs = []
                i = starting_index
                while i < len(conv) - 1:
                    if conv[i]["role"] == "user" and conv[i + 1]["role"] == "assistant":
                        # Store the starting index of the user message and the pair itself
                        available_pairs.append((i, [conv[i], conv[i + 1]]))
                        i += 2  # Move to the next potential pair
                    else:
                        # If we encounter an unexpected pattern, skip to the next message
                        i += 1

                total_pairs = len(available_pairs)

                num_pairs_to_use = 0  # Default to 0
                if total_pairs > 0:
                    # Decide whether to use 0 pairs based on the probability
                    if this_pipeline_random.random() < prob_zero_pairs:
                        num_pairs_to_use = 0
                    else:
                        # Otherwise, select a random number of pairs between 1 and total_pairs
                        num_pairs_to_use = this_pipeline_random.randint(1, total_pairs)

                # Sample and sort pairs only if we need to select some
                sampled_pairs_with_indices = []
                if num_pairs_to_use > 0:
                    # Randomly sample pairs
                    sampled_pairs_with_indices = this_pipeline_random.sample(
                        available_pairs, num_pairs_to_use
                    )
                    # Sort the sampled pairs by their original index to maintain relative order
                    sampled_pairs_with_indices.sort(key=lambda x: x[0])

                # Construct the new conversation
                new_conv = []
                if starting_index == 1:
                    new_conv.append(conv[0])  # Add system message if it exists

                # Add the selected pairs in their original relative order
                for _, pair in sampled_pairs_with_indices:
                    new_conv.extend(pair)

                # Add the final user message
                new_conv.append(conv[-1])

                # Create the new chat dictionary
                processed_chat = {
                    "full_input": new_conv,
                    "full_response": chat["full_response"],
                    # Add other keys from chat if necessary, e.g., detail_key
                    # "detail_key": chat.get("detail_key")
                }

                all_multiturn_segment_chats.append(processed_chat)

    # Dump the processed chat lists to output files
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save single-turn chats
        # if all_singleturn_chats:
        #     singleturn_output_path = os.path.join(output_dir, "singleturn_chats.json")
        #     with open(singleturn_output_path, "w") as f:
        #         json.dump(all_singleturn_chats, f)
        #     print(f"Saved {len(all_singleturn_chats)} single-turn chats to {singleturn_output_path}")

        # # Save multi-turn chats with segments
        # if all_multiturn_segment_chats:
        #     multiturn_output_path = os.path.join(output_dir, "multiturn_segment_chats.json")
        #     with open(multiturn_output_path, "w") as f:
        #         json.dump(all_multiturn_segment_chats, f)
        #     print(f"Saved {len(all_multiturn_segment_chats)} multi-turn segment chats to {multiturn_output_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Print tokenizer information
    print("\n" + "=" * 80)
    print("TOKENIZER INFORMATION")
    print("=" * 80)
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(
        f"BOS token: {tokenizer.bos_token!r}"
        if hasattr(tokenizer, "bos_token")
        else "BOS token: None"
    )
    print(
        f"EOS token: {tokenizer.eos_token!r}"
        if hasattr(tokenizer, "eos_token")
        else "EOS token: None"
    )
    print("=" * 80)

    # Print dataset statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"SINGLETURN SEGMENT CHATS COUNT: {len(all_singleturn_chats)}")
    print(f"MULTITURN SEGMENT CHATS COUNT: {len(all_multiturn_segment_chats)}")
    print("=" * 80 + "\n")

    singleturn_segments = []
    for item in all_singleturn_chats:
        singleturn_segments.append(create_meta_segment_convs(item, tokenizer))

    multiturn_segments = []
    for item in all_multiturn_segment_chats:
        multiturn_segments.append(create_meta_segment_convs(item, tokenizer))

    # Save the segments
    # Save the segments to output files
    if output_dir:
        # Save single-turn segments
        if singleturn_segments:
            singleturn_segments_path = os.path.join(
                output_dir, "singleturn_segments.json"
            )
            with open(singleturn_segments_path, "w") as f:
                json.dump([s for s in singleturn_segments if s is not None], f)
            print(
                f"Saved {len([s for s in singleturn_segments if s is not None])} single-turn segments to {singleturn_segments_path}"
            )

        # Save multi-turn segments
        if multiturn_segments:
            multiturn_segments_path = os.path.join(
                output_dir, "multiturn_segments.json"
            )
            with open(multiturn_segments_path, "w") as f:
                json.dump([s for s in multiturn_segments if s is not None], f)
            print(
                f"Saved {len([s for s in multiturn_segments if s is not None])} multi-turn segments to {multiturn_segments_path}"
            )

    set_progress(task_id, progress=1.0, message="Pipeline Complete!!! ")

    # as part of the API we will expose the aliases in the super config

    # NOTE the order of steps is going to be the following:
    # 1. process the format
    # 2. filter the poor quality things
    # 3. remove from some % of the things,  their few-shot examples ( we will need to make sure that all pipelines have output indicators)

    # this pipeline is not just the plumbing and some extras. We'll have to process the data we get. Some % will have noe examples etc.

    # NOTE this pipeline will not just generate the meta-data, but will also process it into a training set and create a config. Notably this means that anyone can create a model specialized for THEIR dataset generation pipelines. So it is very useful and flexible.

    # we probably need at least 20k ctx for our model. Maybe 30k.


# define a function
# takes a dict with the keys full_input
# (looks something like):
# {
#         "full_input": [
#             {
#                 "role": "system",
#                 "content": "You are an expert educational AI that will make a determination as to whether the contents of the paragraph(s) provided are suitable for making educational questions based off of them; these questions should be able to test the knowledge in in the book. The book in question is medicine_wikipedia.txt, and you should keep this in mind when considering what kind of questions should be capable of being developed. If there is sufficiently deep information to make questions about, you will judge it suitable, even if the knowledge being tested does not reflect typical curricula. Essentially: you will determine if provided text is a table of contents, introductory paragraph for a book, etc., or if it actually contains real information that would be worthy to teach and make questions for an examination from. Your task includes first analyzing the text, thinking through whether or not good questions can be made from it. \n\nEnd-of-chapter exercises, or questions that appear in the text BUT DO NOT HAVE A PROVIDED ANSWER, are not suitable for question generation, since it should be assumed that the creator of the questions must rely on the information in the text to get their answers.\n\nStubs or placeholder pages from websites, e.g., that have headings but no content, are not suitable. This is because the tool you are a part of sometimes has webpages fed into it en masse, and it is important to filter out these stubs so that they are not used to generate questions. A good way to tell stub pages is if they are very short and say very little.\n\nFollowing this, at the very end of your response, you will write \"Suitable\" or \"Not suitable\". It is imperative that you write one of these two things, as your answer is being automatically processed by a regex, so it must match one of those two strings exactly."
#             },
#             {
#                 "role": "user",
#                 "content": "Text details: Through England On a Side Saddle, by Celia Fiennes\n\nText: \n\"\"\"\nThe Project Gutenberg eBook of Through England on a side saddle\n    \nThis ebook is for the use of anyone anywhere in the United States and\nmost other parts of the world at no cost and with almost no restrictions\nwhatsoever. You may copy it, give it away or re-use it under the terms\nof the Project Gutenberg License included with this ebook or online\nat www.gutenberg.org. If you are not located in the United States,\nyou will have to check the laws of the country where you are located\nbefore using this eBook.\n\nTitle: Through England on a side saddle\n        In the time of William and Mary\n\n\nAuthor: Celia Fiennes\n\nContributor: Emily W. Griffiths\n\nRelease date: November 17, 2023 [eBook #72156]\n\nLanguage: English\n\nOriginal publication: London: Simpkin, Marshall & Co.; Hamilton, Adams & Co, 1888\n\nCredits: Steve Mattern, Barry Abrahamsen, and the Online Distributed Proofreading Team at https://www.pgdp.net (This book was produced from images made available by the HathiTrust Digital Library.)\n\n\n*** START OF THE PROJECT GUTENBERG EBOOK THROUGH ENGLAND ON A SIDE SADDLE ***\n\n\n\n\n                            Through England\n\n                            On a Side Saddle\n\n\n                    IN THE TIME OF WILLIAM AND MARY\n\n\n                           BEING THE DIARY OF\n\n                            _CELIA FIENNES._\n\n\n                             --------------\n\n\n                        WITH AN INTRODUCTION BY\n\n                        THE HON. MRS GRIFFITHS.\n\n\n\n\n                         ---------------------\n\n\n\n\n                                _LONDON:\n                Field & Tuer, The Leadenhall Press, E.C.\n            Simpkin, Marshall & Co.; Hamilton, Adams & Co._\n\n                               ----------\n\n          _New York: Scribner & Welford, 743 & 745, Broadway._\n\n                                   --\n\n                                  1888\n\n\n------------------------------------------------------------------------\n\"\"\""
#             },
#             {
#                 "role": "assistant",
#                 "content": "## Reasoning and thought process:\nStep 1. Identify Paragraph Content: This paragraph is from the introductory section of \"Through England on a Side Saddle\" by Celia Fiennes, including copyright and publication information.\nStep 2. Evaluate Educational Relevance: The paragraph contains legal, copyright, and publication information specific to the book's distribution and not its content.\nStep 3. Assess Specific Contexts and Formats:\n  - Purely Descriptive Lists: The paragraph is a descriptive list of publication details without educational content.\n  - Dialogues or Quotes Without Context: Not applicable.\n  - Ambiguous or Unclear Texts: The text is clear but lacks educational substance.\n  - Stubs or Placeholder Pages: The text is not a stub or placeholder page from a website.\nStep 4. Assess the Possibility of Formulating Questions: The paragraph's content does not lend itself to educational questions due to its focus on publication rather than substantive content.\nStep 5. Determine Suitability for Educational Purposes: The paragraph lacks thematic, historical, scientific, or literary information and is thus unsuitable for creating educational questions.\nStep 6. Check for Contextual Completeness: The paragraph, while complete in its context, does not offer educational insights or concepts.\nStep 7. Final Judgment: Unsuitable for educational questions."
#             },
#             {
#                 "role": "user",
#                 "content": "Text details: Thus Spake Zaranthustra, by Friedrich Nietzsche\n\nText:\n\"\"\"\nWhen Zarathustra was thirty years old, he left his home and the lake of\nhis home, and went into the mountains. There he enjoyed his spirit and\nsolitude, and for ten years did not weary of it. But at last his heart\nchanged,\u2014and rising one morning with the rosy dawn, he went before the\nsun, and spake thus unto it:\n\nThou great star! What would be thy happiness if thou hadst not those for\nwhom thou shinest!\n\nFor ten years hast thou climbed hither unto my cave: thou wouldst have\nwearied of thy light and of the journey, had it not been for me, mine\neagle, and my serpent.\n\nBut we awaited thee every morning, took from thee thine overflow and\nblessed thee for it.\n\nLo! I am weary of my wisdom, like the bee that hath gathered too much\nhoney; I need hands outstretched to take it.\n\nI would fain bestow and distribute, until the wise have once more become\njoyous in their folly, and the poor happy in their riches.\n\nTherefore must I descend into the deep: as thou doest in the\nevening, when thou goest behind the sea, and givest light also to the\nnether-world, thou exuberant star!\n\nLike thee must I GO DOWN, as men say, to whom I shall descend.\n\nBless me, then, thou tranquil eye, that canst behold even the greatest\nhappiness without envy!\n\nBless the cup that is about to overflow, that the water may flow golden\nout of it, and carry everywhere the reflection of thy bliss!\n\nLo! This cup is again going to empty itself, and Zarathustra is again\ngoing to be a man.\n\nThus began Zarathustra's down-going.\n\"\"\""
#             },
#             {
#                 "role": "assistant",
#                 "content": "## Reasoning a...
# and the key full_response (string) detail_key (string) and conversations (list of dicts like before)
# and does the following:
# if it is single-turn (one user message in the full input, no assistant message, optionally a system message) then it appends it to the singleturn chats list unchanged
# if it is multi-turn (we have user-assistant message pairs in the input)
