import re
import sys
import os
import nltk

from augmentoolkit.generation_functions.cleanup import cleanup_dir
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.majority_vote_step import MajorityVoteStep
from augmentoolkit.generation_functions.one_to_many_step import OneToManyStep
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.cost_estimation_logging import (
    calculate_pipeline_cost_efficiency,
)
from augmentoolkit.utils.observers import (
    create_input_token_counter,
    create_log_observer,
    create_output_token_counter,
)
from augmentoolkit.utils.work_with_output_files import recombine_many_to_one, save_data
from generation.core_components.chunking import (
    chunk_text_list,
    count_tokens,
    count_total_tokens,
    read_and_chunk_text,
    subset_text_list,
)
from generation.core_components.filter_chunks import (
    filter_out_failed_items,
    filter_out_failed_items_dict,
)
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from generation.core_pipelines.factual_generation_individual.factual_generation_helpers import (
    extract_questions_from_response,
    extract_reasoning_from_context_check,
    judge_paragraph_processor,
    parse_answer_accuracy_validation,
    parse_answer_relevancy_validation_step,
    parse_validation_step,
    save_conversations,
    save_plain_qatuples,
    scrape_text_read_files_manually,
    conversation_generator_step,
)
from redis_config import set_progress

nltk.download("punkt", quiet=True)
from augmentoolkit.generation_functions.process_multiturn_functions import (
    extract_conversation,
)
import augmentoolkit.utils.create_pretraining_set
import augmentoolkit.utils.sentence_chunking_algorithm
from augmentoolkit.utils.parse_bool import parse_bool
import asyncio
import traceback

import augmentoolkit.utils.group_by_text


def filter_the_text(q_or_a):
    list_of_bad_strings = [
        # " the text",
        "according to the text",
        "as stated in",
        "explicitly stated",
        "as defined in",
        "given text",
        "provided information",
        "the text states",
    ]
    if any(bad_string in q_or_a for bad_string in list_of_bad_strings):
        return False
    return True


async def generate_factual_qa_dataset(
    completion_mode,
    phase_index,
    work_in_phases,
    skip_filter_chunks,
    skip_repair_qa_tuples,
    chunk_size,
    use_gutenberg,
    start_url,
    max_books,
    max_failures,
    skip_conversation_generation,
    hub_path,
    private,
    push_to_hub,
    use_filenames,
    input_dir,
    prompts,
    default_prompts,
    use_stop,
    skip_answer_relevancy_check,
    skip_answer_accuracy_check,
    conversation_instructions,
    do_not_use_system_prompts,
    skip_question_check,
    final_assistant_prompts_no_rag,
    final_assistant_prompts_rag,
    rag_failure_percentage,
    items_per_conversation,
    concurrency_limit,
    small_model,
    small_api_key,
    small_base_url,
    small_mode,
    large_model,
    large_api_key,
    large_base_url,
    large_mode,
    use_subset,
    subset_size,
    double_check_counter,
    output_dir,
    cost_per_million_small_input,
    cost_per_million_small_output,
    cost_per_million_large_input,
    cost_per_million_large_output,
    read_files_manually=True,  # only set to false if it is being used as a node, we use kwargs to escape the default problem of clunkiness
    text_chunks_passed_in=[],  # also having node-based things means we need to return something doesn't it. May need to add additional args here. Do we assume that it's a proper dict? No?
    do_meta_datagen=False,
    meta_datagen_keys=[],
    meta_datagen_extras=[],
    chunking_output_dir=None,
    task_id=None,
    seed=1048596,
    **kwargs,
):

    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

    ### Definition of all pipeline steps must happen in here. Since use_filenames changes prompt paths significantly and it's an argument now, not a constant that can easily be shared between files. So we must define based on it.

    if use_filenames:
        judgement_prompt_path = "judge_paragraph_filenames"
    else:
        judgement_prompt_path = "judge_paragraph_no_filenames"

    judgement_regex = re.compile(
        r"Reasoning and thought process \(reason intelligently\):(.+)",
        re.DOTALL | re.IGNORECASE,
    )

    filter_all_questions_step = PipelineStep(
        prompt_path=judgement_prompt_path,
        regex=judgement_regex,
        sampling_params={
            "max_tokens": 1450,
            # "min_p": 0.4,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n\n\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        output_processor=judge_paragraph_processor,
        result_key="judged_worthy_for_questions",
        output_file="judge_paragraph",
        details_key="judgement_details",
    )

    qatuples_gen_regex = re.compile(
        r"Questions \(make 4\):\n(.+)", re.IGNORECASE | re.DOTALL
    )

    prompt_path_qatuples_gen = "qatuples_gen_no_filenames"
    if use_filenames:
        prompt_path_qatuples_gen = "qatuples_gen_filenames"

    question_generation_step = OneToManyStep(
        prompt_path=prompt_path_qatuples_gen,
        regex=qatuples_gen_regex,
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.8,
            "top_p": 1,
        },
        output_processor=extract_questions_from_response,
        result_key="qa_tuples",
        max_retries=3,
        log_full_outputs=False,
        output_file="factual_questions",
        details_key="factual_questions_details",
        input_file="judge_paragraph",
    )

    # class FactualQAValidationStep(MajorityVoteStep):
    #     def process_input_data(self, input_data):
    #         print("INPUT DATA")
    #         print(input_data)
    #         input_data["question"] = input_data["qa_tuples"]["question"]
    #         input_data["answer"] = input_data["qa_tuples"]["answer"]
    #         return super().process_input_data(input_data)

    question_validation_step = MajorityVoteStep(
        prompt_path="check_question",
        regex=re.compile(
            r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):(.+)",
            re.DOTALL | re.IGNORECASE,
        ),
        sampling_params={
            "max_tokens": 1500,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        output_processor=parse_validation_step,
        result_key="question_validation_votes",
        vote_count_needed=double_check_counter,
        percent_true_to_pass=0.5,
        output_file="factual_questions",
        final_determination_key="question_validation_final",
        details_key="question_validation_details",
    )

    answer_relevancy_validation_step = MajorityVoteStep(
        prompt_path="check_answer_relevancy_with_text",
        regex=re.compile(
            r"Reasoning and thought process \(be careful about extra details, even vague ones\):\n(.+)",
            re.DOTALL | re.IGNORECASE,
        ),
        sampling_params={
            "max_tokens": 1500,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        retries=3,
        output_processor=parse_answer_relevancy_validation_step,
        percent_true_to_pass=0.5,
        vote_count_needed=double_check_counter,
        output_file="factual_questions",
        result_key="answer_rel_votes",
        final_determination_key="answer_rel_validation_final",
        details_key="answer_relevancy_validation_details",
    )  # I may want a prompt set for doing this with reasoning models. So that I can train my own model to do reasoning things on the distil

    answer_accuracy_validation_step = MajorityVoteStep(
        prompt_path="check_answer",
        regex=re.compile(
            r"Reasoning and thought process \(the text is your single source of truth\):\n(.+)",
            re.DOTALL,
        ),
        sampling_params={
            "max_tokens": 1500,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        retries=3,
        output_processor=parse_answer_accuracy_validation,
        vote_count_needed=double_check_counter,
        percent_true_to_pass=0.5,
        output_file="factual_questions",
        result_key="answer_acc_votes",
        final_determination_key="answer_acc_validation_final",
        details_key="answer_accuracy_validation_details",
    )

    context_repairer_path = "check_qatuple_context_no_filenames"
    if use_filenames:
        context_repairer_path = "check_qatuple_context_filenames"

    context_repairer_step = PipelineStep(
        prompt_path=context_repairer_path,
        regex=re.compile(
            r"Reasoning and thought process \(be thorough\):(.+)",
            re.DOTALL | re.IGNORECASE,
        ),
        sampling_params={
            "max_tokens": 4000,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n\n\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        output_file="factual_questions",
        output_processor=extract_reasoning_from_context_check,
        result_key="repaired_context",  # we do not employ the result key because we replace the question and answer in the qa dict.
        details_key="context_repair_details",
    )

    ### NOTE end definitions of pipeline steps

    ### NOTE might be a good idea for me to make a "delete key from datagen dict" function to help clean things up at certain points? Ah but that doesn't account for saved values, modifying it in memory does nothing if it's already on disk. hmm tough question.

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
                create_log_observer(output_dir),
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
                create_log_observer(output_dir),
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

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_task_with_limit(task):
        async with semaphore:
            return await task

    # notably, to be used as a node these things need to take their sentence chunks as input, not an input dir path. Which means that this step wouldn't actually fire. We won't be making a pretraining dataset in the original pipeline anymore since that's handled by repvar when run as part of a larger system.
    # We need to engineer better how to make pipelines vs nodes work. When in node mode, these things will behave different. When in pipeline mode they'll take a certain set of args vs when in node mode they'll take a certain other set. How do we represent this best? Perhaps ask AI. What is the cleanest and best way to engineer this such that pipelines can fulfill both the role of pipeline and node 1) without cluttering the code too much, 2) while maintaining the nice modularity of everything so far?
    # Basically the key annoying thing is that when it's a pipeline we use paths to things like inputs, etc. But as nodes we want to pass arguments in, like a list of sentence chunks. We simply need a flag and then alternate kwargs I suppose.
    # Even prompts might not be relative to self, but instead passed in, an absolute path when this is used as a node
    # in fact I think it will be that way, at least by default for me. Because that way you have more control and it is more explicit and it's all centralized in hte single PIPELINE that you run you don't need to manage multiple configs and folders that way.
    # well but at the same time that means that things can't be seamlessly used across things, like filter is.
    # ARGH tough. Well we should support both and just clearly indicate which is which.
    # The options being, either have your nodes use the prompts in their folder, or pass in the prompts folder path for your nodes to use.

    set_progress(
        task_id=task_id,
        progress=0.0,
        message="Pipeline starting; reading and chunking files",
    )

    if read_files_manually:
        print("Using config...")
        if use_gutenberg:
            scrape_text_read_files_manually(
                start_url=start_url,
                max_books=max_books,
                max_failures=max_failures,
                input_dir=input_dir,
            )

        sentence_chunks = read_and_chunk_text(
            input_dir=input_dir,
            chunk_size=chunk_size,
            use_subset=use_subset,
            subset_size=subset_size,
            keep_folder_structure=True,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
            seed=seed,
        )
        print("LENGTH of sentence chunks passed in")
        print(len(sentence_chunks))
    else:
        print("Using text chunks passed in...")
        sentence_chunks = chunk_text_list(
            text_chunks_passed_in,
            chunk_size,
            keep_folder_structure=True,
            input_dir=input_dir,
            output_dir=chunking_output_dir if chunking_output_dir else output_dir,
        )
        if use_subset:
            sentence_chunks = subset_text_list(
                subset_size=subset_size, text_list=sentence_chunks, seed=seed
            )
        #     sentence_chunks = sentence_chunks[:subset_size]
        print("LENGTH of sentence chunks passed in")
        print(len(sentence_chunks))

    total_tokens = count_total_tokens(sentence_chunks)
    print(f"Total tokens: {total_tokens}")

    sentence_dict = hash_input_list(input_list=sentence_chunks, key_to_hash_with="text")
    set_progress(
        task_id,
        progress=0.1,
        message="Files read and chunked! Proceeding with LLM chunk filtering",
    )

    # The good thing about the api -- it extends into the future. The progress thing is as fundamental as you can get. So if/when we gain the ability to tie it directly to pipeline execution progress, the interface can stay the same, we'll just gain finer control. Until then, logfile viewing hack. We need to output things to a logfile...
    await filter_all_questions_step.execute_pipeline(
        input_dict=sentence_dict,
        engine_wrapper=engine_wrapper,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )

    filter_out_failed_items_dict(
        sentence_dict, key_to_check="judged_worthy_for_questions"
    )
    set_progress(
        task_id,
        progress=0.2,
        message=f"Chunks filtered! {len(sentence_dict.items())} items left; proceeding with questions dict generation",
    )

    # Right I forgot that to create one-to-many with the dicts it's pretty easy. Make a new dict and hash along the new output key + the hash of the para + the inner-index to control for duplicates within a group. Just got to change saving and loading?

    # Generate questions
    questions_dict = await question_generation_step.execute_pipeline(
        input_dict=sentence_dict,
        engine_wrapper=engine_wrapper_large,
        rtwl=run_task_with_limit,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        include_details=do_meta_datagen,
    )

    for key, value in questions_dict.items():
        # print("Iterating over items")
        questions_dict[key]["question"] = value["qa_tuples"]["question"]
        questions_dict[key]["answer"] = value["qa_tuples"]["answer"]
        # print(value)

    set_progress(
        task_id,
        progress=0.3,
        message=f"Questions generated! {len(questions_dict.items())} questions made; proceeding with validation (if no steps are skipped)",
    )

    save_data(questions_dict, output_dir, "factual_questions")

    if not skip_question_check:
        await question_validation_step.execute_pipeline(
            input_dict=questions_dict,
            engine_wrapper=engine_wrapper_large,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )

        filter_out_failed_items_dict(
            questions_dict, key_to_check="question_validation_final"
        )

    set_progress(
        task_id,
        progress=0.4,
        message=f"Question validation completed! {len(questions_dict.items())} questions validated; proceeding with answer relevancy check (if not skipped)",
    )

    if not skip_answer_relevancy_check:
        await answer_relevancy_validation_step.execute_pipeline(
            input_dict=questions_dict,
            engine_wrapper=engine_wrapper_large,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )

        filter_out_failed_items_dict(
            questions_dict, key_to_check="answer_rel_validation_final"
        )
    # INSERT_YOUR_CODE
    set_progress(
        task_id,
        progress=0.5,
        message=f"Answer relevancy check completed! {len(questions_dict.items())} questions passed relevancy check; proceeding with answer accuracy check (if not skipped)",
    )

    if not skip_answer_accuracy_check:
        await answer_accuracy_validation_step.execute_pipeline(
            input_dict=questions_dict,
            engine_wrapper=engine_wrapper_large,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )

        filter_out_failed_items_dict(
            questions_dict, key_to_check="answer_acc_validation_final"
        )

    # INSERT_YOUR_CODE
    set_progress(
        task_id,
        progress=0.6,
        message=f"Answer accuracy check completed! {len(questions_dict.items())} questions passed accuracy check; proceeding with context repair (if not skipped)",
    )

    if not skip_repair_qa_tuples:
        await context_repairer_step.execute_pipeline(
            input_dict=questions_dict,
            engine_wrapper=engine_wrapper_large,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )

        filter_out_failed_items_dict(
            questions_dict, key_to_check="repaired_context"
        )  # removes all failed items

        print("Original length, pre-context-repair:")
        print(len(questions_dict))
        for key, value in questions_dict.items():
            if isinstance(value["repaired_context"], bool):
                # if this is the case, we need to set the repaired context question and answer to the same as the actual question and answer
                value["repaired_context"] = [value["question"], value["answer"]]
            else:
                # if the repaired context is a q/a tuple
                value["question"] = value["repaired_context"][0]
                value["answer"] = value["repaired_context"][1]
        print("New length, post-context repair")
        print(len(questions_dict))

    # INSERT_YOUR_CODE
    set_progress(
        task_id,
        progress=0.7,
        message="Context repair completed; proceeding with cleanup and preparation for conversation generation",
    )

    for key in questions_dict:
        # Clean up additional keys we no longer need
        keys_to_delete = [
            "metadata",
            "judged_worthy_for_questions",
            "qa_tuples",
            "question_validation_votes",
            "answer_rel_votes",
            "answer_acc_votes",
        ]
        for k in keys_to_delete:
            if k in questions_dict[key]:
                del questions_dict[key][k]

    qa_dicts_by_text = []
    text_hash_groups = {}

    # First pass - group by text hash
    text_hash_groups = recombine_many_to_one(questions_dict)
    # for key, value in questions_dict.items():
    #     text_hash = key.split('-')[0]
    #     if text_hash not in text_hash_groups:
    #         text_hash_groups[text_hash] = {}
    #     index_hash = key.split('-')[1]
    #     text_hash_groups[text_hash][index_hash] = value
    #     # index hash will be a string number, like '1', etc.

    # print("TEXT HASH GROUPS 1st item")
    # first_key = next(iter(text_hash_groups))
    # print(first_key, text_hash_groups[first_key])

    # Second pass - convert groups to list format
    # for text_hash, items in text_hash_groups.items():
    #     dict_list = []
    #     for index_hash, item in items.items():
    #         dict_list.append(item)
    #     qa_dicts_by_text.append({
    #         "dict_list": dict_list
    #     })

    if not skip_conversation_generation:
        await conversation_generator_step.execute_pipeline(
            input_dict=text_hash_groups,  # each item is a text group. We just need to sort it
            engine_wrapper=engine_wrapper_large,
            rtwl=run_task_with_limit,
            default_prompt_folder=default_prompts,
            prompt_folder=prompts,
            output_dir=output_dir,
            completion_mode=completion_mode,
            use_stop=use_stop,
            include_details=do_meta_datagen,
        )
        save_conversations(
            text_hash_groups,
            final_assistant_prompts_rag=final_assistant_prompts_rag,
            final_assistant_prompts_no_rag=final_assistant_prompts_no_rag,
            rag_failure_percentage=rag_failure_percentage,
            do_not_use_system_prompts=do_not_use_system_prompts,
            items_per_conversation=items_per_conversation,
            output_dir=output_dir,
            hub_path=hub_path,
            push_to_hub=push_to_hub,
        )

    set_progress(
        task_id,
        progress=0.9,
        message="Conversations generated and saved; proceeding with QA tuple saving",
    )

    plain_qa_list, simplified_rag_list = save_plain_qatuples(
        qa_dicts_by_text=text_hash_groups,
        final_assistant_prompts_rag=final_assistant_prompts_rag,
        final_assistant_prompts_no_rag=final_assistant_prompts_no_rag,
        rag_failure_percentage=rag_failure_percentage,
        do_not_use_system_prompts=do_not_use_system_prompts,
        items_per_conversation=items_per_conversation,
        output_dir=output_dir,
        hub_path=hub_path,
        push_to_hub=push_to_hub,
    )

    cleanup_dir(output_dir=output_dir)

    calculate_pipeline_cost_efficiency(
        total_input_tokens=total_tokens,
        token_counters=[small_token_counter, large_token_counter],
    )

    if do_meta_datagen:
        create_meta_dataset(
            data_dicts=[questions_dict, text_hash_groups, sentence_dict],
            meta_datagen_keys=meta_datagen_keys,
            meta_datagen_extras=meta_datagen_extras,
            input_processors=[],
            output_dir=os.path.join(output_dir, "meta_datagen"),
        )

    set_progress(task_id, progress=1.0, message="Pipeline Complete!")

    return plain_qa_list, simplified_rag_list
