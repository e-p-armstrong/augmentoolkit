import random
import itertools
import os
import asyncio
import json
import re
from tqdm import asyncio as tqdmasyncio
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
import logging
from math import ceil
import traceback
import glob
import uuid
from augmentoolkit.generation_functions import (
    create_scenario_plan_many_tuples,
    create_scenario_many_tuples,
    check_answer,
    check_question,
    check_answer_relevancy_with_text,
    generate_new_question,
    generate_questions,
    generate_questions_plan,
    process_multiturn_functions,
    identify_duplicates,
    judge_paragraph,
    multi_turn_conversation,
    check_qatuple_context,
    create_character_card_many_tuples,
    create_character_card_plan_many_tuples,
    extract_name,
)


# Used basically everywhere:
def make_id():
    return str(uuid.uuid4())


# Also used basically everywhere:
def write_output_to_file(output, directory, uuid):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path using the directory and UUID
    file_path = os.path.join(directory, f"{uuid}.txt")

    # Write the output to the file
    with open(file_path, "w") as file:
        file.write(output)

    print(f"Output written to {file_path}")


# multiturn helpers
# These will probably be used for multiturn rapid-fire answering.


# Idea: use multiple short answers to train the task of answering multiple questions in one response. Two-three short answers per response should be enough.
async def make_multiturn_character(qa_tuples, conv_id, engine_wrapper, assistant_mode, use_filenames):
    if (
        assistant_mode
    ):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"
    (
        plan,
        instructions,
        card_plan_output,
    ) = await create_character_card_plan_many_tuples.create_character_card_plan_many_tuples(
        qa_tuples, engine_wrapper, use_filenames=use_filenames
    )  # I will reuse the many tuples function for short question-answers, there's a lot of prompting in here already
    write_output_to_file(card_plan_output, "./multiturn_card_plan_generations", conv_id)
    (
        char,
        char_output,
    ) = await create_character_card_many_tuples.create_character_card_many_tuples(
        qa_tuples, plan, instructions, engine_wrapper, use_filenames=use_filenames
    )  # creates a character card
    write_output_to_file(char_output, "./multiturn_card_generations", conv_id)
    return char, instructions


async def make_multiturn_scenario(
    qa_tuples, character, conv_id, engine_wrapper, assistant_mode
):
    if (
        assistant_mode
    ):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return "will_be_replaced", "will_be_replaced"
    (
        plan,
        scenario_plan_output,
    ) = await create_scenario_plan_many_tuples.create_scenario_plan_many_tuples(
        qa_tuples, character, engine_wrapper
    )
    write_output_to_file(
        scenario_plan_output, "./multiturn_scenario_plan_generations", conv_id
    )
    (
        scenario,
        scenario_output,
    ) = await create_scenario_many_tuples.create_scenario_many_tuples(
        qa_tuples, character, plan, engine_wrapper
    )  # creates a scenario based on a character card and question/answer tuple
    write_output_to_file(scenario_output, "./multiturn_scenario_generations", conv_id)
    return scenario, plan


async def make_multiturn_conversation_info(qa_tuples, engine_wrapper, assistant_mode, use_filenames):
    conv_id = make_id()
    if (
        assistant_mode
    ):  # If assistant mode is on, multiturn convs will have hardcoded information in its prompt file; but we still need to put something in the file
        return (qa_tuples, "will", "be", "replaced", conv_id)
    # thought_plan = create_thought_plan_many_tuples(qa_tuples,character,scenario,logic_llm) # There IS a way to make multiturn chain of thought answering work: generate each pair of messages using a separate prompt or a separate function, each of which has only the thought plan for that question/answer pair. But simply cramming in all the step-by-step things will confuse the hell out of the poor model. So for the first release version we're skipping it and just giving the response, with no reasoning, in the multiturn convs.
    character, instructions = await make_multiturn_character(
        qa_tuples, conv_id, engine_wrapper, assistant_mode, use_filenames
    )
    scenario, scenario_plan = await make_multiturn_scenario(
        qa_tuples, character, conv_id, engine_wrapper, assistant_mode
    )

    return (qa_tuples, character, scenario, scenario_plan, conv_id)


# Group tuples for multiturn example generation (by chunk of source text) and then run that helper (so that we can make multiturn conversations from questions based on the same paragraphs)
def group_by_text(tuples_list):
    # Dictionary to hold the groups with text as the key
    groups = {}

    # Iterate over each tuple in the list
    for question, answer, text, textname in tuples_list:
        # If the text is not yet a key in the dictionary, add it with an empty list
        if text not in groups:
            groups[text] = []

        # Append the current tuple to the appropriate list
        groups[text].append((question, answer, text, textname))

    # Return the values of the dictionary, which are the lists of tuples grouped by text; also remove duplicates
    return [
        identify_duplicates.identify_duplicates(group)
        for group in list(groups.values())
    ]


# Postprocessing function for question/answer validation
async def repair_qatuple_context(idx, tup, engine_wrapper, writepath, vetted_qa_tuples,use_filenames=False):
    file_path = os.path.join(writepath, f"revised_{idx}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()  # Read the file once and store its content
            print(file_path)
            if content == "failed":
                print("Loaded failed file")
                vetted_qa_tuples[idx] = None
                return None
            print("Loaded file:")
            print(content)
            try:
                data = json.loads(content)  # Convert the string back to JSON
                vetted_qa_tuples[idx] = (data[0], data[1], data[2], data[3])
                return None
            except json.JSONDecodeError:
                print("JSON decode error with the contents:", content)

    try:
        revision_id = make_id()
        revision, revision_output = await check_qatuple_context.check_qatuple_context(
            tup, engine_wrapper, use_filenames=use_filenames
        )
        write_output_to_file(
            revision_output, "./question_context_revision_generations", revision_id
        )  # incidentally, identifying the problem and fixing it in the same step (without another planning step) works a lot better than identifying it and then trying to fix it in the next step.
        if isinstance(revision[0], str):  # if the thing was reworded
            vetted_qa_tuples[idx] = revision
        elif not revision[0]:
            vetted_qa_tuples[
                idx
            ] = None  # prepare item for deletion later; right now we just store it as None because indexes
        # else, if it passed, we just leave it be.

        # Write in-progress
        if not os.path.exists(writepath):
            os.makedirs(writepath)

        if vetted_qa_tuples[idx]:
            with open(file_path, "w") as file:
                json.dump(vetted_qa_tuples[idx], file, indent=4)
        else:
            with open(file_path, "w") as file:
                file.write("failed")

    except Exception as e:
        print("!!! ERROR!", e)
        traceback.print_exc()


# Control flow helpers -- Question/Answer Validation
async def vet_answer_accuracy_loop(
    qa_tuple, total_retries, run_id, engine_wrapper=None, double_check_counter=3
):
    try:
        qtuple = qa_tuple
        # print(
        # f"\n\nStarting ACCURACY loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            # print(
            # f"\n\nACCURACY CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
            # )
            judgement, answer_accuracy_output = await check_answer.check_answer(
                qtuple, engine_wrapper
            )
            write_output_to_file(
                answer_accuracy_output, "./check_answer_accuracy_generations", run_id
            )
            if not judgement[0]:  # if not accurate
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1
            if passed_checks >= ceil(double_check_counter / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(double_check_counter / 2):
                break

        if passed_checks >= ceil(double_check_counter / 2):  # if question checks passed
            # print(f"\n\ANSWER ACCURACY CHECKS PASSED retries: {total_retries}")
            return qtuple
        else:
            # Generate new question and restart the loop
            # print(
            # f"\n\nACCURACY CHECKS FAILED - SENDING BACK TO QUESTION LOOP retries: {total_retries}"
            # )
            total_retries += 1
            (
                qtuple,
                generate_new_q_output,
            ) = await generate_new_question.generate_new_question(
                qtuple, engine_wrapper
            )
            write_output_to_file(
                generate_new_q_output, "./regenerate_question_generations", run_id
            )
            return await vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
            )  # going to get one hell of a call stack by the end of this, but it should be fine
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])


async def vet_answer_relevance_loop(
    qa_tuple, total_retries, run_id, engine_wrapper=None, double_check_counter=3
):
    try:
        qtuple = qa_tuple
        # print(
        # f"\n\nStarting RELEVANCE loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            # print(
            # f"\n\nRELEVANCE CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
            # )
            (
                judgement,
                answer_relevancy_output,
            ) = await check_answer_relevancy_with_text.check_answer_relevancy_with_text(
                qtuple, engine_wrapper
            )
            write_output_to_file(
                answer_relevancy_output, "./check_answer_relevancy_generations", run_id
            )
            if not judgement[0]:  # if not relevant
                dissenting_reasoning = judgement[1]
            else:
                passed_checks += 1
            times_checked += 1
            if passed_checks >= ceil(double_check_counter / 2):
                break
            failed_checks = times_checked - passed_checks
            if failed_checks >= ceil(double_check_counter / 2):
                break

        if passed_checks >= ceil(double_check_counter / 2):
            # print(f"\n\nRELEVANCE CHECKS PASSED")
            return await vet_answer_accuracy_loop(
                qtuple,
                total_retries,
                run_id,
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
            )
        else:
            # print(f"\n\nRELEVANCE CHECKS FAILED - SENDING BACK TO QUESTION LOOP")
            total_retries += 1
            (
                qtuple,
                generate_new_q_output,
            ) = await generate_new_question.generate_new_question(
                qtuple, engine_wrapper
            )
            write_output_to_file(
                generate_new_q_output, "./regenerate_question_generations", run_id
            )
            return await vet_question_loop(
                qtuple,
                total_retries,
                question_group_id=run_id.split("--subquestion--")[0],
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
            )
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])


async def vet_question_loop(
    qa_tuple,
    total_retries,
    question_group_id=None,
    engine_wrapper=None,
    double_check_counter=3,
):
    try:
        qtuple = qa_tuple
        # print(
        #     f"\n\nStarting QUESTION loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        while total_retries <= 4:
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""
            while times_checked < double_check_counter:
                # print(
                #     f"\n\nQUESTION CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
                # )
                judgement, check_q_output = await check_question.check_question(
                    qtuple, engine_wrapper
                )
                write_output_to_file(
                    check_q_output, "./check_question_generations", run_id
                )
                if not judgement[0]:  # if not relevant
                    dissenting_reasoning = judgement[1]
                else:
                    passed_checks += 1
                times_checked += 1
                if passed_checks >= ceil(double_check_counter / 2):
                    break
                failed_checks = times_checked - passed_checks
                if failed_checks >= ceil(double_check_counter / 2):
                    break

            if passed_checks >= ceil(
                double_check_counter / 2
            ):  # if all question checks passed
                # print(f"\n\nQUESTION CHECKS PASSED retries: {total_retries}")
                return await vet_answer_relevance_loop(
                    qtuple,
                    total_retries,
                    run_id,
                    engine_wrapper=engine_wrapper,
                    double_check_counter=double_check_counter,
                )
            else:
                # Generate new question and restart the loop
                # print(
                #     f"\n\nQUESTION CHECKS FAILED - GENERATING NEW QUESTION retries: {total_retries}"
                # )
                total_retries += 1
                if (
                    total_retries <= 4
                ):  # only regen question if we're not already at max regens
                    (
                        qtuple,
                        generate_new_q_output,
                    ) = await generate_new_question.generate_new_question(
                        qtuple, engine_wrapper
                    )
                    write_output_to_file(
                        generate_new_q_output,
                        "./regenerate_question_generations",
                        run_id,
                    )
                    print("New question: ", qtuple)
                # no calling of vet_question_loop, since we're already in a while loop
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    return (None, None, None, qtuple[3])


# Question generation
async def generate_qatuples_from_para(
    idx,
    para,
    engine_wrapper=None,
    vetted_qa_tuples=None,
    qa_tuples_dir=None,
    double_check_counter=3,
    use_filenames=False,
    rtwl=None
):
    try:
        existing_files = glob.glob(
            os.path.join(qa_tuples_dir, f"para_{idx}_*.json")
        )  # check if qs already exist

        if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
            print(f"Skipping para_{idx} as files already exist; loading said files")
            for file_path in existing_files:
                with open(file_path, "r") as file:
                    qa_tuple = tuple(json.load(file))
                vetted_qa_tuples.append(qa_tuple)
            return
        question_group_id = make_id()
        # print(f"\n\n\nOUTER LOOP CALL GENERATE QPLAN para: {para}, \n\n idx: {idx}")
        (
            plan,
            questions_plan_output,
        ) = await generate_questions_plan.generate_questions_plan(para, engine_wrapper,use_filenames=use_filenames)
        write_output_to_file(
            questions_plan_output, "./question_plan_generations", question_group_id
        )
        # print(
        # f"\n\n\nOUTER LOOP CALL GENERATE Q: {para}, \n\n idx: {idx} \n\n plan: {plan}"
        # )
        (
            question_answer_tuples,
            question_generation_output,
        ) = await generate_questions.generate_questions(para, plan, engine_wrapper,use_filenames=use_filenames)
        write_output_to_file(
            question_generation_output,
            "./question_generation_generations",
            question_group_id,
        )
        for qnum, question_answer_tuple in enumerate(question_answer_tuples):
            print(f"\n\n=======!!=BEGIN VETTING QA TUPLE {idx}_{qnum}=!!=======\n\n")
            good_qa_tuple = await vet_question_loop(
                question_answer_tuple,
                0,
                question_group_id=question_group_id,
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
            )

            # Write resulting question file if the tuple is not None
            if good_qa_tuple[0] is not None:
                file_path = os.path.join(qa_tuples_dir, f"para_{idx}_q_{qnum}.json")
                with open(file_path, "w") as file:
                    json.dump(good_qa_tuple, file, indent=4)

            vetted_qa_tuples.append(
                good_qa_tuple
            )  # We must filter out all None values at the end; but appending Nones lets us know where things went wrong, and how often.
    except Exception as e:
        print(f"Q ERROR: {e}")
        traceback.print_exc()


# Graphing code generated by GPT-4. May be suboptimal/ugly.
def filter_and_graph(tuples):
    # Count the occurrences of None and non-None for each source text
    source_counts = Counter()
    for paragraph, source in tuples:
        if paragraph is None:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][0] += 1
        else:
            source_counts[source] = source_counts.get(source, [0, 0])
            source_counts[source][1] += 1

    # Prepare data for the graph
    labels = list(source_counts.keys())
    none_counts = [source_counts[source][0] for source in labels]
    non_none_counts = [source_counts[source][1] for source in labels]

    # Plotting the graph
    x = range(len(labels))
    plt.bar(x, none_counts, width=0.4, label="Not suitable", align="center")
    plt.bar(x, non_none_counts, width=0.4, label="Valid Paragraphs", align="edge")
    plt.xlabel("Source Text")
    plt.ylabel("Number of Paragraphs")
    plt.title("Paragraphs Suitable for Questions by Source Text")
    plt.xticks(x, labels, rotation="vertical")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Filter out tuples with None and return the new list
    filtered_list = [t for t in tuples if t[0] is not None]
    return filtered_list


## Paragraph Filtering (worthy for questions?)
async def determine_worthy(
    idx,
    p,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    use_filenames,
):
    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)
    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append((None, data[7:]))
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
    else:
        judgement = await judge_paragraph.judge_paragraph(p, engine_wrapper,use_filenames=use_filenames)
        judged_worthy_for_questions.append(judgement)

        # Prepare the data to be written to the file
        if judgement[0] is not None:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": judgement[0], "metadata": judgement[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{judgement[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement[0] is not None:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")


async def filter_all_questions(
    paragraphs_processed,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    take_subset=False,
    use_filenames=False,
    rtwl=None
):
    if not take_subset:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                engine_wrapper,
                output_dir,
                use_filenames
            )
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(
                idx,
                p,
                judged_worthy_for_questions,
                engine_wrapper,
                output_dir,
                use_filenames
            )
            for idx, p in enumerate(paragraphs_processed[:13])
        ]
    limited_tasks = [rtwl(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks):
        await future


def sentence_chunking_algorithm(file_path, tokenizer, max_token_length=400):
    """
    This function takes a plaintext file and chunks it into sentences.

    :param file_path: Path to the plaintext file
    :param tokenizer: SentencePiece tokenizer
    :param max_token_length: The maximum token length for a chunk of sentences
    :return: List of sentence chunks with source text information
    """
    sentence_chunks_with_source = []
    current_chunk = []
    token_count = 0
    source_name = file_path.replace(".txt", "")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove Gutenberg header and footer
    content = re.sub(
        r"^.*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n",
        "",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^.*?END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$\n",
        "",
        content,
        flags=re.MULTILINE,
    )

    sentences = sent_tokenize(content)

    for sentence in tqdm(sentences, desc=f"Processing {file_path}"):
        sentence_token_count = len(tokenizer.encode(sentence))

        if token_count + sentence_token_count <= max_token_length:
            current_chunk.append(sentence)
            token_count += sentence_token_count
        else:
            sentence_chunks_with_source.append((" ".join(current_chunk), source_name))
            current_chunk = [sentence]
            token_count = sentence_token_count

    # Add the last chunk if it exists
    if current_chunk:
        sentence_chunks_with_source.append((" ".join(current_chunk), source_name))

    return sentence_chunks_with_source


def fix_text(to_replace_arr, text):
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text


async def ensure_multiple_answers_are_same(
    info, conv, engine_wrapper, assistant_mode
):  # why is this a whole separate function? Once upon a time, LLMs were used in validation here, too. But programmatic validation SEEMS to catch the common problems. This is here so that I can add it back in if I have to.
    """Loop to ensure that the answer is consistent in the conversation and in the tuple."""
    retries = 0
    c = conv
    while retries < 2:  # try twice, since multiturn is an expensive operation
        if process_multiturn_functions.call_all_processors(
            c[0], info[0]
        ):  # if programmatic validation passes
            return c

        retries += 1
        if retries >= 2:
            return None
        # If we're here, majority of relevance checks failed
        print("----------------\n\n\n\nRETRYING!!!!\n\n\n\n----------------")
        # Broken info is 1) rare and 2) handled by the retry limit. We don't want to waste compute on regenerating info as they take time.
        retry = await make_multiturn_conversation(info, engine_wrapper, assistant_mode)
        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            return None

    return None


async def make_multiturn_conversation(info, engine_wrapper, assistant_mode):
    conv, conv_output = await multi_turn_conversation.multi_turn_conversation(
        info[0],
        info[1],
        info[2],
        info[3],
        engine_wrapper,
        assistant_mode=assistant_mode,
    )
    write_output_to_file(conv_output, "./multiturn_conversation_generations", info[4])

    return conv


async def create_info(
    idx,
    group,
    engine_wrapper,
    assistant_mode,
    multi_turn_convs_info,
    multi_turn_convs_info_dir,
    rearrangements_to_take,
    use_filenames
):
    all_permutations = list(itertools.permutations(group))

    sample_size = min(rearrangements_to_take, len(all_permutations))
    sampled_permutations = random.sample(all_permutations, sample_size)

    group_convs_info = []

    for iter, perm in enumerate(sampled_permutations):
        file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}_{iter}.json")

        # Skip if file already exists
        if not os.path.exists(file_path):
            try:
                info = await make_multiturn_conversation_info(
                    perm, engine_wrapper, assistant_mode, use_filenames
                )

                if info is not None:
                    with open(file_path, "w") as file:
                        json.dump(info, file, indent=4)

                group_convs_info.append(info)
            except Exception as e:
                print("ERROR!!!!--!!!!", e)
                traceback.print_exc()
        else:
            print(f"Skipped generating {file_path} as it already exists")

    multi_turn_convs_info.append(group_convs_info)


def read_json_files_info(directory):
    # Create a list to hold the tuples
    tuple_list = []

    # Get all the .json files in the directory, sorted
    json_files = sorted([f for f in os.listdir(directory) if f.endswith(".json")])

    # Read each file and convert the contents
    for file in json_files:
        with open(os.path.join(directory, file), "r") as f:
            data = json.load(f)
            # Ensure the data is in the correct format before converting to tuple
            if (
                isinstance(data, list)
                and len(data) == 5
                and isinstance(data[0], list)
                and all(len(item) == 4 for item in data[0])
                and all(isinstance(i, str) for i in data[1:])
            ):
                tuple_list.append((data[0], data[1], data[2], data[3], data[4]))

    return tuple_list


async def create_conversation(
    idx, info, engine_wrapper, multi_turn_convs, multi_turn_convs_dir, assistant_mode
):
    file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")

    # Skip if file already exists
    if not os.path.exists(file_path):
        try:
            conv = await make_multiturn_conversation(
                info, engine_wrapper, assistant_mode
            )
            final_conv = await ensure_multiple_answers_are_same(
                info, conv, engine_wrapper, assistant_mode
            )

            if final_conv is not None:
                with open(file_path, "w") as file:
                    json.dump(final_conv, file, indent=4)

            multi_turn_convs.append(final_conv)
        except Exception as e:
            print("Had an error, retrying...", e)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            multi_turn_convs.append(data)
        print(f"Skipped generating {file_path} as it already exists")


def convert_directory_to_list(directory_path):
    master_list = []
    simplified_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                if isinstance(data, list) and all(
                    isinstance(item, (list, str)) for item in data
                ):
                    master_list.append(data)

                    # Extract and process conversation
                    conversation, primary_char_desc = data[0], data[1]
                    primary_char_name = extract_name.extract_name(primary_char_desc)
                    dialogues = process_multiturn_functions.extract_conversation(
                        conversation
                    )

                    # Convert to simplified format
                    simplified_conversations = []
                    for i, (charname, message) in enumerate(
                        dialogues
                    ):  # Skipping the first message
                        from_person = (
                            "human" if charname == primary_char_name else "gpt"
                        )
                        simplified_conversations.append(
                            {"from": from_person, "value": f"{charname}: {message}"}
                        )

                    if simplified_conversations:  # If there are any conversations
                        simplified_list.append(
                            {"conversations": simplified_conversations}
                        )

    # Write the master list to a new .jsonl file
    with open("master_list.jsonl", "w") as file:
        for item in master_list:
            file.write(json.dumps(item) + "\n")

    # Write the simplified data to a different .jsonl file
    with open("simplified_data.jsonl", "w") as file:
        for item in simplified_list:
            file.write(json.dumps(item) + "\n")

    print(
        "Conversion complete. Master list written to 'master_list.json'. Simplified data written to 'simplified_data.json'."
    )


def convert_directory_and_process_conversations(directory_path):
    master_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

                if isinstance(data, list) and all(
                    isinstance(item, (list, str)) for item in data
                ):
                    # Extract and process the conversation part
                    conversations = process_multiturn_functions.extract_conversation(
                        data[0]
                    )
                    # Convert tuples back to the formatted string as required
                    data[0] = [
                        f"{charname}: {message}" for charname, message in conversations
                    ]
                    master_list.append(data)
                else:
                    print(f"File {filename} is not in the expected format.")

    # Write the master list to a new file
    with open("processed_master_list.json", "w") as file:
        json.dump(master_list, file)

    print(
        "Conversion complete. The processed master list is written to 'processed_master_list.json'."
    )
