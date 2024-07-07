import os
import json
import re
import sys
from tqdm import asyncio as tqdmasyncio
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from augmentoolkit.generation_functions.safe_formatter import safe_format
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
from collections import Counter
import logging
from math import ceil
import traceback
import glob
import yaml

from augmentoolkit.utils.create_conv_starter import create_conv_starter
from augmentoolkit.utils.extract_steps import extract_steps
from augmentoolkit.utils.escape_unescaped_quotes import escape_unescaped_quotes

from augmentoolkit.generation_functions import (
    extract_question_answer,
    identify_duplicates,
    process_multiturn_functions,
    extract_name,
    random_name,
    strip_steps,
)
from augmentoolkit.generation_functions.format_qatuples import format_qatuples

from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.generation_functions.special_instructions import special_instructions

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

DEFAULT_PROMPT_PATH = obj_conf["PATH"]["DEFAULT_PROMPTS"]

def extract_qa_tuples(text):
    pattern = r"\*\*QUESTION:\*\*\s*((?:.|\n)*?)\s*\*\*ANSWER:\*\*\s*((?:.|\n)*?)(?=\s*\*\*QUESTION:\*\*|\Z)"
    matches = re.findall(
        pattern, text + "\n\n**QUESTION:**", re.DOTALL
    )  # The addition is a hack to get around the tricky lookahead problem
    return [(question.strip(), answer.strip()) for question, answer in matches]

import os


# Also used basically everywhere:
def convert_logging_to_dataset(directory):
    print("entering saving mode")
    # found a solution to overfitting on the examples:
    # TRAIN WITHOUT THEM
    # This will produce a WEALTH of instruct data
    # fucking awesome, hopefully
    # also it's also about the domain, lmao
    # so more domain knowledge
    
    output_dir = os.path.join(obj_conf["PATH"]["OUTPUT"], directory)
    
    output_file_path = os.path.join(obj_conf["PATH"]["OUTPUT"], directory + "_DATAGEN_OUTPUT.jsonl")
    
    
    
    if not os.path.exists(output_dir):
        raise Exception("ERROR!! Trying to convert a logging directory to a dataset, when that directory does not exist!")
        
    with open(output_file_path, "w") as f:
        existing_files = glob.glob(
            os.path.join(output_dir, "*.txt")
        )
        
        for file in existing_files:
            with open(file,'r') as file2:
                file_list_of_dicts = yaml.safe_load(file2)
                
            # print(file_list_of_dicts)
            
            sysprompt = {"from": "system", "value": file_list_of_dicts[0]["content"]}
            input = {"from": "human", "value": file_list_of_dicts[-2]["content"]}
            output = {"from": "gpt", "value": file_list_of_dicts[-1]["content"]}
            
            json_to_write = {"conversations": [sysprompt, input, output]}
            
            f.write(json.dumps(json_to_write) + "\n")
    print("...Converted successfully (we think)")
    
    
    
    
    
    
def convert_revised_questions_to_question_generation_training(qa_tuples_by_paragraph, use_filenames):
    print("entering saving mode")
    # found a solution to overfitting on the examples:
    # TRAIN WITHOUT THEM
    # This will produce a WEALTH of instruct data
    # fucking awesome, hopefully
    # also it's also about the domain, lmao
    # so more domain knowledge
    
    output_file_path = os.path.join(obj_conf["PATH"]["OUTPUT"], "questions_generation_dataset.jsonl")
    
    if use_filenames:
        question_generation_prompt = os.path.join(obj_conf["PATH"]["PROMPTS"], "qatuples_gen_filenames.yaml")
    else:
        question_generation_prompt = os.path.join(obj_conf["PATH"]["PROMPTS"], "qatuples_gen_no_filenames.yaml")

    with open(question_generation_prompt, "r") as f:
        qgen_prompt_full = yaml.safe_load(f)
        
        sysprompt = qgen_prompt_full[0]["content"]
        input_template = qgen_prompt_full[-1]["content"]
    
    # revised_questions_output_path = os.path.join(obj_conf["PATH"]["OUTPUT"], "qatuples_revised")
    with open(output_file_path, 'w') as out_file:
        for qatup_group in qa_tuples_by_paragraph:
            answer = format_qatuples(qatup_group)
            text = qatup_group[0][2]
            
            # print(text)
            if not use_filenames:
                input_text = safe_format(input_template, text=text)
            else:
                textname = qatup_group[0][3]
                input_text = safe_format(input_template, text=text, textname=textname)
            sysprompt_obj = {"from": "system", "value": sysprompt}
            input_obj = {"from": "human", "value": input_text}
            answer_obj = {"from": "gpt", "value": answer}
            
            convo = [sysprompt_obj, input_obj, answer_obj]
            out_file.write(json.dumps(convo) + "\n")

    print("...Converted successfully (we think)")
    
    
    
    

def extract_reasoning_from_context_check(response):
    # print("\n----\/----\n RESPONSE:")
    # print(response)
    # print("\n\n\n---/\---\n\n")
    decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
    determination = decision_pattern.search(response)
    if determination:
        determination = determination.group(1).strip()
    if not determination:
        print("Did not contain a determination! MEGA MODEL FAIL LOOK INTO THIS EVAN!!!")
        return None, response
    if "PASS" in determination:
        print("Leaving be...")
        return (True, response)  # , completion
    elif "REWORD" in determination:
        print("Rewording...")
        q, a = extract_question_answer.extract_question_answer(response)
        print((q, a))
        if "the provided" in a.lower(): # catch infrequent cases where the reworded answer contains reference to provided information
            print("'The provided' found in reworded answer -- Setting to None...")
            return (False, response)
        if "the reworded" in a.lower(): # Catch infrequent cases where it talks about the reworded question and answer pair
            print("'The reworded' found in reworded answer -- Setting to None...")
            return (False, response)
        if "mention" in a.lower():
            print("'Mention' found in reworded answer -- Setting to None...")
            return (False, response)
        if "no information" in a.lower():
            print("'No information' found in reworded answer -- Setting to None...")
            return (False, response)
        if "follow the instructions in a separate" in a.lower():
            print("'Follow the instructions in a separate' found in reworded answer -- Setting to None...")
            return (False, response)
        return (q, a)  # (q, a, qatuple[2], qatuple[3]), completion
    elif "FAIL" in determination:
        print("Setting to None...")
        return (False, response)  # , completion
    else:
        print("Did not contain relevant or irrelevant! Retrying")
        # print("!!! RESPONSE !!!")
        # print("\n\n\n---\/---\n\n")
        # print(response)
        # print("\n\n\n---/\---\n\n")
        raise Exception("error in judgement extraction (ans relevancy)")

# Postprocessing function for question/answer validation
async def repair_qatuple_context(
    idx,
    tup,
    engine_wrapper,
    writepath,
    vetted_qa_tuples,
    use_filenames=False,
    completion_mode=None,
    logging_level=logging.INFO,
):
    # NOTE set up the generation step
    context_repairer_path = "check_qatuple_context_no_filenames"
    if use_filenames:
        context_repairer_path = "check_qatuple_context_filenames"
    if completion_mode:
        context_repairer_path = context_repairer_path + ".txt"
    else:
        context_repairer_path = context_repairer_path + ".yaml"

    repair_context_regex = re.compile(
        r"Reasoning and thought process \(be thorough\):(.+)",
        re.DOTALL | re.IGNORECASE,
    )
    context_repairer = GenerationStep(
        prompt_path=context_repairer_path,
        regex=repair_context_regex,
        sampling_params={
            "max_tokens": 2000,
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
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=extract_reasoning_from_context_check,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
        use_stop=obj_conf["SYSTEM"]["STOP"]
    )

    # Resume normal control flow
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
                vetted_qa_tuples[idx] = (data[0], data[1], data[2], data[3], data[4], data[5], data[6])
                return None
            except json.JSONDecodeError:
                print("JSON decode error with the contents:", content)

    try:
        revision_id = make_id()
        revision, revision_output = await context_repairer.generate(
            arguments={
                "textname": tup[3],
                "question": tup[0],
                "answer": tup[1],
            }
        )
        write_output_to_file(
            revision_output,
            obj_conf["PATH"]["OUTPUT"] + "/question_context_revision_generations",
            revision_id,
        )  # incidentally, identifying the problem and fixing it in the same step (without another planning step) works a lot better than identifying it and then trying to fix it in the next step.
        if isinstance(revision[0], str):  # if the thing was reworded
            vetted_qa_tuples[idx] = (
                revision[0],
                revision[1],
                tup[2],
                tup[3],
                tup[4],
                tup[5],
                tup[6]
            )  # replace the old tuple with the new one, revision doesn't have text name so we keep the old one
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


def parse_answer_accuracy_validation(response):
    determination_pattern = re.compile(
        r"Overall Accuracy Determination:(.+)", re.DOTALL
    )
    try:
        determination = determination_pattern.search(response).group(1).strip()
    except Exception as e:
        print("Error encountered, model messed up output format")
        print(e)
        return (False, response)
    if (
        "inaccurate" in determination.lower()
        or "Inaccurate" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "irrelevant" in determination.lower()
    ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
        return (False, response)
    elif "accurate" in determination.lower():
        return (True, response)
    else:
        print("Answer accuracy validation made a mistake")
        raise Exception("answer accuracy validation did not include a judgement")


# Control flow helpers -- Question/Answer Validation
async def vet_answer_accuracy_loop(
    qa_tuple,
    run_id,
    engine_wrapper=None,
    double_check_counter=3,
    completion_mode=None,
    logging_level=None,
    file_path=None,
):
    # NOTE Set up answer check generation step
    prompt_path_ans_accuracy_check = "check_answer"
    if completion_mode:
        prompt_path_ans_accuracy_check = prompt_path_ans_accuracy_check + ".txt"
    else:
        prompt_path_ans_accuracy_check = prompt_path_ans_accuracy_check + ".yaml"
    check_ans_accuracy_regex = re.compile(
        r"Reasoning and thought process \(the text is your single source of truth\):\n(.+)",
        re.DOTALL,
    )
    # TODO performance improvement could be gained by using async for to do the checks simultaneously
    answer_accuracy_checker = GenerationStep(
        prompt_path=prompt_path_ans_accuracy_check,
        regex=check_ans_accuracy_regex,
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
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=parse_answer_accuracy_validation,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
        use_stop=obj_conf["SYSTEM"]["STOP"],
    )

    # Resume normal control flow code

    try:
        qtuple = qa_tuple
        # print(
        # f"\n\nStarting ACCURACY loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            check_id = make_id()
            # print(
            # f"\n\nACCURACY CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
            # )
            judgement, answer_accuracy_output = await answer_accuracy_checker.generate(
                arguments={
                    "text": qtuple[2],
                    "question": qtuple[0],
                    "answer": qtuple[1],
                }
            )
            write_output_to_file(
                answer_accuracy_output,
                obj_conf["PATH"]["OUTPUT"] + "/check_answer_accuracy_generations",
                run_id + "--check--" + check_id,
            )
            if not judgement[0]:  # if not accurate
                dissenting_reasoning = judgement[1]
                print("\nNegative Vote Cast! Here was the reasoning:\n")
                print(dissenting_reasoning)
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
            print("Answer accuracy validation failed! Tossing")
            with open(file_path, "w") as file:
                    file.write("failed")
            return
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    with open(file_path, "w") as file:
        file.write("failed")
    return


def parse_answer_relevancy_validation_step(thought_process):
    judgement_pattern = re.compile(
        r"Explanation of Judgment:(.+)", re.DOTALL | re.IGNORECASE
    )
    try:
        determination = judgement_pattern.search(thought_process).group(1).strip()
        if (
            "irrelevant" in determination.lower()
            or "mostly" in determination.lower()
            or "partial" in determination.lower()
            or "introduces information not present in the text" in determination.lower()
        ):  # Hack to get around faulty outputs
            return (False, thought_process)  # , completion
        elif "relevant" in determination or "Relevant" in determination:
            return (True, thought_process)  # , completion
        else:
            print(f"Answer relevancy parsing failed! Retrying! {judgement_pattern}")
            raise Exception("error in judgement extranction (ans relevancy)")
    except Exception as e:
        print("Model did not provide a judgement")
        print(e)
        # raise Exception("retry")
        return (False, thought_process)


async def vet_answer_relevance_loop(
    qa_tuple,
    run_id,
    engine_wrapper=None,
    double_check_counter=3,
    completion_mode=None,
    logging_level=None,
    file_path=None,
):
    # NOTE Set up answer check generation step
    prompt_path_ans_relevancy_check = "check_answer_relevancy_with_text"
    check_ans_relevancy_regex = re.compile(
        r"Reasoning and thought process \(be careful about extra details, even vague ones\):\n(.+)",
        re.DOTALL | re.IGNORECASE,
    )

    if completion_mode:
        prompt_path_ans_relevancy_check = prompt_path_ans_relevancy_check + ".txt"
    else:
        prompt_path_ans_relevancy_check = prompt_path_ans_relevancy_check + ".yaml"

    answer_relevancy_checker = GenerationStep(
        prompt_path=prompt_path_ans_relevancy_check,
        regex=check_ans_relevancy_regex,
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
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        output_processor=parse_answer_relevancy_validation_step,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
        use_stop=obj_conf["SYSTEM"]["STOP"]
    )

    # Resume normal control flow code
    try:
        qtuple = qa_tuple
        # print(
        # f"\n\nStarting RELEVANCE loop for question: {qtuple[0]}, context: {qtuple[2]}"
        # )
        passed_checks = 0
        times_checked = 0
        dissenting_reasoning = ""
        while times_checked < double_check_counter:
            check_id = make_id()
            
            # print(
            # f"\n\nRELEVANCE CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
            # )
            (
                judgement,
                answer_relevancy_output,
            ) = await answer_relevancy_checker.generate(
                arguments={
                    "text": qtuple[2],
                    "question": qtuple[0],
                    "answer": qtuple[1],
                }
            )
            write_output_to_file(
                answer_relevancy_output,
                obj_conf["PATH"]["OUTPUT"] + "/check_answer_relevancy_generations",
                check_id,
            )
            if not judgement[0]:  # if not relevant
                dissenting_reasoning = judgement[1]
                print("\nNegative Vote Cast! Here was the reasoning:\n")
                print(dissenting_reasoning)
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
            return await vet_answer_accuracy_loop(
                qtuple,
                run_id,
                engine_wrapper=engine_wrapper,
                double_check_counter=double_check_counter,
                completion_mode=completion_mode,
                logging_level=logging_level,
                file_path=file_path
            )
        else:
            print("Answer relevancy validation failed! Tossing")
            with open(file_path, "w") as file:
                    file.write("failed")
            return
    except Exception as e:
        print("!!ERROR!!")
        print(e)
        traceback.print_exc()

    with open(file_path, "w") as file:
        file.write("failed")
    return


def parse_validation_step(response):
    # print("!!! RESPONSE !!!")
    # print(response)
    decision_pattern = re.compile(r"Critical Evaluation and Final Judgment:(.+)", re.DOTALL | re.IGNORECASE)
    determination = decision_pattern.search(response).group(1).strip()
    # print("!!! DETERMINATION !!!")
    # print(determination)
    if (
        "irrelevant" in determination
        or "Irrelevant" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "introduces information not present in the text" in determination.lower()
    ):
        return (
            False,
            response,
        )  # TODO ensure that in the control flow code it passes on (False, response), completion
    elif "relevant" in determination.lower():
        return (True, response)  # TODO same as above(True, response), completion
    else:
        print("Did not contain relevant or irrelevant! Retrying")
        raise Exception(
            "Validation step screwed up and did not reach a conclusion! Retrying!"
        )


async def vet_question_loop(
    qa_tuple,
    question_group_id=None,
    engine_wrapper=None,
    qa_tuples_dir=None, # idx is qa_tuple[5]. Really should've used a dict at this point, oh well.
    vetted_qa_tuples=None,
    double_check_counter=3,
    completion_mode=None,
    logging_level=None,
):
    try:
        file_path = os.path.join(qa_tuples_dir, f"para_{qa_tuple[5]}_q_{qa_tuple[6]}.json")
        idx = qa_tuple[5]
        # Check for existing qa tuples
        existing_files = glob.glob(
            os.path.join(qa_tuples_dir, f"para_{idx}_q_{qa_tuple[6]}.json")
        )  # check if qs already exist

        if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
            print(f"Loading file")
            for file_path in existing_files:
                with open(file_path, "r") as file:
                    file_body = file.read()
                    if file_body == "failed":
                        qa_tuple = None
                    else:
                        file.seek(0)
                        qa_tuple = tuple(json.loads(file_body))
                vetted_qa_tuples.append(qa_tuple)
            return
        
        
        # NOTE Set up question check generation step
        prompt_path_q_check = "check_question"
        check_q_regex = re.compile(
            r"Reasoning and thought process \(be careful around \"how\" and \"why\" questions\):(.+)",
            re.DOTALL | re.IGNORECASE,
        )

        if completion_mode:
            prompt_path_q_check = prompt_path_q_check + ".txt"
        else:
            prompt_path_q_check = prompt_path_q_check + ".yaml"

        question_checker = GenerationStep(
            prompt_path=prompt_path_q_check,
            regex=check_q_regex,
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
            completion_mode=completion_mode,
            retries=1,
            engine_wrapper=engine_wrapper,
            logging_level=logging_level,
            output_processor=parse_validation_step,
            prompt_folder=obj_conf["PATH"]["PROMPTS"],
            default_prompt_folder=DEFAULT_PROMPT_PATH,
            use_stop=obj_conf["SYSTEM"]["STOP"],
        )

        # NOTE Set up generate new question step
        # MODIFICATION: so that the conversations make sense, we just toss failed questions, rather than regenning. They're plentiful enough.
        try:
            qtuple = qa_tuple
            # print(
            #     f"\n\nStarting QUESTION loop for question: {qtuple[0]}, context: {qtuple[2]}"
            # )
            run_id = question_group_id + "--subquestion--" + make_id()
            passed_checks = 0
            times_checked = 0
            dissenting_reasoning = ""
            if obj_conf["SKIP"]["QUESTION_CHECK"]:
                print("DEBUG: Skipping question check")
                return await vet_answer_accuracy_loop(
                    qtuple,
                    run_id,
                    engine_wrapper=engine_wrapper,
                    double_check_counter=double_check_counter,
                    completion_mode=completion_mode,
                    logging_level=logging_level,
                    file_path=file_path
                )
            while times_checked < double_check_counter:
                check_id = make_id()
                # print(
                #     f"\n\nQUESTION CALL CHECK ANSWER: {qtuple[0]}, context: {qtuple[2]}, retries: {total_retries}, dissenting reasoning: {dissenting_reasoning}"
                # )
                judgement, check_q_output = await question_checker.generate(
                    arguments={"text": qtuple[2], "question": qtuple[0], "answer": qtuple[1]}
                )

                # Now we need to put the judgement together into the format it expects it to be in

                write_output_to_file(
                    check_q_output,
                    obj_conf["PATH"]["OUTPUT"] + "/check_question_generations",
                    run_id + "--check--" + check_id,
                )
                
                # print("JUDGEMENT:")
                # print(judgement)
                if not judgement[0]:  # if not relevant
                    dissenting_reasoning = judgement[1]
                    print("\nNegative Vote Cast! Here was the reasoning:\n")
                    print(dissenting_reasoning)
                    print(f"ID: {check_id}")
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
                
                if obj_conf["SKIP"]["ANSWER_RELEVANCY_CHECK"]:
                    res = await vet_answer_accuracy_loop(
                        qtuple,
                        run_id,
                        engine_wrapper=engine_wrapper,
                        double_check_counter=double_check_counter,
                        completion_mode=completion_mode,
                        logging_level=logging_level,
                        file_path=file_path
                    )
                else:
                    res = await vet_answer_relevance_loop(
                        qtuple,
                        run_id,
                        engine_wrapper=engine_wrapper,
                        double_check_counter=double_check_counter,
                        completion_mode=completion_mode,
                        logging_level=logging_level,
                        file_path=file_path
                    )
                
                # Return response
                
                vetted_qa_tuples.append(res)
                if res is not None:
                    with open(file_path, "w") as file:
                        json.dump(res, file, indent=4)
                return
            else: # this path is probably redundant
                print("Question accuracy validation failed! Tossing")
                with open(file_path, "w") as file:
                    file.write("failed")
                return
        except Exception as e:
            print("!!ERROR!!")
            print(e)
            traceback.print_exc()
        with open(file_path, "w") as file:
            file.write("failed")
    except Exception as e:
        print(f"Q ERROR: {e}")
        traceback.print_exc()


def extract_questions_from_response(
    generation,
):  # TODO extract to non-controlflow file
    questions = extract_qa_tuples(generation)
    if len(questions) == 0:
        print("FAILED TO GENERATE QUESTIONS!")
        return []
    return questions


def extract_question_from_response(
    generation,
):  # TODO extract to non-controlflow file
    return extract_questions_from_response(generation)[0]


# Question generation
async def generate_qatuples_from_para(
    idx,
    para,
    engine_wrapper_large=None,
    generated_qa_tuples=None,
    qa_tuples_dir=None,
    use_filenames=False,
    completion_mode=None,
    logging_level=None,
):

    # NOTE Set up qatuple generation step #
    prompt_path_qatuples_gen = "qatuples_gen_no_filenames"
    if use_filenames:
        prompt_path_qatuples_gen = "qatuples_gen_filenames"

    if completion_mode:
        prompt_path_qatuples_gen = prompt_path_qatuples_gen + ".txt"
    else:
        prompt_path_qatuples_gen = prompt_path_qatuples_gen + ".yaml"

    qatuples_gen_regex = re.compile(
        r"Questions \(make 4\):\n(.+)", re.IGNORECASE | re.DOTALL
    )
    qatuples_generator = GenerationStep(
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
            # top_k=-1,
            "top_p": 1,
            # min_p=0.5,
        },
        completion_mode=completion_mode,
        retries=3,
        engine_wrapper=engine_wrapper_large,
        logging_level=logging_level,
        output_processor=extract_questions_from_response,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
    use_stop=obj_conf["SYSTEM"]["STOP"]
    )
    # Resume normal control flow code
    try:
        existing_files = glob.glob(
            os.path.join(qa_tuples_dir, f"para_{idx}_*.json")
        )  # check if qs already exist

        if len(existing_files) > 0:  # If files exist, skip this paragraph entirely
            print(f"Skipping para_{idx} as files already exist; loading said files")
            for file_path in existing_files:
                with open(file_path, "r") as file:
                    qa_tuple = tuple(json.load(file))
                generated_qa_tuples.append(qa_tuple)
            return
        question_group_id = make_id()
        # print(f"\n\n\nOUTER LOOP CALL GENERATE QPLAN para: {para}, \n\n idx: {idx}")
        # print(
        #     f"\n\n\nOUTER LOOP CALL GENERATE Q: {para}, \n\n idx: {idx} \n\n plan: {plan}"
        # )
        (
            question_answer_tuples,
            question_generation_output,
        ) = await qatuples_generator.generate(
            arguments={
                "text": para[0],
                "textdetails": para[1],
            }
        )

        question_answer_tuples_more_info = [
            (qatup[0], qatup[1], para[0], para[1], question_group_id, idx, qnum) for qnum, qatup in enumerate(question_answer_tuples)
        ]
        write_output_to_file(
            question_generation_output,
            obj_conf["PATH"]["OUTPUT"] + "/question_generation_generations",
            question_group_id,
        )
        
        for qatup in question_answer_tuples_more_info:
            generated_qa_tuples.append(qatup)
            if qatup[0] is not None:
                file_path = os.path.join(qa_tuples_dir, f"para_{qatup[5]}_q_{qatup[6]}.json")
                with open(file_path, "w") as file:
                    json.dump(qatup, file, indent=4)
            
    except Exception as e:
        print(f"Q ERROR: {e}")
        traceback.print_exc()


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

    # Filter out tuples with None and return the new list
    filtered_list = [t for t in tuples if t[0] is not None]
    return filtered_list


## Paragraph Filtering (worthy for questions?)
async def determine_worthy(
    idx,
    p,
    judged_worthy_for_questions,
    output_dir,
    judge: GenerationStep,
):
    # for idx, p in tqdm(enumerate(paragraphs_processed[:10])):
    id = make_id()
    
    file_name = f"{idx}.json"
    file_path = os.path.join(output_dir, file_name)
    # Check if the judgement for this paragraph already exists
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
            print("LOADING: ", data)
        if isinstance(data, str):
            judged_worthy_for_questions.append(
                (None, data[7:])
            )  # hacky way of appending only the text name. See the file output of a failed judgement for details (Takes after "failed|")
        else:
            judged_worthy_for_questions.append((data["paragraph"], data["metadata"]))
    else:
        judgement, judgement_output = await judge.generate(arguments={"text": p[0], "textname": p[1]})
        write_output_to_file(judgement_output, obj_conf["PATH"]["OUTPUT"] + "/judge_paragraph_generations", id)
        to_append = (None, p[1])
        if judgement:
            to_append = (p[0], p[1])

        judged_worthy_for_questions.append(to_append)

        # Prepare the data to be written to the file
        if judgement:
            # The paragraph passed the judgement
            data_to_write = {"paragraph": to_append[0], "metadata": to_append[1]}
        else:
            # The paragraph did not pass the judgement
            data_to_write = f"failed|{to_append[1]}"

        # Write the judgement to a unique file as JSON
        with open(file_path, "w") as file:
            json.dump(data_to_write, file)

        # Debug messages
        try:
            if judgement:
                print(f"DEBUG model decided that index {idx} was suitable")
            else:
                print(f"DEBUG model decided that index {idx} was not suitable")
        except:
            print(f"DEBUG max retries exceeded for index {idx}")


def judge_paragraph_processor(
    determination,
):  # TODO extract to separate file to avoid muddying the control flow code
    if "unsuitable" in determination.lower() or "table of contents" in determination.lower():
        return False  # control flow has been modified to use the information it has, based on the determination of the output processors
    elif "suitable" in determination.lower():
        return True


# EXEMPLAR
async def filter_all_questions(
    paragraphs_processed,
    judged_worthy_for_questions,
    engine_wrapper,
    output_dir,
    take_subset=False,
    subset_size=None,
    use_filenames=False,
    rtwl=None,
    completion_mode=None,
    logging_level=None,
):
    if use_filenames:
        prompt_path = "judge_paragraph_filenames"
    else:
        prompt_path = "judge_paragraph_no_filenames"

    judgement_regex = re.compile(
        r"Reasoning and thought process \(reason intelligently\):(.+)",
        re.DOTALL | re.IGNORECASE,
    )

    if completion_mode:
        prompt_path = prompt_path + ".txt"
    else:
        prompt_path = prompt_path + ".yaml"

    judge = GenerationStep(
        prompt_path=prompt_path,
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
        completion_mode=completion_mode,
        retries=2,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,  # TODO change to warning
        output_processor=judge_paragraph_processor,
        # return_input_too=False,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
        use_stop=obj_conf["SYSTEM"]["STOP"]
    )
    if not take_subset:
        tasks = [
            determine_worthy(idx, p, judged_worthy_for_questions, output_dir, judge)
            for idx, p in enumerate(paragraphs_processed)
        ]
    else:
        tasks = [
            determine_worthy(idx, p, judged_worthy_for_questions, output_dir, judge)
            for idx, p in enumerate(paragraphs_processed[:subset_size])
        ]
    limited_tasks = [rtwl(task) for task in tasks]
    for future in tqdmasyncio.tqdm.as_completed(limited_tasks):
        await future


def sentence_chunking_algorithm(file_path, max_char_length=1900):
    """
    This function takes a plaintext file and chunks it into paragraphs or sentences if the paragraph exceeds max_char_length.

    :param file_path: Path to the plaintext file
    :param max_char_length: The maximum char5acter length for a chunk
    :return: List of chunks with source text information
    """
    chunks_with_source = []
    current_chunk = []
    char_count = 0
    source_name = file_path.replace(".txt", "")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # try:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         content = f.read()
    # except Exception as e:
    #     print(f"\nError reading file {file_path}: {e}\n")
    #     return []

    paragraphs = content.split(
        "\n\n"
    )  # Assuming paragraphs are separated by two newlines # TODO change so that if the length is 1 after this, split by tabs instead

    # HOW TO DO IT probably:
    # add tokens to the paragraph until we reach the max length,
    # create chunks out of the remainder of the paragraph (split at max chunk length until it's done)
    # if the final chunk does not have the max length, then make it the new current chunk, set the current token count to its length, and continue with the for loop.

    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove leading and trailing whitespace
        if not paragraph:  # Skip empty paragraphs
            continue

        paragraph_char_count = len(paragraph)

        # Check if the paragraph itself exceeds the max token length
        if paragraph_char_count > max_char_length:

            # Fallback to character chunking for this paragraph
            end_index = (
                max_char_length - char_count
            )  # after this we will take max_char_length chunks starting from end index until the end of the paragraph
            current_chunk.append(paragraph[:end_index])
            # characters = list(paragraph)
            chunks_with_source.append(("".join(current_chunk), source_name))
            current_chunk = []
            while end_index < paragraph_char_count:
                current_chunk.append(paragraph[end_index : end_index + max_char_length])
                chunks_with_source.append(("".join(current_chunk), source_name))
                current_chunk = []
                end_index += max_char_length

            # # handle the remainder of the paragraph
            # end_index = end_index - max_char_length
            # current_chunk.append(paragraph[end_index:])

            # char_count = paragraph_char_count - end_index
        else:
            if char_count + paragraph_char_count <= max_char_length:
                current_chunk.append(paragraph)
                char_count += paragraph_char_count
            else:
                chunks_with_source.append(("".join(current_chunk), source_name))
                current_chunk = [paragraph]
                char_count = paragraph_char_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_source.append(("\n\n".join(current_chunk), source_name))
        
    # filter out chunks with fewer than 50 characters
    chunks_with_source = [chunk for chunk in chunks_with_source if len(chunk[0]) >= 50]

    return chunks_with_source


def fix_text(to_replace_arr, text):
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text


async def ensure_multiple_answers_are_same(
    info, conv, multi_turn_conv_generator, completion_mode=None, conversation_instructions="For this conversation, you are generating a chat between a general-purpose AI assistant and a human."
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
        retry = await make_multiturn_conversation(
            info, multi_turn_conv_generator, completion_mode=completion_mode, conversation_instructions=conversation_instructions
        )
        if retry is not None:  # Note: retry CANNOT actually be None
            c = retry
        else:
            # If we failed to generate a retry, don't waste compute
            return None

    return None



async def make_multiturn_conversation(
    info, multi_turn_conv_generator, completion_mode=None, conversation_instructions="For this conversation, you are generating a chat between a general-purpose AI assistant and a human."
):

    conv, conv_output = await multi_turn_conv_generator.generate(
        arguments={
            "question_answer_list": format_qatuples(info[0]).strip(),
            "conversation_instructions": conversation_instructions
        }
    )
    write_output_to_file(
        conv_output,
        obj_conf["PATH"]["OUTPUT"] + "/multiturn_conversation_generations",
        info[4],
    )

    return (conv, info[1], info[2], info[3], info[0])

async def create_info(
    idx,
    group,
    multi_turn_convs_info,
    multi_turn_convs_info_dir,
):

    file_path = os.path.join(multi_turn_convs_info_dir, f"info_{idx}.json")

    # Skip if file already exists
    if not os.path.exists(file_path):
        info = (group, "will", "be", "replaced", make_id())

        with open(file_path, "w") as file:
            json.dump(info, file, indent=4)
    else:
        with open(file_path, "r") as file:
            info = json.load(file)

    multi_turn_convs_info.append(
        [info]
    )  # hacky-looking things because the legacy functionality was simplified.

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
                and all(len(item) == 7 for item in data[0])
                and all(isinstance(i, str) for i in data[1:])
            ):
                tuple_list.append((data[0], data[1], data[2], data[3], data[4]))

    return tuple_list


async def create_conversation(
    idx,
    info,
    engine_wrapper,
    multi_turn_convs,
    multi_turn_convs_dir,
    completion_mode=None,
    logging_level=logging.INFO,
    conversation_instructions="For this conversation, you are generating a chat between a general-purpose AI assistant and a human."
):
    file_path = os.path.join(multi_turn_convs_dir, f"conv_{idx}.json")
    multi_turn_conversation_prompt_path = "multi_turn_assistant_conversation"

    conversation_regex = re.compile(
        f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; AI Assistant will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
        re.IGNORECASE | re.DOTALL,
    )

    if completion_mode:
        multi_turn_conversation_prompt_path = (
            multi_turn_conversation_prompt_path + ".txt"
        )
    else:
        multi_turn_conversation_prompt_path = (
            multi_turn_conversation_prompt_path + ".yaml"
        )

    multi_turn_conv_generator = GenerationStep(
        prompt_path=multi_turn_conversation_prompt_path,
        regex=conversation_regex,
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "### Response",
                "\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "### Information",
                "## Information",
                "## Instruction",
                "Name:",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.8,
            # "top_k": -1,
            "top_p": 1,
            # "min_p": 0.6,
        },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging_level,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
        use_stop=obj_conf["SYSTEM"]["STOP"],
    )

    # Skip if file already exists
    if not os.path.exists(file_path):
        try:
            conv = await make_multiturn_conversation(
                info, multi_turn_conv_generator, completion_mode=completion_mode, conversation_instructions=conversation_instructions
            )
            final_conv = await ensure_multiple_answers_are_same(
                info, conv, multi_turn_conv_generator, completion_mode=completion_mode, conversation_instructions=conversation_instructions
            )

            if final_conv is not None:
                final_conv = (
                    final_conv[0],
                    "AI Assistant",
                    "",
                    "N/A",
                    final_conv[4],
                )
                with open(file_path, "w") as file:
                    json.dump(final_conv, file, indent=4)

            multi_turn_convs.append(final_conv)
        except Exception as e:
            traceback.print_exc()
            print("Had an error, retrying...", e)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                multi_turn_convs.append(data)
            print(f"Skipped generating {file_path} as it already exists")
        except Exception as e:
            print(f"Error reading {file_path}:", e)
            print("Continuing...")


def convert_directory_to_list(directory_path):
    master_list = []
    simplified_list = []
    simplified_rag_list = []

    for filename in os.listdir(directory_path):  # for each file
        if filename.endswith(".json"):  # if it's a conversation file
            filepath = os.path.join(directory_path, filename)  # get the path
            with open(filepath, "r") as file:  # open it
                try:
                    data = json.load(file)  # load its data
                    if isinstance(data, list) and all(
                        isinstance(item, (list, str))
                        for item in data  # if it has the correct format
                    ):

                        data_dict = {
                            "conversation": data[0],
                            "qa_tuples": [
                                tup[:2] for tup in data[4]
                            ],  # only take first two items from each tuple
                            "rag_context": data[4][0][2],
                            "source_filename": data[4][0][3],
                        }
                        master_list.append(
                            data_dict
                        )  # append it as-is to the master-list

                        # Extract and process conversation
                        conversation, primary_char_desc = (
                            data[0],
                            data[1],
                        )  # first and second items are conv and char desc
                        dialogues = process_multiturn_functions.extract_conversation(
                            conversation
                        )

                        # Convert to simplified format
                        simplified_conversations = []
                        simplified_conversations_rag = []

                        # Load system prompts
                        system_prompt_norag = obj_conf["SYSTEM"][
                            "FINAL_ASSISTANT_PROMPT_NO_RAG"
                        ]
                        system_prompt_rag = obj_conf["SYSTEM"][
                            "FINAL_ASSISTANT_PROMPT_RAG"
                        ]
                        simplified_conversations.append(
                            {"from": "system", "value": system_prompt_norag}
                        )

                        simplified_conversations_rag.append(
                            {
                                "from": "system",
                                "value": system_prompt_rag.replace(
                                    "{data}", data_dict["rag_context"]
                                ),
                            }
                        )
                        for i, (charname, message) in enumerate(
                            dialogues
                        ):  # Skipping the first message
                            from_person = "human" if (i % 2) == 0 else "gpt"
                            simplified_conversations.append(
                                {"from": from_person, "value": f"{message}"}
                            )
                            simplified_conversations_rag.append(
                                {
                                    "from": from_person,
                                    "value": f"{message}",
                                }  # same as above, but for the RAG context
                            )

                        if simplified_conversations:  # If there are any conversations
                            simplified_list.append(
                                {"conversations": simplified_conversations}
                            )
                            simplified_rag_list.append(
                                {"conversations": simplified_conversations_rag}
                            )
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # Write the master list to a new .jsonl file
    write_1 = obj_conf["PATH"]["OUTPUT"] + "/master_list.jsonl"
    with open(write_1, "w") as file:
        for item in master_list:
            file.write(json.dumps(item) + "\n")

    # Write the simplified data to a different .jsonl file
    write_2 = obj_conf["PATH"]["OUTPUT"] + "/simplified_data_no_rag.jsonl"
    with open(write_2, "w") as file:
        for item in simplified_list:
            file.write(json.dumps(item) + "\n")

    write_3 = obj_conf["PATH"]["OUTPUT"] + "/simplified_data_rag.jsonl"
    with open(write_3, "w") as file:
        for item in simplified_rag_list:
            file.write(json.dumps(item) + "\n")

    print(
        f"Conversion complete. Master list written to {write_1}. Simplified data written to {write_2} (no RAG) and {write_3} (RAG)."
    )


def convert_directory_and_process_conversations(directory_path):
    master_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as file:
                try:
                    data = json.load(file)

                    if isinstance(data, list) and all(
                        isinstance(item, (list, str)) for item in data
                    ):
                        # Extract and process the conversation part
                        conversations = (
                            process_multiturn_functions.extract_conversation(data[0])
                        )
                        # Convert tuples back to the formatted string as required
                        data[0] = [
                            f"{charname}: {message}"
                            for charname, message in conversations
                        ]
                        master_list.append(data)
                    else:
                        print(f"File {filename} is not in the expected format.")
                except:
                    print(f"Error reading {filename}")

    # Write the master list to a new file
    with open(obj_conf["PATH"]["OUTPUT"] + "/processed_master_list.json", "w") as file:
        json.dump(master_list, file)

    print(
        "Conversion complete. The processed master list is written to 'processed_master_list.json'."
    )

def create_pretraining_set(directory_path, json_file):
    # Initialize a variable to store the combined text of all files
    combined_text = ""

    # Walk through all directories and files in the directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Read the contents of the file
            with open(file_path, "r") as file:
                file_contents = file.read()

            # Append the file contents to the combined text, with a separator
            if combined_text:
                combined_text += "\n\n---NEW FILE---\n\n"
            combined_text += file_contents

    # Create a dictionary with the combined text
    data = {"text": combined_text}

    # Save the dictionary as a JSON file
    with open(json_file, "w") as file:
        json.dump(data, file)

    print("JSON file saved successfully.")

def create_pretraining_set(directory_path, json_file):
    # Initialize a variable to store the combined text of all files
    combined_text = ""

    # Walk through all directories and files in the directory
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Read the contents of the file
            with open(file_path, "r") as file:
                file_contents = file.read()

            # Append the file contents to the combined text, with a separator
            if combined_text:
                combined_text += "\n\n---NEW FILE---\n\n"
            combined_text += file_contents

    # Create a dictionary with the combined text
    data = {"text": combined_text}

    # Save the dictionary as a JSON file
    with open(json_file, "w") as file:
        json.dump(data, file)

    print("JSON file saved successfully.")
