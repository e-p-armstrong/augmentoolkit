import random
from bs4 import BeautifulSoup
from logging import INFO
import os
import json
import re
import sys
import requests
from tqdm import asyncio as tqdmasyncio
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
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
from datasets import load_dataset
import hashlib


from augmentoolkit.utils.create_conv_starter import create_conv_starter
from augmentoolkit.utils.extract_steps import extract_steps
from augmentoolkit.utils.escape_unescaped_quotes import escape_unescaped_quotes

from augmentoolkit.generation_functions import (
    extract_question_answer,
    process_multiturn_functions,
)

from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.generation_functions.special_instructions import special_instructions


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise ValueError(f"Cannot parse '{value}' as boolean")


has_pushed_yet = False


def extract_qa_tuples(text):
    pattern = r"\*\*QUESTION:\*\*\s*((?:.|\n)*?)\s*\*\*ANSWER:\*\*\s*((?:.|\n)*?)(?=\s*\*\*QUESTION:\*\*|\Z)"
    matches = re.findall(
        pattern, text + "\n\n**QUESTION:**", re.DOTALL
    )  # The addition is a hack to get around the tricky lookahead problem
    return [
        {"question": question.strip(), "answer": answer.strip()}
        for question, answer in matches
    ]


import os


def extract_reasoning_from_context_check(
    response,
):  # we will keep this simple, for now. It is likely that we will want to add in 1. multiple key assignments from a single step, and 2. passing the input data into output processors as a required thing. However, we also need to ship. Keeping scope creep to an absolute minimum for now. If we need custom logic, we can subclass or even better, just assign it/clean it up outside the classes,  with normal code.
    decision_pattern = re.compile(r"Final judgment:(.+)", re.IGNORECASE)
    determination = decision_pattern.search(response)
    if determination:
        determination = determination.group(1).strip()
    if not determination:
        print(
            "LLM ISSUE: Did not contain a determination! Maybe check your LLM it is being stupid, or perhaps the input is diffuclt."
        )
        print(response)
        return None
    if "PASS" in determination:
        print("Leaving be...")
        return True  # , completion. Also you know, there's the possibility that we might want to be able to save multiple different items to multiple different keys. The current input processor thing is a hack.
    elif "REWORD" in determination:
        print("Rewording...")
        q, a = extract_question_answer.extract_question_answer(response)
        print((q, a))
        if (
            "the provided" in a.lower()
        ):  # catch infrequent cases where the reworded answer contains reference to provided information
            print("'The provided' found in reworded answer -- Setting to None...")
            return False
        if (
            "the reworded" in a.lower()
        ):  # Catch infrequent cases where it talks about the reworded question and answer pair
            print("'The reworded' found in reworded answer -- Setting to None...")
            return False
        if "mention" in a.lower():
            print("'Mention' found in reworded answer -- Setting to None...")
            return False
        if "no information" in a.lower():
            print("'No information' found in reworded answer -- Setting to None...")
            return False
        if "follow the instructions in a separate" in a.lower():
            print(
                "'Follow the instructions in a separate' found in reworded answer -- Setting to None..."
            )
            return False
        return (q, a)  # (q, a, qatuple[2], qatuple[3]), completion
    elif "FAIL" in determination:
        print("Setting to None...")
        return False  # , completion
    else:
        print("Did not contain relevant or irrelevant! Retrying")
        raise Exception("error in judgement extraction (ans relevancy)")


### CONTEXT REPAIR SECTION


def parse_answer_accuracy_validation(response):
    determination_pattern = re.compile(
        r"Overall Accuracy Determination:(.+)", re.DOTALL
    )
    try:
        determination = determination_pattern.search(response).group(1).strip()
    except Exception as e:
        print("Error encountered, model messed up output format")
        print(e)
        return False
    if (
        "inaccurate" in determination.lower()
        or "Inaccurate" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "irrelevant" in determination.lower()
    ):  # The "mostly" is there to catch "mostly accurate" which the model says occasionally, and which actually means inaccurate.
        return False
    elif "accurate" in determination.lower():
        return True
    else:
        print("Answer accuracy validation made a mistake")
        raise Exception("answer accuracy validation did not include a judgement")


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
            return False  # , completion
        elif "relevant" in determination or "Relevant" in determination:
            return True  # , completion
        else:
            print(f"Answer relevancy parsing failed! Retrying! {judgement_pattern}")
            raise Exception("error in judgement extranction (ans relevancy)")
    except Exception as e:
        print("Model did not provide a judgement")
        print(e)
        # raise Exception("retry")
        return False


def parse_validation_step(response):
    # print("!!! RESPONSE !!!")
    # print(response)
    decision_pattern = re.compile(r"Final Judgment:(.+)", re.DOTALL | re.IGNORECASE)
    determination = decision_pattern.search(response).group(1).strip()
    # print("!!! DETERMINATION !!!")
    # print(determination)
    if (
        "irrelevant" in determination.lower()
        or "Irrelevant" in determination.lower()
        or "mostly" in determination.lower()
        or "partial" in determination.lower()
        or "introduces information not present in the text" in determination.lower()
    ):
        return False  # TODO ensure that in the control flow code it passes on False, completion
    elif "relevant" in determination.lower():
        return True  # TODO same as aboveTrue, completion
    else:
        print("Did not contain relevant or irrelevant! Retrying")
        raise Exception(
            "Validation step screwed up and did not reach a conclusion! Retrying!"
        )


### Question Generation Section


def extract_questions_from_response(
    generation,
):  # TODO extract to non-controlflow file
    # replace any instances of **QUESTION 1:** with **QUESTION:**
    # in fact, for any digit, replace with nothing
    generation = re.sub(r"\*\*QUESTION \d:\*\*", "**QUESTION:**", generation)
    questions = extract_qa_tuples(generation)
    if len(questions) == 0:
        print("FAILED TO GENERATE QUESTIONS!")
        return []
    return questions


### JUDGEMENT SECTION


def judge_paragraph_processor(
    determination,
):  # TODO extract to separate file to avoid muddying the control flow code
    if (
        "unsuitable" in determination.lower()
        or "table of contents" in determination.lower()
    ):
        return False  # control flow has been modified to use the information it has, based on the determination of the output processors
    elif "suitable" in determination.lower():
        return True


### CONVERSATION CREATION SECTION

multi_turn_conversation_prompt_path = "multi_turn_assistant_conversation"

conversation_regex = re.compile(
    f"Conversation that answers the provided question \(be sure that you do not change the questions or answers themselves; AI Assistant will answer the questions, not ask them; the questions and answers provided should be copied word for word, and surrounded by compelling conversation\):\n(.+)",
    re.IGNORECASE | re.DOTALL,
)


class ConversationGenerator(PipelineStep):
    def __init__(self):
        super().__init__(
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
                "top_p": 0.9,
                # "min_p": 0.6,
            },
            output_file="conversations_and_questiongroups",
            result_key="conversation",
            max_retries=3,
            # validation_function=process_multiturn_functions.call_all_processors, # NOTE somehow, SOMEHOW this vlaidation function was impacting the final format of the data outputs... bizarre. Be cautious of this one.
            output_processor=process_multiturn_functions.extract_conversation,
            details_key="conversation_details",
        )

    def process_input_data(self, input_data):
        # First, the keys of any given input data will be stringified numbers. like '1', '0', etc.
        # Create a list of their values, sorted by their keys
        qa_pairs = []
        for item in input_data["sorted"]:
            qa_pairs.append(
                f"**QUESTION:**\n{item['question']}\n\n**ANSWER**:\n{item['answer']}"
            )
        qa_pairs_string = "\n\n".join(qa_pairs)
        input_data["question_answer_pairs_string"] = qa_pairs_string
        return super().process_input_data(input_data)


conversation_generator_step = ConversationGenerator()


import os
import json
import random
from datasets import load_dataset


def save_conversations(
    text_hash_groups,
    final_assistant_prompts_rag,
    final_assistant_prompts_no_rag,
    rag_failure_percentage,
    do_not_use_system_prompts,
    items_per_conversation,  # this parameter isn't needed here but kept for consistency
    output_dir,
    hub_path,
    push_to_hub,
):
    simplified_conversation_list = []
    simplified_conversation_rag_list = []

    # Convert values to list for random sampling
    text_entries_list = list(text_hash_groups.values())

    for text_entry in text_entries_list:
        conversations_raw = text_entry.get("conversation", [])
        if not conversations_raw:
            continue  # skip if no conversation is defined

        # Prepare conversation lists
        conversations = []
        conversations_rag = []

        # RAG system prompt with paragraph (with configurable failure rate)
        system_prompt_rag = random.choice(final_assistant_prompts_rag)
        if random.random() < rag_failure_percentage:
            paragraph_entry = random.choice(text_entries_list)
        else:
            paragraph_entry = text_entry

        # Extract paragraph text correctly
        paragraph_text = paragraph_entry["sorted"][0]["text"]
        conversations_rag.append(
            {
                "from": "system",
                "value": system_prompt_rag.replace(
                    "{data}", paragraph_text.replace("\n", "\\n")
                ),
            }
        )

        # Non-RAG system prompt
        if not do_not_use_system_prompts:
            system_prompt_norag = random.choice(final_assistant_prompts_no_rag)
            conversations.append({"from": "system", "value": system_prompt_norag})

        # Append conversation turns
        for turn in conversations_raw:
            if len(turn) != 2:
                continue  # Skip malformed turns
            speaker_label = turn[0].lower().replace(":", "").strip()
            message = turn[1].strip()
            if "user" in speaker_label:
                role = "human"
            else:
                role = "gpt"
            conversations.append({"from": role, "value": message})
            conversations_rag.append({"from": role, "value": message})

        # Add to final lists
        simplified_conversation_list.append({"conversations": conversations})
        simplified_conversation_rag_list.append({"conversations": conversations_rag})

    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Write simplified conversation JSONL (no RAG)
    simplified_write_norag = os.path.join(
        output_dir, "simplified_conversation_list.jsonl"
    )
    with open(simplified_write_norag, "w", encoding="utf-8") as f:
        for item in simplified_conversation_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write simplified conversation RAG JSONL
    simplified_write_rag = os.path.join(
        output_dir, "simplified_conversation_rag_list.jsonl"
    )
    with open(simplified_write_rag, "w", encoding="utf-8") as f:
        for item in simplified_conversation_rag_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Optional push to Hugging Face Hub
    if push_to_hub:
        # Push simplified conversation data (no RAG)
        temp_norag = os.path.join(output_dir, "temp_simplified_conversation.json")
        with open(temp_norag, "w", encoding="utf-8") as temp_file:
            json.dump(
                {"train": simplified_conversation_list}, temp_file, ensure_ascii=False
            )

        dataset_norag = load_dataset("json", data_files=temp_norag, split="train")
        dataset_norag.to_parquet(
            f"hf://datasets/{hub_path}/data/train-conversation.parquet"
        )
        os.remove(temp_norag)

        # Push simplified conversation RAG data
        temp_rag = os.path.join(output_dir, "temp_simplified_conversation_rag.json")
        with open(temp_rag, "w", encoding="utf-8") as temp_file:
            json.dump(
                {"train": simplified_conversation_rag_list},
                temp_file,
                ensure_ascii=False,
            )

        dataset_rag = load_dataset("json", data_files=temp_rag, split="train")
        dataset_rag.to_parquet(
            f"hf://datasets/{hub_path}/data/train-conversation-rag.parquet"
        )
        os.remove(temp_rag)

    # Inform user
    print(
        f"Conversation data saved. Simplified conversations written to {simplified_write_norag}."
    )
    print(f"Simplified RAG conversations written to {simplified_write_rag}.")

    if push_to_hub:
        print("Conversation data successfully pushed to Hugging Face Hub.")


# NOTE to self: will have to find a way to assign indices to the questions generated, because we need that to maintain order in save plain qatuples. We need to retain the original ordering. And so, we need to save the index during question generation. This is an overall requirement of the one to many -- we need items to remember what index they are. Actually hell we don't need to hash the "index + content" we can just have [texthash]-index-[outputhash] yeah that'll work. that's how it's grouped now. And taht is how it will be loaded!


def save_plain_qatuples(
    qa_dicts_by_text,
    final_assistant_prompts_rag,
    final_assistant_prompts_no_rag,
    rag_failure_percentage,
    do_not_use_system_prompts,
    items_per_conversation,
    output_dir,
    hub_path,
    push_to_hub,
):
    # Helper function to generate deterministic seed from content
    def get_seed(s):
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 10**8

    plain_qa_list = []
    simplified_rag_list = []

    # Separate single-item and multi-item dicts
    single_item_dicts = []
    multi_item_dicts = []

    for text_entry in qa_dicts_by_text.values():
        dict_list = text_entry["sorted"]
        if len(dict_list) == 1:
            single_item_dicts.append(text_entry)
        else:
            multi_item_dicts.append(text_entry)

    # Convert qa_dicts_by_text values to a list for random sampling
    qa_text_entries_list = list(qa_dicts_by_text.values())

    # Process multi-item dicts first
    for text_entry in multi_item_dicts:
        conversations = []
        conversations_rag = []
        dict_list = text_entry["sorted"]
        paragraph_text = text_entry["sorted"][0]["text"]  # [1]

        # Deterministic system prompt choice
        text_seed = get_seed(paragraph_text)
        rand = random.Random(text_seed)
        system_prompt_rag = rand.choice(final_assistant_prompts_rag)

        # Deterministic paragraph choice for RAG failure
        if rand.random() < rag_failure_percentage:
            para_seed = get_seed(paragraph_text)
            para_rand = random.Random(para_seed)
            paragraph_entry = para_rand.choice(qa_text_entries_list)
        else:
            paragraph_entry = text_entry
        paragraph = paragraph_entry["sorted"][0][
            "text"
        ]  # using repaired_context for paragraph
        conversations_rag.append(
            {
                "from": "system",
                "value": system_prompt_rag.replace(
                    "{data}", paragraph.replace("\n", "\\n")
                ),
            }
        )

        # Non-RAG system prompt
        if not do_not_use_system_prompts:
            system_prompt_norag = rand.choice(final_assistant_prompts_no_rag)
            conversations.append({"from": "system", "value": system_prompt_norag})

        # Add QA pairs
        for d in dict_list:
            q = d["question"]
            a = d["answer"]
            conversations.append({"from": "human", "value": q})
            conversations.append({"from": "gpt", "value": a})
            conversations_rag.append({"from": "human", "value": q})
            conversations_rag.append({"from": "gpt", "value": a})

        plain_qa_list.append({"conversations": conversations})
        simplified_rag_list.append({"conversations": conversations_rag})

    # Process single-item dicts with deterministic ordering
    # Sort by hash of content instead of shuffling
    single_item_dicts.sort(key=lambda x: get_seed(x["sorted"][0]["text"]))

    for idx in range(0, len(single_item_dicts), items_per_conversation):
        combined_entries = single_item_dicts[idx : idx + items_per_conversation]
        conversations = []
        conversations_rag = []
        # Generate seed from combined entries
        combined_seed = get_seed(
            "".join(e["sorted"][0]["text"] for e in combined_entries)
        )
        entry_rand = random.Random(combined_seed)

        # Deterministic system prompt choice
        system_prompt_rag = entry_rand.choice(final_assistant_prompts_rag)

        # Deterministic paragraph choice for RAG failure
        if entry_rand.random() < rag_failure_percentage:
            para_seed = get_seed(
                "".join(e["sorted"][0]["text"] for e in combined_entries)
            )
            para_rand = random.Random(para_seed)
            paragraph_entry = para_rand.choice(qa_text_entries_list)
        else:
            paragraph_entry = combined_entries[0]
        paragraph = paragraph_entry["sorted"][0]["text"]
        conversations_rag.append(
            {
                "from": "system",
                "value": system_prompt_rag.replace(
                    "{data}", paragraph.replace("\n", "\\n")
                ),
            }
        )

        # Non-RAG system prompt
        if not do_not_use_system_prompts:
            system_prompt_norag = entry_rand.choice(final_assistant_prompts_no_rag)
            conversations.append({"from": "system", "value": system_prompt_norag})

        # Add QA pairs
        for entry in combined_entries:
            qa_pair = entry["sorted"][0]
            q = qa_pair["question"]
            a = qa_pair["answer"]
            conversations.append({"from": "human", "value": q})
            conversations.append({"from": "gpt", "value": a})
            conversations_rag.append({"from": "human", "value": q})
            conversations_rag.append({"from": "gpt", "value": a})

        plain_qa_list.append({"conversations": conversations})
        simplified_rag_list.append({"conversations": conversations_rag})

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    write_plain = os.path.join(output_dir, "plain_qa_list.jsonl")
    with open(write_plain, "w", encoding="utf-8") as file:
        for item in plain_qa_list:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_rag = os.path.join(output_dir, "simplified_data_rag.jsonl")
    with open(write_rag, "w", encoding="utf-8") as file:
        for item in simplified_rag_list:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    if push_to_hub:
        # Push plain QA data to hub
        temp_plain = os.path.join(output_dir, "temp_plain_qa.json")
        with open(temp_plain, "w", encoding="utf-8") as temp_file:
            json.dump({"train": plain_qa_list}, temp_file, ensure_ascii=False)

        dataset_plain = load_dataset("json", data_files=temp_plain, split="train")
        dataset_plain.to_parquet(f"hf://datasets/{hub_path}/data/train-plain.parquet")
        os.remove(temp_plain)

        # Push RAG data to hub
        temp_rag = os.path.join(output_dir, "temp_simplified_data_rag.json")
        with open(temp_rag, "w", encoding="utf-8") as temp_file:
            json.dump({"train": simplified_rag_list}, temp_file, ensure_ascii=False)

        dataset_rag = load_dataset("json", data_files=temp_rag, split="train")
        dataset_rag.to_parquet(f"hf://datasets/{hub_path}/data/train-rag.parquet")
        os.remove(temp_rag)

    print(
        f"Conversion complete. Plain QA data written to {write_plain}. Simplified RAG data written to {write_rag}."
    )
    if push_to_hub:
        print("Data successfully pushed to Hugging Face Hub.")

    return plain_qa_list, simplified_rag_list


### SCRAPING ###


def download_book(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    book_id = url.split("/")[-1]
    txt_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    response = requests.get(txt_url)
    if response.status_code == 200:
        filename = os.path.join(folder, f"{book_id}.txt")
        if not os.path.exists(filename):
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded: {filename}")
        else:
            print(f"Already downloaded: {filename}")
        return True
    else:
        print(f"Failed to download: {txt_url}")
        return False


def scrape_and_download(
    url,
    out_folder,
    max_books,
    books_downloaded,
    consecutive_failures,
    max_consecutive_failures,
):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        if books_downloaded >= max_books:
            return books_downloaded, consecutive_failures

        href = link.get("href")
        if href and href.startswith("/ebooks/"):
            full_url = f"https://www.gutenberg.org{href}"
            if full_url.count("/") == 4:  # This is likely a book link
                if download_book(full_url, out_folder):
                    books_downloaded += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            f"Aborting: {max_consecutive_failures} consecutive download failures"
                        )
                        return books_downloaded, consecutive_failures

                if books_downloaded >= max_books:
                    return books_downloaded, consecutive_failures

    return books_downloaded, consecutive_failures


def scrape_text_read_files_manually(
    start_url="", max_books="", max_failures="", input_dir=None
):

    books_downloaded = 0
    page_index = 0
    consecutive_failures = 0

    while books_downloaded < max_books and consecutive_failures < max_failures:
        current_url = (
            start_url
            if page_index == 0
            else f"{start_url}&start_index={page_index * 25 + 1}"
        )
        books_downloaded, consecutive_failures = scrape_and_download(
            current_url,
            input_dir,
            max_books,
            books_downloaded,
            consecutive_failures,
            max_failures,
        )

        if books_downloaded >= max_books or consecutive_failures >= max_failures:
            break

        page_index += 1

    print(f"Total books downloaded: {books_downloaded}")
    if consecutive_failures >= max_failures:
        print(
            f"Scraping aborted due to {consecutive_failures} consecutive download failures"
        )
