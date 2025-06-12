import random
import itertools
import os
import asyncio
import json
import re
from typing import List
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from transformers import AutoTokenizer
import logging
from math import ceil
import traceback
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
import uuid
import yaml
import nltk
from augmentoolkit.utils import parse_string_list
from augmentoolkit.utils.parse_bool import parse_bool
from pathlib import Path
from jinja2 import Template


def extract_atomic_facts(output):
    # get thing in-between <atomic_facts> and </atomic_facts>
    # return that
    return output.split("<atomic_facts>")[1].split("</atomic_facts>")[0]


def stringify_rag_chunks(input_chunks):
    result_string = ""
    for idx, chunk in enumerate(input_chunks):
        result_string += f"""Chunk {idx}
Source: {chunk["metadata"]}
---
{chunk["text"]}
---

"""
    return result_string


# TODO have a great way of stringifying the rag chunks things


def extract_qa_tuples(text):
    pattern = r"\*\*QUESTION:\*\*\s*((?:.|\n)*?)\s*\*\*ANSWER:\*\*\s*((?:.|\n)*?)(?=\s*\*\*QUESTION:\*\*|\Z)"
    matches = re.findall(
        pattern, text + "\n\n**QUESTION:**", re.DOTALL
    )  # The addition is a hack to get around the tricky lookahead problem
    return [
        {"question": question.strip(), "answer": answer.strip()}
        for question, answer in matches
    ]


# NOTE below was my stream-of-consciousness thought process for building this masking system. I preserved it for reference.
# Basic idea:
# singleturn or multiturn with followups? I think we can do both in the same pipeline. Here's how:
# each group of qadicts by ground truth gets its order preserved and is kept together when shuffling and combining with other groups.
# and the prompts are for either single or multiturn. And we parse out each input/output pair like ATK original.
# So, if everything's singleturn we just get a bunch of independent singleturn things and then we can group it together and mask it all nice
# if it's mutliturn the logic is the same, it just naturally adapts since for each completion we make the RAG'd context the same as that stuff was generated with initially.
# Masking used on previous messages + the system prompt every time. Only thing we train on is the response here. Well maybe we do the sysprompt too. But since previous messages may be referring to RAG'd context that HAS CHANGED AND IS NO LONGER THERE we do not want to train it to hallucinate, so after combining we make masks for each group (this means that when combining we need to keep track of at what index each group starts and ends, so we can make masks for each group).
# by masks for each group I mean, say we combine groups 1 2 and 3 into a conversation. We can therefore make 3 completions out of this. For 1, train on the sysprompt + group 1 and then mask groups 2 and 3 so we don't learn from them. For group 2, train on sysprompt, mask group 1, train on group 2, mask group 3. For group 3, train on sysprompt, mask groups 1 and 2, train on group 3.


def save_combined_conversations(
    combined_conversations,
    output_dir,
    num_items_per_group,
    final_assistant_prompts,
    system_template,
    user_format,
    assistant_format,
    bos,
):
    """Convert RAG conversations into Axolotl-compatible format with proper masking

    NOTE we are doing this entirely differently than the big code comment you see below. Generate only what the specific instruction says and no more.

    Args:
        combined_conversations: List of dicts containing rag_conversation entries
    """
    axolotl_conversations = []

    # Shuffle and group conversations
    random.shuffle(combined_conversations)

    # Group conversations
    conversation_groups = []
    for i in range(0, len(combined_conversations), num_items_per_group):
        group = combined_conversations[i : i + num_items_per_group]
        if group:  # Ensure we don't add empty groups
            conversation_groups.append(group)

    for group in conversation_groups:

        for idx, item in enumerate(
            group
        ):  # we will construct each masked sequence thing one part at a time
            # create one conversation for each group and append them to training_data_convs. In a group's conversation, everything except that group is masked with label: False. So it will look like: group 1 conv = sysprompt masked group1 not masked groups 2 and 3 masked. group 2 conv = sysprompt group 1 label false (masked) group 2 label True group 3 masked.
            # Create a conversation for this item in the group
            prompt_text = random.sample(final_assistant_prompts, 1)[
                0
            ]  # where final assistant prompts is empty, that is where an assert would be useful for instance
            # I can really tidy up my code style I think
            # first thing though is to get ATK 3.0 shippable
            # spending a whole week focusing on one task.... to get it out of the way. That feels promising.
            prompt_text = prompt_text.replace(
                "{data}", f"{stringify_rag_chunks(item['rag_chunks'])}"
            )
            segments = []

            segments.append(
                {
                    "label": False,
                    "text": bos + Template(system_template).render(system=prompt_text),
                }
            )

            # Process each item in the group
            for inner_idx, inner_item in enumerate(group):
                # Get the conversation data for this item
                qa_pairs = inner_item["rag_conversation"]

                # For each QA pair in this item
                for qa in qa_pairs:
                    # Render these messages using the format templates
                    rendered_user = Template(user_format).render(user=qa["question"])
                    rendered_assistant = Template(assistant_format).render(
                        assistant=qa["answer"]
                    )

                    # Add to segments with appropriate masking
                    # If this is the current item we're focusing on, label as True (unmasked)
                    # Otherwise, label as False (masked)
                    segments.append({"label": inner_idx == idx, "text": rendered_user})

                    segments.append(
                        {"label": inner_idx == idx, "text": rendered_assistant}
                    )

            # Add this conversation to our training data
            axolotl_conversations.append({"segments": segments})

    # Save the axolotl conversations to a file
    # output_file = os.path.join(output_dir, "axolotl_conversations.json")

    # # Ensure the output directory exists
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # # Write the conversations to the file
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(axolotl_conversations, f, ensure_ascii=False, indent=2)

    # print(f"Saved {len(axolotl_conversations)} axolotl conversations to {output_file}")

    # Save to file
    output_path = os.path.join(output_dir, "axolotl_rag_conversations.jsonl")
    with open(output_path, "w") as f:
        for conv in axolotl_conversations:
            f.write(json.dumps(conv) + "\n")
