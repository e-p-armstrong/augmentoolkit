import os
import json
import re
from augmentoolkit.generation_functions.depth_first_pipeline_step_class import (
    DepthFirstPipelineStep,
)
from collections import defaultdict, deque
import traceback
from augmentoolkit.generation_functions.generalized_parsing_and_writing_formats import (
    extract_structured_data,
)
from generation.core_components.chunking import count_tokens


# Helpers for said abstraction
def validate_length_callback(
    length,
):  # returns a function that checks if a string is a certain length
    def inner(string, input_data):
        return count_tokens(string) <= length

    return inner


def find_repetitions(text, min_length):
    pattern = r"(.{" + str(min_length) + r",}?)\1+"
    repetitions = re.finditer(pattern, text)

    return repetitions


def validate_consecutive_repetition_callback(
    min_length,
):  # returns a function that checks if a string has a repetition of a certain length
    def inner(string, input_data):
        return len(list(find_repetitions(string, min_length))) == 0

    return inner


def find_frequent_substrings(text, min_length, min_occurrences, cluster_threshold):
    def update_counts(substring, index):
        # Update positions and remove indices that are out of the current cluster scope
        while (
            substring_positions[substring]
            and index - substring_positions[substring][0] > cluster_threshold
        ):
            substring_positions[substring].popleft()
            active_substrings[substring] -= 1
            if not active_substrings[
                substring
            ]:  # Clean up if no occurrences are within the threshold
                del active_substrings[substring]
                del substring_positions[substring]

        # Add new index and update count
        substring_positions[substring].append(index)
        active_substrings[substring] += 1

        return active_substrings[substring]

    active_substrings = defaultdict(
        int
    )  # Counts of substrings currently within the cluster threshold
    substring_positions = defaultdict(
        deque
    )  # Positions of each substring's occurrences
    max_repetitions = 0
    most_repeated_substring = ""

    for start in range(len(text) - min_length + 1):
        for end in range(
            start + min_length,
            min(start + min_length + cluster_threshold, len(text)) + 1,
        ):
            substring = text[start:end]
            count = update_counts(substring, start)

            # Check if this substring is the most repeated within the constraints
            if count >= min_occurrences and (
                count > max_repetitions
                or (
                    count == max_repetitions
                    and len(substring) > len(most_repeated_substring)
                )
            ):
                max_repetitions = count
                most_repeated_substring = substring

    print(f"The highest number of repetitions is {max_repetitions}")
    print(f"The substring that repeated the most is '{most_repeated_substring}'")

    return max_repetitions, most_repeated_substring


def validate_repetition_callback(
    min_length, num_repetitions_allowed, cluster_threshold
):  # returns a function that checks if a string has a repetition of a certain length
    def inner(string, input_data):
        count, substr = find_frequent_substrings(
            string, min_length, num_repetitions_allowed, cluster_threshold
        )
        res = count <= num_repetitions_allowed
        if not res:
            print(f"\n\nRepetition found: '{substr}' (repeated {count} times)\n\n")
        return res

    return inner


def validate_not_none(input):
    # print(input)
    if input == None:
        return False
    else:
        return True


def fix_text(
    to_replace_arr, text
):  # see common errors across your input text? Give this an array of tuples ("bad string from input text","string it should be") and this'll fix up the raw inputs. Example for double spaces only: to_replace_arr = [("  ", "")]
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text


#### BEGIN GENERATION STEPS
## Step

emotion_length_validation = validate_length_callback(600)
emotion_repetition_callback = validate_repetition_callback(16, 3, 400)


def validate_emotion_format(emotion_output, input_data):
    """
    Combines length validation, repetition validation, and format checking for emotions.

    Args:
        emotion_output (str): The emotion text to validate

    Returns:
        tuple: (bool, str) - (is_valid, reason_if_invalid)
    """
    # Check length
    if not emotion_length_validation(emotion_output, input_data):
        return False, "Emotion text exceeds maximum allowed length"

    # Check repetition
    if not emotion_repetition_callback(emotion_output, input_data):
        count, substr = find_frequent_substrings(emotion_output, 16, 3, 400)
        return False, f"Excessive repetition found: '{substr}' repeated {count} times"

    # Check format (all caps before colon)
    colon_pos = emotion_output.find(":")
    if colon_pos == -1:
        if any(c.islower() for c in emotion_output):
            return False, "Emotion format incorrect: contains lowercase characters"
    else:
        before_colon = emotion_output[:colon_pos]
        if any(c.islower() for c in before_colon):
            return (
                False,
                "Emotion format incorrect: contains lowercase characters before colon",
            )

    return True


def check_start_format(s):
    # Find the position of the first colon, if any
    colon_pos = s.find(":")

    # If there's no colon, check the whole string
    if colon_pos == -1:
        return not any(c.islower() for c in s)

    # Check only the part before the colon
    before_colon = s[:colon_pos]
    return not any(c.islower() for c in before_colon)


def emotion_output_processor(emotion_output):
    # check_1 = emotion_length_validation(emotion_output)
    # check_2 = emotion_repetition_callback(emotion_output)
    # check_3 = check_start_format(emotion_output)
    # if all([check_1, check_2, check_3]):
    return emotion_output


generate_emotion_pipeline_step = DepthFirstPipelineStep(
    prompt_path="generate_emotion_from_text",
    output_processor=emotion_output_processor,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n",
            "\n\n",
        ],
        "temperature": 0.5,
    },
    output_file="story_generation",
    result_key="emotion",
    validation_function=validate_emotion_format,
    details_key="emotion_details",
)

# generate_emotion = create_depth_first_step_function(pipeline_kwargs={
#     "prompt_path": "generate_emotion_from_text",
#     "output_processor": emotion_output_processor,
#     "sampling_params": {
#         "max_tokens": 2000,
#         "stop": [
#             "\n",
#             "\n\n",
#         ],
#         "temperature": 0.5,
#     },
#     "output_subdir": "emotion_generation",
#     "intermediate_output_path": "emotion_generation_historical_outputs", # more used for debugging?
#     "save_path": "emotion_generation_resumable_outputs", # more used for machine-reading in the previous output
#     "result_key": "emotion",
#     "validation_function": validate_emotion_format
# })

### Extract Emotion Constrained


## Helpers
def validate_emotion_key_contained(emotions):
    def inner(text):
        return any(emotion in text for emotion in emotions)

    return inner


def stringify_emotion_list(emotions):
    return "[" + ", ".join(emotions) + "]"


def create_generate_constrained_emotion_step(emotions):
    return DepthFirstPipelineStep(
        prompt_path="generate_emotion_constrained",
        validation_function=validate_emotion_key_contained(emotions=emotions),
        sampling_params={
            "max_tokens": 2000,
            "stop": [
                "\n\n\n\n\n",
            ],
            "temperature": 0.8,
        },
        output_file="story_generation",
        result_key="emotion",
        possible_emotions_list=stringify_emotion_list(emotions=emotions),
        details_key="emotion_details",
    )


### Extract Features

## Helpers


# Start of Selection
def parse_string_to_dict(string, headings):
    lines = string.strip().split("\n")
    result = {}
    current_heading = None
    current_list = []

    # First pass: Identify all headings and their approximate positions
    heading_positions = {}
    for idx, line in enumerate(lines):
        # More robust heading detection with markdown support
        clean_line = re.sub(r"^[\*\-\s]+|[\*\-\s]+$", "", line.strip())
        if clean_line.endswith(":"):
            heading = clean_line[:-1].strip()
            heading_positions[heading] = idx

    # Second pass: Process lines while handling different formats
    for idx, line in enumerate(lines):
        line = line.strip()

        # Improved heading detection with markdown stripping
        stripped_line = re.sub(r"^[\*\-\s]+", "", line)
        if any(stripped_line.startswith(h + ":") for h in headings):
            colon_pos = stripped_line.find(":")
            heading_candidate = stripped_line[:colon_pos].strip()

            if (
                heading_candidate in heading_positions
                and heading_positions[heading_candidate] == idx
            ):
                if current_heading is not None:
                    result[current_heading] = current_list
                    current_list = []
                current_heading = heading_candidate
                continue

        # Handle bullet points under current heading
        if current_heading is not None:
            # More flexible bullet point detection
            bullet_match = re.match(r"^[\*\-\s]*(.*)", line)
            if bullet_match:
                cleaned_line = bullet_match.group(1).strip()
                if cleaned_line:
                    current_list.append(cleaned_line)

    # Add the final collected items
    if current_heading is not None:
        result[current_heading] = current_list

    # Verify required headings are present (case-insensitive)
    available_headings = {h.lower(): h for h in result.keys()}
    missing = []
    for h in headings:
        if h.lower() not in available_headings:
            missing.append(h)
        else:
            # Preserve original casing from input
            result[h] = result.pop(available_headings[h.lower()])

    if missing:
        raise ValueError(f"Missing required headings: {', '.join(missing)}")

    # Ensure all requested headings exist, even if empty
    for heading in headings:
        if heading not in result:
            result[heading] = []

    return result


def dict_to_string(features):
    return "\n\n".join(
        [
            f"{key}:\n\n" + "\n".join([f"* {item}" for item in value])
            for key, value in features.items()
        ]
    )


## Helpers


def extract_text(input_text):
    # This regex looks for a line that starts with one or more all-caps words followed by a colon
    # and captures until the end of the line or paragraph.
    pattern = r"^([A-Z\s]+:\s.*?)(?=\n\n|\Z)"
    matches = re.findall(pattern, input_text, re.MULTILINE | re.DOTALL)
    if matches:
        return matches[0].strip()  # Return the first match found
    return None  # Return None if no match is found


### Extract archetype

create_archetype_step = DepthFirstPipelineStep(
    prompt_path="generate_archetype",
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    output_file="story_generation",
    result_key="archetype",  # only used by a specific prompt set that actually includes archetypes as a thing.
    output_processor=extract_text,
    details_key="archetype_details",
)
### Generate Story

## Helpers and Output Processors


def get_character_name(scene_text, user_placeholder="{user}"):
    """
    Extracts the character name from a scene text based on the rules provided.

    Parameters:
    - scene_text: The text containing the scene.
    - user_placeholder: The placeholder for the user's name in the scene text.

    Returns:
    - The name of the character that occurs most frequently, other than the user, based on the specified conditions.
    """
    from collections import Counter
    import math

    # Split the text into lines and initialize variables
    lines = scene_text.split("\n")
    character_counts = Counter()
    user_count = 0

    # Iterate over each line to find character names and count user occurrences
    for line in lines:
        if line.strip().startswith(user_placeholder + ":"):
            user_count += 1
        else:
            # Check if line starts with a character name pattern
            if ":" in line:
                character_name = line.split(":", 1)[0].strip()
                if character_name != user_placeholder:
                    character_counts[character_name] += 1

    # Determine the threshold for minimum occurrences
    threshold = math.ceil(user_count / 2)

    # Filter characters who meet the threshold requirement
    valid_characters = {
        name: count for name, count in character_counts.items() if count >= threshold
    }

    if not valid_characters:
        return None

    # Find the character with the most occurrences, resolving ties by selecting the first found
    most_common_character = max(valid_characters, key=valid_characters.get)

    return most_common_character


def parse_chatlog(chatlog, charname):
    messages = []
    current_owner = None
    current_content = []

    for line in chatlog.split("\n"):
        if line.startswith(charname + ":") or line.startswith("{user}:") or line.startswith("{narrator}:"):
            if current_owner and current_content:
                messages.append(
                    {"owner": current_owner, "content": "\n".join(current_content)}
                )
                current_content = []
            current_owner = line.split(":")[0]
            current_content.append(line.split(":")[1])
        else:
            if current_owner:
                current_content.append(line)

    if current_owner and current_content:
        messages.append({"owner": current_owner, "content": "\n".join(current_content)})

    return [
        {"owner": message["owner"], "content": message["content"].strip()}
        for message in messages
    ]


def find_duplicate_character_message(messages):
    character_messages = [
        msg["content"] for msg in messages if msg["owner"] != "{user}"
    ]
    for i, msg in enumerate(character_messages):
        if msg in character_messages[i + 1 :]:
            return messages.index(
                next(
                    m
                    for m in messages
                    if m["owner"] != "{user}" and m["content"] == msg
                )
            )
    return None


def find_message_exceeding_threshold(messages, threshold):
    for i, msg in enumerate(messages):
        if count_tokens(msg["content"]) > threshold:
            return i
    return None


def stringify_chatlog_list(
    chatlog_list,
):  # converts from chatlog list back to a chatlog. Basically the inverse function (except the decision about what perspective to write in is lost in the first conversion to a list)
    return "\n\n".join(
        [f'{message["owner"]}: {message["content"]}' for message in chatlog_list]
    )


def ends_with_fullstop(text):
    """check if a function ends with  either . ? ! or any of these followed by a " character"""
    # print("!!!!PROBLEM BELOW \/\/\/\/\/")
    # print(text.strip())
    # print("!!!!PROBLEM ABOVE ^^^^^^^^^")
    return text.strip().endswith((".", "?", "!", '."', '?"', '!"', ".*", "!*", "?*"))


def split_last_message_at_note(
    chatlog_list,
):  # helps if you feed degenerate shit into the AI and it writes something but is like "Note: this is actually really bad!"
    last_message = chatlog_list[-1]
    last_message_content = last_message["content"]
    note_index = last_message_content.find("Note")
    if note_index != -1:
        return chatlog_list[:-1] + [
            {
                "owner": last_message["owner"],
                "content": last_message_content[:note_index].strip(),
            }
        ]
    return chatlog_list


def parse_story_messages(large_mode):
    def inner(story):
        charname = get_character_name(story)
        if not charname:
            print("ERROR: Character name not found in story, format horribly broken")
            raise ValueError("Character name not found in story")
        chatlog_list = parse_chatlog(story, charname)

        # if not starts_with_charname(chatlog_list[0]["content"], charname):
        #     print("ERROR: Story does not start with the character name")
        #     raise ValueError("Story does not start with the character name")

        truncated = False

        chatlog_list = split_last_message_at_note(chatlog_list)

        # print(chatlog_list)
        threshold_message_index = find_message_exceeding_threshold(chatlog_list, 650)
        if threshold_message_index:
            print("\n\TOO LONG MESSAGES DETECTED -- TRUNCATING STORY")
            chatlog_list = chatlog_list[:threshold_message_index]
            truncated = True

        duplicate_message_index = find_duplicate_character_message(chatlog_list)
        if duplicate_message_index:
            print("\n\nDUPLICATE MESSAGES DETECTED -- TRUNCATING STORY")
            chatlog_list = chatlog_list[:duplicate_message_index] + [
                chatlog_list[duplicate_message_index]
            ]  # take the first instance of the duplicated message, it's probably alright
            truncated = True

        if (
            count_tokens(stringify_chatlog_list(chatlog_list)) > 3500
            and large_mode == "cohere"
            and not truncated
        ):
            print("\n\nSTORY VERY LONG -- DROPPING LAST MESSAGE AS SAFEGUARD")
            chatlog_list = chatlog_list[:-1]
            truncated = True

        # truncate last message if it doesn't end with a full stop
        if not ends_with_fullstop(chatlog_list[-1]["content"]):
            print("\n\nLAST MESSAGE DOES NOT END WITH FULL STOP -- TRUNCATING STORY")
            chatlog_list = chatlog_list[:-1]
            truncated = True

        processed_story_string = stringify_chatlog_list(chatlog_list)

        for line in processed_story_string.split("\n"):
            if line.startswith("{narrator}:"):
                processed_story_string = processed_story_string.replace(
                    "{narrator}:", charname + ":", 1
                )

        print("==================================")
        return processed_story_string

    return inner


## Step


def validate_narrator_replacement(story: str, charname: str) -> bool:
    """
    Checks if narrator replacement logic will succeed.
    Returns True if safe to proceed, False if likely to fail.
    """
    # Check for None inputs
    if story is None or charname is None:
        return False

    # Check type safety
    if not isinstance(story, str) or not isinstance(charname, str):
        return False

    # Check for empty character name
    if len(charname.strip()) == 0:
        return False

    # Check for colon in character name (would break replacement logic)
    if ":" in charname:
        return False

    # Check if any narrator lines exist
    has_narrator_lines = any(
        line.startswith("{narrator}:") for line in story.split("\n")
    )

    # If narrator lines exist but character name is invalid, return False
    if has_narrator_lines and len(charname) == 0:
        return False

    return True


def create_generate_story_pipeline_step(include_chunk_in_prompt, use_min_p, large_mode):
    prompt_path = (
        "generate_story" if not include_chunk_in_prompt else "generate_story_with_chunk"
    )
    if use_min_p:
        sampling_params = {
            "max_tokens": 7000 if large_mode != "cohere" else 4000,
            "temperature": 2,
            "top_p": 0.8,
            "min_p": 0.2,
            "stop": ["###", "### TASK: ###", "ROLEPLAY SESSION END"],
        }
    else:
        sampling_params = {
            "max_tokens": 7000 if large_mode != "cohere" else 4000,
            "temperature": 1.5,
            "top_p": 0.7,
            "stop": ["###", "### TASK: ###", "ROLEPLAY SESSION END"],
        }

    return DepthFirstPipelineStep(
        prompt_path=prompt_path,
        output_processor=parse_story_messages(large_mode),
        sampling_params=sampling_params,
        output_file="story_generation",
        result_key="story",
        regex=re.compile(r"### NOW! THE STORY BEGINS ###(.*?)###", re.DOTALL),
        details_key="story_details",
    )


### End Generate Story

### Rate Story

## Helpers


def validate_rating_keys_presence(data):
    # print(data)
    # Define the required keys
    required_keys = ["coherence", "rule following", "quality"]

    # Check if all required keys are in the dictionary
    are_keys_present = all(key in data for key in required_keys)
    print(f"\n\nARE KEYS PRESENT? THIS CODE SAYS: {are_keys_present}")

    return are_keys_present


def extract_ratings(input_text):
    # Compile a regular expression to match category names followed by any text and then "RATING:" and the rating value
    pattern = re.compile(r"(\w+):\n(?:.|\n)+?RATING:\s*(\w+)", re.MULTILINE)

    # Find all matches of the pattern in the input text
    matches = pattern.findall(input_text)

    # Construct a dictionary from the matches
    ratings_dict = {
        category.strip().lower(): rating.strip().lower() for category, rating in matches
    }

    if not validate_rating_keys_presence(ratings_dict):
        raise ValueError("Not all required keys are present in the input text")

    return ratings_dict


# RPToolkit rating thing -- the AI must never speak for the user
# add that


# here's what I Want to do, right?
# we want the model to always output things in the right format.
# That being said, the model has often outputted things in its own format.
# I want to be able to extract teh structure from the lack of structure.
# what can we guarantee? The headings. And that the content will follow the heading. Until the next heading
# A heading may be any casing,  surrounded by any marks. But we know what headings to look for.

# I want to build a generalized component for data extraction that will 1. work to make this step not error regardless of opinionated AI, 2. will make extracting data and formatting it into different output formats in the data easy (because we'll want to extract and clean stuff in there, for sure); 3. it would be invaluable in other pipelines as well


def parse_story_ratings(story_ratings):
    try:
        ratings_obj = extract_structured_data(
            story_ratings,
            headings={
                "COHERENCE": {"name": "rating", "prefix": "RATING:"},
                "RULE FOLLOWING": {"name": "rating", "prefix": "RATING:"},
                "QUALITY": {"name": "rating", "prefix": "RATING:"},
            },
        )

        # Check for missing ratings
        missing_ratings = [
            key
            for key in ["coherence", "rule following", "quality"]
            if ratings_obj.get(key) is None
        ]
        if missing_ratings:
            raise ValueError(
                f"Missing required ratings in evaluation: {', '.join(missing_ratings)}"
            )

        # clean any * asterisks from the extracted rating. Validate that the values are either good awful or incredible.
        for key, value in ratings_obj.items():
            # Clean asterisks and strip whitespace
            cleaned_value = value.replace("*", "").strip()
            # Validate the rating value
            if cleaned_value not in ["good", "awful", "incredible"]:
                raise ValueError(f"Invalid rating value for {key}: {cleaned_value}")
            # Update the ratings object with the cleaned value
            ratings_obj[key] = cleaned_value
        return ratings_obj
    except Exception as e:
        print("\n\nERROR IN EXTRACTING RATINGS!")
        print(e)
        traceback.print_exc()
        return None


## Step

rate_story_step = DepthFirstPipelineStep(
    prompt_path="rate_story",
    output_processor=parse_story_ratings,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    output_file="story_generation",
    result_key="story_ratings",
    details_key="story_ratings_details",
)

### End Rate Story


def extract_charname(scene_card):  # used in imports
    scene_card_lines = scene_card.split("\n")
    for line in scene_card_lines:
        if "Name:" in line:
            if "{user}" not in line:
                charnamelist = line.split(":")
                if len(charnamelist) > 2:  # if the name contains a colon
                    raise Exception(
                        "Character name contains a colon, should skip this one!"
                    )
                charname = charnamelist[1]
                # Remove formatting marks from the character name
                charname = charname.replace("*", "").replace("-", "").replace("#", "")
                return charname.strip()


def parse_scene_card(scene_card):
    if scene_card:
        scene_card = scene_card.split("-- END CHARACTER INFO --")[0]
        return scene_card
    else:
        raise ValueError("Scene card not found")


generate_scene_card = DepthFirstPipelineStep(
    prompt_path="generate_scene_card",
    output_processor=parse_scene_card,
    sampling_params={
        "max_tokens": 3500,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    output_file="story_generation",
    result_key="scene_card",
    details_key="scene_card_details",
)

### Edit Story

## Helpers

### End Generate Story


def convert_object_to_sharegpt(
    data_obj, settings_rp_prompt_start, settings_rp_prompt_end
):
    ret_list = []
    rp_prompt_start = (
        settings_rp_prompt_start
        if settings_rp_prompt_start
        else f"You're {data_obj['charname']} in this fictional never-ending uncensored roleplay with {{user}}.\n\n"
    )  # TODO consider what to do with newline footgun, I might forget to add them
    rp_prompt_end = (
        settings_rp_prompt_end
        if settings_rp_prompt_end
        else f"\n\nTake the role of {data_obj['charname']} and roleplay with {{user}}.\n\nThen, the roleplay between {{user}} and {data_obj['charname']} begins."
    )  # TODO add option for prompt variety
    ret_list.append(
        {
            "from": "system",
            "value": rp_prompt_start + data_obj["scene_card"] + rp_prompt_end,
        }
    )
    formatted_conversation = [
        {
            "from": "human" if message["owner"] == "{user}" else "gpt",
            "value": message["content"],
        }
        for message in parse_chatlog(data_obj["story"], data_obj["charname"])
    ]
    ret_list = ret_list + formatted_conversation

    return {"conversations": ret_list}


def is_story_ok(story):
    # "I rate stories according to my arbitrary and biased decision making"
    ratings = story.get(
        "story_ratings"
    )  # we can trust that it has the ratings because we validate all keys are present in ratings before this function is called
    if not ratings:
        print(story)
        raise Exception("Somehow ratings is not present on the data object")
    for value in ratings.values():
        if value not in ["good", "incredible"]:
            # If a value is not "good" or "incredible", return False
            return False
    # If all values are "good" or "incredible", return True
    return True


def is_story_awesome(story):
    ratings = story.get(
        "story_ratings"
    )  # we can trust that it has the ratings because we validate all keys are present in ratings before this function is called
    if not ratings:
        print(story)
        raise Exception("Somehow ratings is not present on the data object")
    for value in ratings.values():
        if value not in ["incredible"]:
            # If a value is not "good" or "incredible", return False
            return False
    # If all values are "good" or "incredible", return True
    return True


def write_final_dataset_files(
    story_data, name, output_folder, settings_rp_prompt_start, settings_rp_prompt_end
):
    with open(
        f"{output_folder}/final_outputs/{name}_complete_format.json",
        "w",
        encoding="utf-8",
    ) as file1:  # complete_format includes
        json.dump(story_data, file1, indent=4)
    sharegpt_data = [
        convert_object_to_sharegpt(
            story, settings_rp_prompt_start, settings_rp_prompt_end
        )
        for story in story_data
        if story["charname"]
    ]
    with open(
        f"{output_folder}/final_outputs/{name}_sharegpt.json", "w", encoding="utf-8"
    ) as file2:
        json.dump(sharegpt_data, file2, indent=4)


## some helpers
import re
import random

import time
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial