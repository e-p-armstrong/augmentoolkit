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
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
import logging
from math import ceil
import traceback
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
import uuid
import yaml

tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
    )

def count_tokens(message):
    return len(tokenizer.encode(message))

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT_FOLDER = os.path.abspath(obj_conf["PATH"]["OUTPUT"])
DEFAULT_PROMPT_PATH = os.path.abspath(obj_conf["PATH"]["DEFAULT_PROMPTS"])
PROMPTS = os.path.abspath(obj_conf["PATH"]["PROMPTS"])
COMPLETION_MODE = obj_conf["SYSTEM"]["COMPLETION_MODE"]
LOGGING_LEVEL = logging.INFO
LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
API_KEY_A = obj_conf["API"]["API_KEY_A"]
API_KEY_B = obj_conf["API"]["API_KEY_B"]
BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
MODE_A = obj_conf["SYSTEM"]["MODE_A"]
MODE_B = obj_conf["SYSTEM"]["MODE_B"]
CONCURRENCY_LIMIT = obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"]
USE_STOP = obj_conf["SYSTEM"]["STOP"]

## Chunking Logic for Raw Input Text ##
def chunking_algorithm(file_path, max_token_length=1500):
    """
    This function takes a plaintext file and chunks it into paragraphs or sentences if the paragraph exceeds max_token_length.

    :param file_path: Path to the plaintext file
    :param tokenizer: SentencePiece tokenizer
    :param max_token_length: The maximum token length for a chunk
    :return: List of chunks with source text information
    """
    chunks_with_source = []
    current_chunk = []
    token_count = 0
    source_name = file_path.replace(".txt", "")


    with open(file_path, "r", encoding="utf-8",errors='ignore') as f:
        content = f.read()
        
    paragraphs = content.split('\n\n')  # Assuming paragraphs are separated by two newlines # TODO change so that if the length is 1 after this, split by tabs instead

    for paragraph in paragraphs:
        paragraph = paragraph.strip()  # Remove leading and trailing whitespace
        if not paragraph:  # Skip empty paragraphs
            continue
        
        paragraph_token_count = count_tokens(paragraph)
        
        # Check if the paragraph itself exceeds the max token length
        if paragraph_token_count > max_token_length:
            # Fallback to sentence chunking for this paragraph
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)
                if token_count + sentence_token_count <= max_token_length:
                    current_chunk.append(sentence)
                    token_count += sentence_token_count
                else:
                    chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_source.append({"chunk": " ".join(current_chunk), "source": source_name})

    return chunks_with_source

## Post-generation validation and retry abstraction
async def validate_generation(gen_func, validation_functions, retries=1, gen_func_args=[]): # higher-order function that takes a list of validation functions and a generation function, and checks if the generation function's output passes all the validation functions; if not, it retries up to a certain number of times. If it fails it returns None, otherwise it returns the output of the generation function.
    """
    The interface for validation functions compatible with validate_generation is as follows:
    they should take a single input: the output of the generation function
    they should return false if that input does not pass validation and True if it does
    The gen_func_args should ALWAYS have the id be the last argument
    """
    times_tried = 0
    while times_tried <= retries:
        try:
            response = await gen_func(*gen_func_args[:-1], str(gen_func_args[-1]) + f"_{times_tried}")
            success = True
            for validation_function in validation_functions:
                if not validation_function(response):
                    print("VALIDATION FAILED")
                    success = False
                    break
            if success:
                return response
            else:
                times_tried += 1
        except Exception as e:
            times_tried += 1
            print(f"Error in Generation Step: {e}")
            print(e)
            traceback.print_exc()
    raise Exception("VALIDATION FAILED TOO MANY TIMES -- CUTTING LOSSES AND SKIPPING THIS CHUNK\n\n\n")

# Helpers for said abstraction
def validate_length_callback(length): # returns a function that checks if a string is a certain length
    def inner(string):
        return count_tokens(string) <= length
    return inner


def find_repetitions(text, min_length):
    pattern = r'(.{' + str(min_length) + r',}?)\1+'
    repetitions = re.finditer(pattern, text) 

    return repetitions

def validate_consecutive_repetition_callback(min_length): # returns a function that checks if a string has a repetition of a certain length
    def inner(string):
        return len(list(find_repetitions(string, min_length))) == 0
    return inner


def find_frequent_substrings(text, min_length, min_occurrences, cluster_threshold):
    def update_counts(substring, index):
        # Update positions and remove indices that are out of the current cluster scope
        while substring_positions[substring] and index - substring_positions[substring][0] > cluster_threshold:
            substring_positions[substring].popleft()
            active_substrings[substring] -= 1
            if not active_substrings[substring]:  # Clean up if no occurrences are within the threshold
                del active_substrings[substring]
                del substring_positions[substring]

        # Add new index and update count
        substring_positions[substring].append(index)
        active_substrings[substring] += 1

        return active_substrings[substring]

    active_substrings = defaultdict(int)  # Counts of substrings currently within the cluster threshold
    substring_positions = defaultdict(deque)  # Positions of each substring's occurrences
    max_repetitions = 0
    most_repeated_substring = ""

    for start in range(len(text) - min_length + 1):
        for end in range(start + min_length, min(start + min_length + cluster_threshold, len(text)) + 1):
            substring = text[start:end]
            count = update_counts(substring, start)

            # Check if this substring is the most repeated within the constraints
            if count >= min_occurrences and (count > max_repetitions or (count == max_repetitions and len(substring) > len(most_repeated_substring))):
                max_repetitions = count
                most_repeated_substring = substring

    print(f"The highest number of repetitions is {max_repetitions}")
    print(f"The substring that repeated the most is '{most_repeated_substring}'")

    return max_repetitions, most_repeated_substring

def validate_repetition_callback(min_length, num_repetitions_allowed, cluster_threshold): # returns a function that checks if a string has a repetition of a certain length
    def inner(string):
        count, substr = find_frequent_substrings(string, min_length, num_repetitions_allowed, cluster_threshold)
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
    
def fix_text(to_replace_arr, text): # see common errors across your input text? Give this an array of tuples ("bad string from input text","string it should be") and this'll fix up the raw inputs. Example for double spaces only: to_replace_arr = [("  ", "")]
    for startup in to_replace_arr:
        text = text.replace(startup[0], startup[1])
    return text
    
    
class DepthFirstPipelineStep(PipelineStep): # RPTOOLKIT is depth-first rather than breadth-first, so it is easiest to build these steps out using a different class of pipeline step that is focused on returning rather than appending to an output list
    def read_previous_output(self, idx):
        save_path_file = super().make_save_path_file(idx)
        if os.path.exists(save_path_file):
            with open(save_path_file, "r") as f:
                output_data = json.load(f)
            return output_data
        return False
    
    def save(self, result=None,
    full_output=None,
    idx=None,
    input_data=None,):
        
        if result:
            id = make_id()
            save_path_file = super().make_save_path_file(idx)
            
            output_data = input_data
            output_data[self.result_key] = result
            write_output_to_file(full_output, self.intermediate_output_path_full, id)
            
            os.makedirs(self.save_path, exist_ok=True)
            with open(save_path_file, "w") as f:
                f.write(json.dumps(output_data))
            
            return output_data
    
    async def run(self, idx=None,
    input_data=None,
    engine_wrapper=None,
      ): # things that are args here are produced during inference time. Including config settings.
        
        read_previous_item = self.read_previous_output(idx)
        if read_previous_item:
            return read_previous_item
        
        processed_data = super().process_input_data(input_data)
        
        complete = False
        max_retries = self.max_retries
        while not complete and max_retries > 0:
            try:
                result, full_output = await super().generate_data(processed_data, engine_wrapper)
                if self.validation_function(result, input_data):
                    complete = True
            except Exception as e:
                print(e)
                traceback.print_exc() 
                max_retries -= 1
        if not complete: # consider raising here and catching in the actual pipeline.
            return
        
        return self.save(result=result, full_output=full_output, idx=idx, input_data=input_data)
#### BEGIN GENERATION STEPS
    
    
### Generate Emotion

## Helpers

## Step

emotion_generator_path = "generate_emotion_from_text"

emotion_length_validation = validate_length_callback(600)
emotion_repetition_callback = validate_repetition_callback(16, 3, 400)
def check_start_format(s):
    # Find the position of the first colon, if any
    colon_pos = s.find(':')
    
    # If there's no colon, check the whole string
    if colon_pos == -1:
        return not any(c.islower() for c in s)
    
    # Check only the part before the colon
    before_colon = s[:colon_pos]
    return not any(c.islower() for c in before_colon)

def emotion_output_processor(emotion_output):
    check_1 = emotion_length_validation(emotion_output)
    check_2 = emotion_repetition_callback(emotion_output)
    check_3 = check_start_format(emotion_output)
    if all([check_1, check_2, check_3]):
        return emotion_output
    print("Failed checks:")
    print(check_1,check_2, check_3)
    raise ValueError("Emotion failed validations")
        
emotion_generator = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path=emotion_generator_path,
    output_processor=emotion_output_processor,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n",
            "\n\n",
        ],
        "temperature": 0.5,
    },
    completion_mode=COMPLETION_MODE, 
    output_dir=OUTPUT_FOLDER,
    output_subdir="emotion_generation",
    intermediate_output_path="emotion_generation_historical_outputs", # more used for debugging?
    save_path="emotion_generation_resumable_outputs", # more used for machine-reading in the previous output
    result_key="emotion",
    use_stop=USE_STOP,
)
        

async def generate_emotion_from_text(chunk, engine_wrapper, idx): # TODO make the two newline stop token into something less hacky. Prompt adjustment.
    
    return await emotion_generator.run(idx=idx, input_data=chunk, engine_wrapper=engine_wrapper)
    
### End Generate Emotion

### Extract Emotion Constrained

## Helpers
def validate_emotion_key_contained(text):
    return any(emotion in text for emotion in obj_conf["SYSTEM"]["EMOTIONS"])

def confirm_text_emotions(text):
    if validate_emotion_key_contained(text):
        return text
    raise ValueError("Emotion not in list")


# async def generate_emotion_constrained(chunk, engine_wrapper, id):
#     result = await validate_generation(generate_emotion_constrained_inner, [validate_not_none, validate_emotion_key_contained], 2, [chunk, engine_wrapper, id])
#     return result

def stringify_emotion_list():
    # Create a string that's the list of keys in the obj_conf["SYSTEM"]["EMOTIONS"] list
    return "[" + ", ".join(obj_conf["SYSTEM"]["EMOTIONS"]) + "]"

constrained_emotion_generator = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path="generate_emotion_constrained",
    output_processor=confirm_text_emotions,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    completion_mode=COMPLETION_MODE,
    output_dir=OUTPUT_FOLDER,
    output_subdir="emotion_generation",
    intermediate_output_path="emotion_generation_historical_outputs", # more used for debugging?
    save_path="emotion_generation_resumable_outputs", # more used for machine-reading in the previous output
    result_key="emotion",
    use_stop=USE_STOP,
    possible_emotions_list=stringify_emotion_list(),
)

async def generate_emotion_constrained(chunk, engine_wrapper, idx):
    return await constrained_emotion_generator.run(idx=idx, input_data=chunk, engine_wrapper=engine_wrapper)

### Extract Features

## Helpers
def parse_string_to_dict(string, headings): # I believe this works
    lines = string.strip().split('\n')
    result = {}
    current_heading = None
    current_list = []

    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            if current_heading is not None:  # Checks if there's a heading already being processed.
                result[current_heading] = current_list  # Saves the list of items under the current heading.
                current_list = []  # Resets the list for the next set of items.
            current_heading = line[:-1]  # Sets the new heading, removing the colon.
        elif line.startswith('*') or line.startswith('-'):
            current_list.append(line[1:].strip())  # Adds items that start with '*' or '-' to the current list.
        else:
            continue  # Skips lines that do not start with a bullet.

    if current_heading is not None:  # Checks and saves any remaining items under the last heading.
        result[current_heading] = current_list

    # Checks for any headings that were specified but not found in the string.
    missing_headings = set(headings) - set(result.keys())
    if missing_headings:
        raise ValueError(f"Missing headings: {', '.join(missing_headings)}")

    return result

def dict_to_string(features):
    return "\n\n".join([f"{key}:\n\n" + "\n".join([f"* {item}" for item in value]) for key, value in features.items()])

def parse_features(features):
    try:
        to_include = ["Initiating Event", "Character Traits", "Feelings", "Physical Traits", "Physical Props", "Settings", "Genre Tags"]
        features_obj = parse_string_to_dict(features, to_include)
        # remove keys that are not in the given to-include list
        features_obj = {key: value for key, value in features_obj.items() if key in to_include}
        
        # print("\n\nFEATURES OBJECT POSTPROCESSING DEBUGGING:")
        # print(features_obj)
        features_obj["Genre Tags"] = features_obj["Genre Tags"][:10] # postprocessing to fix run-on generations
        # print(features_obj)
        return dict_to_string(features_obj)
    except Exception as e:
        print("\n\nERROR IN EXTRACTING FEATURES!")
        print(e)
        traceback.print_exc()
        return None

## Step

feature_extractor = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path="extract_features",
    output_processor=parse_features,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    completion_mode=COMPLETION_MODE,
    output_dir=OUTPUT_FOLDER,
    output_subdir="feature_extraction",
    intermediate_output_path="feature_extraction_historical_outputs", # more used for debugging?
    save_path="feature_extraction_resumable_outputs", # more used for machine-reading in the previous output
    result_key="features",
    use_stop=USE_STOP,
)


async def extract_features(chunk, engine_wrapper, idx):
    return await feature_extractor.run(idx=idx, input_data=chunk, engine_wrapper=engine_wrapper)

## Helpers

def extract_text(input_text):
    # This regex looks for a line that starts with one or more all-caps words followed by a colon
    # and captures until the end of the line or paragraph.
    pattern = r'^([A-Z\s]+:\s.*?)(?=\n\n|\Z)'
    matches = re.findall(pattern, input_text, re.MULTILINE | re.DOTALL)
    if matches:
        return matches[0].strip()  # Return the first match found
    return None  # Return None if no match is found

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
    lines = scene_text.split('\n')
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
    valid_characters = {name: count for name, count in character_counts.items() if count >= threshold}

    if not valid_characters:
        return None

    # Find the character with the most occurrences, resolving ties by selecting the first found
    most_common_character = max(valid_characters, key=valid_characters.get)

    return most_common_character

def parse_chatlog(chatlog,charname):
    messages = []
    current_owner = None
    current_content = []

    for line in chatlog.split("\n"):
        if line.startswith(charname + ":") or line.startswith("{user}:"):
            if current_owner and current_content:
                messages.append({"owner": current_owner, "content": "\n".join(current_content)})
                current_content = []
            current_owner = line.split(":")[0]
            current_content.append(line.split(":")[1])
        else:
            if current_owner:
                current_content.append(line)

    if current_owner and current_content:
        messages.append({"owner": current_owner, "content": "\n".join(current_content)})

    return [{ "owner": message["owner"], "content": message["content"].strip()} for message in messages]

def find_duplicate_character_message(messages):
    character_messages = [msg["content"] for msg in messages if msg["owner"] != "{user}"]
    for i, msg in enumerate(character_messages):
        if msg in character_messages[i+1:]:
            return messages.index(next(m for m in messages if m["owner"] != "{user}" and m["content"] == msg))
    return None

def find_message_exceeding_threshold(messages, threshold):
    for i, msg in enumerate(messages):
        if count_tokens(msg["content"]) > threshold:
            return i
    return None

def stringify_chatlog_list(chatlog_list): # converts from chatlog list back to a chatlog. Basically the inverse function (except the decision about what perspective to write in is lost in the first conversion to a list)
    return "\n\n".join([f'{message["owner"]}: {message["content"]}' for message in chatlog_list])

def parse_story_messages(story):
    charname = get_character_name(story)
    if not charname:
        print("ERROR: Character name not found in story, format horribly broken")
        return None
    chatlog_list = parse_chatlog(story,charname)
    truncated = False
    # print(chatlog_list)
    threshold_message_index = find_message_exceeding_threshold(chatlog_list, 650)
    if threshold_message_index:
        print("\n\TOO LONG MESSAGES DETECTED -- TRUNCATING STORY")
        chatlog_list = chatlog_list[:threshold_message_index]
        truncated = True
        
    duplicate_message_index = find_duplicate_character_message(chatlog_list)
    if duplicate_message_index:
        print("\n\nDUPLICATE MESSAGES DETECTED -- TRUNCATING STORY")
        chatlog_list = chatlog_list[:duplicate_message_index] + [chatlog_list[duplicate_message_index]] # take the first instance of the duplicated message, it's probably alright
        truncated = True
        
    if count_tokens(stringify_chatlog_list(chatlog_list)) > 3500 and obj_conf["SYSTEM"]["MODE_B"] == "cohere" and not truncated:
        print("\n\nSTORY VERY LONG -- DROPPING LAST MESSAGE AS SAFEGUARD")
        chatlog_list = chatlog_list[:-1]
        truncated = True
        
    processed_story_string = stringify_chatlog_list(chatlog_list)
    
    print("==================================")
    return (processed_story_string, truncated)

## Step

story_generator = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path="generate_story",
    output_processor=parse_story_messages,
    sampling_params={
        "max_tokens": 7000 if obj_conf["SYSTEM"]["MODE_B"] != "cohere" else 4000,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    completion_mode=COMPLETION_MODE,
    output_dir=OUTPUT_FOLDER,
    output_subdir="story_generation",
    intermediate_output_path="story_generation_historical_outputs", # more used for debugging?
    save_path="story_generation_resumable_outputs", # more used for machine-reading in the previous output
    result_key="story",
    use_stop=USE_STOP,
)


async def generate_story(input_data=None,engine_wrapper=None, charname=None, idx=None):

    
    out = await story_generator.run(idx=idx, input_data=input_data, engine_wrapper=engine_wrapper)
    truncated = out["story"][1]
    out["story"] = out["story"][0]
    print("!!!!OUTS!!!!")
    print(out)
    print("----------------")
    story = out["story"]
    
    print("-----!!!!! POST POSTPROCESSING OUTPUT !!!!!---------") # NOTE that if you want to postprocess you can just put it after the pipeline step is called
    try:
        for line in story.split("\n"):
            if line.startswith("{narrator}:"):
                story = story.replace("{narrator}:", charname + ":", 1)
        out["story"] = story
        return out, truncated
    except Exception as e:
        print(f"\n\n\nLooks like unpacking stuff failed")
        print(e)
        raise Exception("\n\nUnpacking failed!!")
    print("----------------------------------------------------")

### End Generate Story

### Rate Story

## Helpers

def validate_rating_keys_presence(data):
    # print(data)
    # Define the required keys
    required_keys = ["coherence", "following", "quality", "sense"]
    
    # Check if all required keys are in the dictionary
    are_keys_present = all(key in data for key in required_keys)
    print(f"\n\nARE KEYS PRESENT? THIS CODE SAYS: {are_keys_present}")
    
    return are_keys_present

def extract_ratings(input_text):
    # Compile a regular expression to match category names followed by any text and then "RATING:" and the rating value
    pattern = re.compile(r'(\w+):\n(?:.|\n)+?RATING:\s*(\w+)', re.MULTILINE)
    
    # Find all matches of the pattern in the input text
    matches = pattern.findall(input_text)
    
    # Construct a dictionary from the matches
    ratings_dict = {category.strip().lower(): rating.strip().lower() for category, rating in matches}
    
    if not validate_rating_keys_presence(ratings_dict):
        raise ValueError("Not all required keys are present in the input text")
    
    return ratings_dict

def parse_story_ratings(story_ratings):
    try:
        ratings_obj = extract_ratings(story_ratings)
        return ratings_obj
    except Exception as e:
        print("\n\nERROR IN EXTRACTING RATINGS!")
        print(e)
        traceback.print_exc()
        return None

## Step

story_rater = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path="rate_story",
    output_processor=parse_story_ratings,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    completion_mode=COMPLETION_MODE,
    output_dir=OUTPUT_FOLDER,
    output_subdir="story_rating",
    intermediate_output_path="story_rating_historical_outputs", # more used for debugging?
    save_path="story_rating_resumable_outputs", # more used for machine-reading in the previous output
    result_key="story_ratings",
    use_stop=USE_STOP,
)

async def rate_story(story, engine_wrapper, id):
    return await story_rater.run(idx=id, input_data=story, engine_wrapper=engine_wrapper)

### End Rate Story
    
def extract_charname(scene_card):
    scene_card_lines = scene_card.split("\n")
    for line in scene_card_lines:
        if line.startswith("Name:"):
            if "{user}" not in line:
                charnamelist = line.split(":")
                if len(charnamelist) > 2: # if the name contains a colon
                    raise Exception("Character name contains a colon, should skip this one!")
                charname = charnamelist[1]
                return charname.strip()
    
def parse_scene_card(scene_card):
    if scene_card:
        scene_card = scene_card.split("-- END CHARACTER INFO --")[0]
        return scene_card
    else:
        raise ValueError("Scene card not found")
    
    

scene_card_generator = DepthFirstPipelineStep(
    prompt_folder=PROMPTS,
    default_prompt_folder=DEFAULT_PROMPT_PATH,
    prompt_path="generate_scene_card",
    output_processor=parse_scene_card,
    sampling_params={
        "max_tokens": 2000,
        "stop": [
            "\n\n\n\n\n",
        ],
        "temperature": 0.8,
    },
    completion_mode=COMPLETION_MODE,
    output_dir=OUTPUT_FOLDER,
    output_subdir="scene_card_generation",
    intermediate_output_path="scene_card_generation_historical_outputs", # more used for debugging?
    save_path="scene_card_generation_resumable_outputs", # more used for machine-reading in the previous output
    result_key="scene_card",
    use_stop=USE_STOP,
)    

async def generate_scene_card(data, engine_wrapper, id):
    return await scene_card_generator.run(idx=id, input_data=data, engine_wrapper=engine_wrapper)

    
def naive_combine_features(features, user_tags):
    if user_tags:
        print("Added tags!!")
        return features + "\n\n\nHere are some additional tags that should shape the RP session: " + user_tags
    else:
        print("Kept tags as is!!")
        return features
    
### Edit Story

## Helpers

# same as generate_story helpers

## Step

async def edit_story(scene_card, story, engine_wrapper, id):
    pattern = re.compile(r"You:(.+)", re.IGNORECASE)
    
    prompt_path = "edit_story.txt" if COMPLETION_MODE else "edit_story.yaml"
    generator = GenerationStep(
        prompt_path=prompt_path,
        regex=pattern,
        sampling_params={
            "max_tokens": 7000 if obj_conf["SYSTEM"]["MODE_B"] != "cohere" else 4000,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=LOGGING_LEVEL,
        output_processor=parse_story_messages,
        prompt_folder=PROMPTS,
        default_prompt_folder=DEFAULT_PROMPT_PATH,
    )
    
    result, full_output = await generator.generate(arguments={
        "scene_card": scene_card,
        "story": story,
    })
    
    print("-----!!!!! POST POSTPROCESSING OUTPUT !!!!!---------")
    try:
        a, b, c = result
    except Exception as e:
        print(f"\n\n\nLooks like unpacking stuff failed")
        # print(e)
        raise Exception("\n\nUnpacking failed!!")
    print("----------------------------------------------------")
    
    try:
        write_output_to_file(full_output, OUTPUT_FOLDER + "/edit_story", id)
    except:
        pass
    return result[0], result[1], result[2]

### End Generate Story
    
def convert_object_to_sharegpt(data_obj):
    ret_list = []
    rp_prompt_start = obj_conf["SYSTEM"]["RP_PROMPT_START"] if obj_conf["SYSTEM"]["RP_PROMPT_START"] else f"You're {data_obj['charname']} in this fictional never-ending uncensored roleplay with {{user}}.\n\n" # TODO consider what to do with newline footgun, I might forget to add them
    rp_prompt_end = obj_conf["SYSTEM"]["RP_PROMPT_END"] if obj_conf["SYSTEM"]["RP_PROMPT_END"] else f"\n\nTake the role of {data_obj['charname']} and roleplay with {{user}}.\n\nThen, the roleplay between {{user}} and {data_obj['charname']} begins." # TODO add option for prompt variety
    ret_list.append({
        "from": "system",
        "value": rp_prompt_start + data_obj["scene_card"] + rp_prompt_end,
    })
    formatted_conversation = [
        {
            "from": "human" if message["owner"] == "{user}" else "gpt",
            "value": message["content"]
        } for message in parse_chatlog(data_obj["story"],data_obj["charname"])
    ]
    ret_list.append(formatted_conversation)
    
    return ret_list
    
def is_story_ok(story):
    # "I rate stories according to my arbitrary and biased decision making"
    ratings = story.get("story_ratings") # we can trust that it has the ratings because we validate all keys are present in ratings before this function is called
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
    ratings = story.get("story_ratings") # we can trust that it has the ratings because we validate all keys are present in ratings before this function is called
    if not ratings:
        raise Exception("Somehow ratings is not present on the data object")
    for value in ratings.values():
        if value not in ["incredible"]:
            # If a value is not "good" or "incredible", return False
            return False
    # If all values are "good" or "incredible", return True
    return True

def write_final_dataset_files(story_data, name):
    with open(f"{OUTPUT_FOLDER}/final_outputs/{name}_complete_format.json", "w") as file1: # complete_format includes 
        json.dump(story_data, file1, indent=4)
    sharegpt_data = [convert_object_to_sharegpt(story) for story in story_data]
    with open(f"{OUTPUT_FOLDER}/final_outputs/{name}_sharegpt.json", "w") as file2:
        json.dump(sharegpt_data, file2, indent=4)
    

## some helpers

import re

# from .character_card_grammar import character_card_grammar
import string
import random


def extract_author_name(title):
    pattern = re.compile(r"\b(?:by|By)\s+([^,]+),")
    match = re.search(pattern, title)
    if match:
        author_name = match.group(1)
    else:
        author_name = [False]
    return author_name[0]  # first letter of Author name


def select_random_capital(exclusions):
    # Create a list of capital letters excluding the ones in the exclusions list
    capitals = [letter for letter in string.ascii_uppercase if letter not in exclusions]

    # Select a random capital letter from the filtered list
    if capitals:
        return random.choice(capitals)
    else:
        return "No available capital letters to choose from"


def extract_capital_letters(input_string):
    capital_letters = []
    for char in input_string:
        if char.isupper():
            capital_letters.append(char)
    return capital_letters



def extract_name(str):
    # Regular expression to match 'Name:' followed by any characters until the end of the line
    name_regex = r"^Name:\s*([^\s]*)"

    # Searching in the multiline string
    match = re.search(name_regex, str, re.MULTILINE)

    if match:
        name = match.group(1)
        print(f"Extracted name: {name}")
        return name
    else:
        print("No name found, retrying with different regex")
        name_regex = r"Name: *([^\\]*)"

        # Searching in the multiline string
        match = re.search(name_regex, str, re.MULTILINE)

        if match:
            name = match.group(1)
            print(f"Extracted name: {name}")
            return name
