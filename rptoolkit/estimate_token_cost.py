import glob
from transformers import AutoTokenizer
import os
import yaml
from nltk.tokenize import sent_tokenize

tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
    )

# TODO leverage this as part of each pipeline, make it a util and make it general
# Would welcome a PR I need sleep >.<
# someoneeeeeeeee

def count_tokens(string):
    return len(tokenizer.encode(string))

def concatenate_and_count_tokens(file_paths):
    combined_content = ""

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                if isinstance(data, list):
                    # Take all but last element, we trace that in another way
                    for obj in data[:-1]:
                        if 'content' in obj:
                            combined_content += obj['content'] + " "
                elif isinstance(data, dict):
                    if 'content' in data:
                        combined_content += data['content'] + " "
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {file_path}")
                print(str(e))

    token_count = count_tokens(combined_content.strip())
    return token_count



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
    # try:
    #     with open(file_path, "r", encoding="utf-8") as f:
    #         content = f.read()
    # except Exception as e:
    #     print(f"\nError reading file {file_path}: {e}\n")
    #     return []
        
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
                    chunks_with_source.append((" ".join(current_chunk), source_name))
                    current_chunk = [sentence]
                    token_count = sentence_token_count
        else:
            if token_count + paragraph_token_count <= max_token_length:
                current_chunk.append(paragraph)
                token_count += paragraph_token_count
            else:
                chunks_with_source.append((" ".join(current_chunk), source_name))
                current_chunk = [paragraph]
                token_count = paragraph_token_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_source.append((" ".join(current_chunk), source_name))

    return chunks_with_source

def calculate_cost_worst_case(token_count_medium_input=0, token_count_large_input=0, token_count_medium_output=0, token_count_large_output=0, cost_medium_input=2.7, cost_large_input=8, cost_medium_output=8.1, cost_large_output=24):
    
    cost_medium_input_per_token = cost_medium_input / 1000000
    cost_large_input_per_token = cost_large_input / 1000000
    cost_medium_output_per_token = cost_medium_output / 1000000
    cost_large_output_per_token = cost_large_output / 1000000
    
    cost = (token_count_medium_input * cost_medium_input_per_token) + (token_count_large_input * cost_large_input_per_token) + (token_count_medium_output * cost_medium_output_per_token) + (token_count_large_output * cost_large_output_per_token)
    cost = 2 * cost  # we assume that everything gets regenerated once
    return cost

def calculate_cost_best_case(token_count_medium_input=0, token_count_large_input=0, token_count_medium_output=0, token_count_large_output=0, cost_medium_input=2.7, cost_large_input=8, cost_medium_output=8.1, cost_large_output=24):
    
    cost_medium_input_per_token = cost_medium_input / 1000000
    cost_large_input_per_token = cost_large_input / 1000000
    cost_medium_output_per_token = cost_medium_output / 1000000
    cost_large_output_per_token = cost_large_output / 1000000
    
    cost = (token_count_medium_input * cost_medium_input_per_token) + (token_count_large_input * cost_large_input_per_token) + (token_count_medium_output * cost_medium_output_per_token) + (token_count_large_output * cost_large_output_per_token)  # assume no regenerates
    return cost

if __name__ == "__main__":
    with open("config.yaml", 'r') as file:
        obj_conf = yaml.safe_load(file)
    prompt_folder = obj_conf["PATH"]["PROMPTS"]
    INPUT_FOLDER = obj_conf["PATH"]["INPUT"]
    cost_medium_input = obj_conf["COST"]["INPUT_A"]
    cost_large_input = obj_conf["COST"]["INPUT_B"]
    cost_medium_output = obj_conf["COST"]["OUTPUT_A"]
    cost_large_output = obj_conf["COST"]["OUTPUT_B"]

    # Get token count for each chunk
    path = f"{INPUT_FOLDER}/*" + ".txt"
    source_texts = glob.glob(path)
    chunks = []
    for source_text in source_texts:
        chunks += chunking_algorithm(source_text)

    # Define the step dependencies and output token counts
    step_info = {
        "chunk": {"output_tokens": 0, "feeds_into": ["generate_emotion_from_text", "extract_features"]},
        "generate_emotion_from_text": {"output_tokens": 170, "feeds_into": ["extract_features", "generate_scene_card", "generate_story"]},
        "extract_features": {"output_tokens": 230, "feeds_into": ["generate_scene_card", "generate_story"]},
        "generate_scene_card": {"output_tokens": 1000, "feeds_into": ["generate_story"]},
        "generate_story": {"output_tokens": 5000, "feeds_into": ["rate_story"]},
        "rate_story": {"output_tokens": 840, "feeds_into": []}
    }

    # Calculate the total input and output token counts for each step
    total_input_tokens = {step: 0 for step in step_info}
    total_output_tokens = {step: 0 for step in step_info}

    for chunk, _ in chunks:
        chunk_token_count = count_tokens(chunk)
        total_input_tokens["chunk"] += chunk_token_count
        total_output_tokens["chunk"] += chunk_token_count

        for step, info in step_info.items():
            if step == "chunk":
                continue

            input_token_count = 0
            for prev_step in step_info:
                if step in step_info[prev_step]["feeds_into"]:
                    input_token_count += step_info[prev_step]["output_tokens"]

            total_input_tokens[step] += input_token_count
            total_output_tokens[step] += info["output_tokens"]

    # Calculate the cost for medium and large models
    token_count_medium_input = sum(total_input_tokens[step] for step in step_info if step != "generate_story")
    token_count_large_input = total_input_tokens["generate_story"]
    token_count_medium_output = sum(total_output_tokens[step] for step in step_info if step != "generate_story")
    token_count_large_output = total_output_tokens["generate_story"]

    worst_case_cost = calculate_cost_worst_case(token_count_medium_input, token_count_large_input, token_count_medium_output, token_count_large_output, cost_medium_input, cost_large_input, cost_medium_output, cost_large_output)
    best_case_cost = calculate_cost_best_case(token_count_medium_input, token_count_large_input, token_count_medium_output, token_count_large_output, cost_medium_input, cost_large_input, cost_medium_output, cost_large_output)

    print(f"Worst-case cost estimate: ${worst_case_cost:.2f}")
    print(f"Best-case cost estimate: ${best_case_cost:.2f}")