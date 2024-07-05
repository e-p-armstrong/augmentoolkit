import glob
import json
import re
import yaml


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
def write_output_to_file(output, directory, uuid):
    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path using the directory and UUID
    file_path = os.path.join(directory, f"{uuid}.yaml")

    # Write the output to the file
    with open(file_path, "w") as file:
        file.write(output)

    print(f"Output written to {file_path}")


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
            
            
convert_logging_to_dataset("judge_paragraph_generations")