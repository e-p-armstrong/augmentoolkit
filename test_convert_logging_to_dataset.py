import glob
import json
import os

import yaml

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)
    
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
        
    with open(output_file_path, "w",encoding='utf-8') as f:
        existing_files = glob.glob(
            os.path.join(output_dir, "*.txt")
        )
        
        print(existing_files)
        
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
    
convert_logging_to_dataset("judge_paragraph_generations")