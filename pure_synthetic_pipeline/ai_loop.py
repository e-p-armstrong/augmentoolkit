import asyncio
import os
import yaml
from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from input_field_handlers import create_dict_keys
from synthetic_data_skeleton import file_template
import random
import re
from faker import Faker
from create_prompt import do_first_pass

fake = Faker()


with open('config.yaml', 'r') as file:
    obj_conf = yaml.safe_load(file)

API_KEY_C = obj_conf["API"]["API_KEY_C"]
BASE_URL_C = obj_conf["API"]["BASE_URL_C"]
LOGICAL_MODEL_C = obj_conf["API"]["LOGICAL_MODEL_C"]
MODE = obj_conf["API"]["MODE"]
COMPLETION_MODE = obj_conf["SYSTEM"]["COMPLETION_MODE"]
METAPIPELINE_OUTPUT_FOLDER = obj_conf["PATH"]["METAPIPELINE_OUTPUT_FOLDER"]
METAPIPELINE_PY_FILE = obj_conf["PATH"]["METAPIPELINE_PY_FILE"]


def convert_task_name_to_key(task_name):
    return task_name.lower().replace(" ", "_")

def generate_task_functions(): # Why have a whole function for this? Legacy reasons, still need to update.
    function_template = """
async def generate_data(args, id):
    generate_conversation_path = (
        "generate_conversation.txt" if COMPLETION_MODE else "generate_conversation.yaml"
    )
    conversation_generator = GenerationStep(
        prompt_path=generate_conversation_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\\n\\n\\n\\n\\n\\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_large,
        prompt_folder=obj_conf["PATH"]["PROMPTS"],
        default_prompt_folder=DEFAULT_PROMPT_PATH,
    )
    
    result, full_output = await conversation_generator.generate(args)
    write_output_to_file(full_output, OUTPUT_FOLDER + "/conversation_generation", id)
    return result
"""

    return function_template

# Generates the control flow based on task names
def generate_data_routing():
    # chained ifelse that sends it to generate_{task_name}_data depending on convert_task_name_to_key(input_dict["task"])
    
    # Validation functions will have to be a dictionary with the task name as the key and the validation function as the value
    
    routing_template = """
async def generate_conv(input_dict=None, output_file=None):
    id = make_id()
    try:
        data_output = await validate_generation(gen_func=generate_data, validation_functions=[validate_repetition_callback(25, 3, 400)], retries=2, gen_func_args=[input_dict, id])
    except:
        return
    conversation_sharegpt = {"conversations": parse_conversation_to_sharegpt_format(data_output)}
    with open(output_file, "a") as f:
        f.write(json.dumps(conversation_sharegpt) + "\\n")"""
        
    return routing_template

# async def check_and_improve_prompt_file(path="", args={}, id=""):
    
# When executing the revise example prompt it must have outputs from the code-generated input fields visible, so it knows how to create convincing few-shot example inputs.


def format_list_items(list):
    return "\n* ".join(list)

def generate_system_prompts(overall_task_desc="", guidelines=[], objectives=[], avoids=[], end_message="Strive for variety in the interactions you write, representing realistic behavior in the participants. Try to avoid reusing phrases word for word."):
    system_prompt_template = f"""
{overall_task_desc}

**Rules and Guidelines:**
{format_list_items(guidelines)}

**You will also note the following things to avoid:**
{format_list_items(avoids)}

**Finally, the following objectives will be fulfilled, preferably in order:**
{format_list_items(objectives)}

{end_message}"""
    
    return system_prompt_template

def indent_string(string, indent_level=1):
    indent = " " * indent_level
    return re.sub(r"^", indent, string, flags=re.MULTILINE)

async def main():
    
    LOG_EVERYTHING = obj_conf["SYSTEM"]["LOG_EVERYTHING"]
    # First we need to create the prompt files and the synthetic data generation file.
    
    # Then we run the AI loop to modify it/run it until it's good.
    task_functions = generate_task_functions()
    
    input_fields, mutators, exclusives = await do_first_pass()
    
    data_routing = generate_data_routing()

    if LOG_EVERYTHING:
        print("Generated task functions and data routing")
        print("\n\nTASK FUNCTIONS\n\n--------------")
        print(task_functions)
        
        print("\n\nDATA ROUTING\n\n----------------")
        print(data_routing)
    
    input_fields_indented = indent_string(input_fields, 8)
    input_fields_dict_keys = indent_string(create_dict_keys(input_fields), 12)
    with open(METAPIPELINE_PY_FILE, 'w') as file:
        filled_in_template = file_template.format(generators=task_functions, data_routing=data_routing, input_fields=input_fields_indented, input_field_dict_items=input_fields_dict_keys, mutators=mutators, exclusives=exclusives)
        file.write(filled_in_template)
    
if __name__ == "__main__":
    asyncio.run(main())

    
# Things it codes by hand: valdiation functions, the faker input fields, and the prompts.

# Set up an interface, start from what the AI does and then "What does it need to do next to get to the final pipeline?"

# initial setup of the script template; fill stuff in based on the config.


