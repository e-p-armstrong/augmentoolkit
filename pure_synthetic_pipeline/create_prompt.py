import asyncio
import io
import json
import os
import sys
import traceback
import yaml
from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.generation_functions.generation_step_class import GenerationStep
from gen_engine_core.utillity_functions import make_id, write_output_to_file
import random
import re
from faker import Faker
from input_field_handlers import code_to_function, code_to_format_string, execute_code_to_dict, extract_first_code_block, format_prompt_yaml

fake = Faker()

with open('config.yaml', 'r') as file:
    obj_conf = yaml.safe_load(file)

API_KEY_C = obj_conf["API"]["API_KEY_C"]
BASE_URL_C = obj_conf["API"]["BASE_URL_C"]
LOGICAL_MODEL_C = obj_conf["API"]["LOGICAL_MODEL_C"]
API_KEY_D = obj_conf["API"]["API_KEY_D"]
BASE_URL_D = obj_conf["API"]["BASE_URL_D"]
LOGICAL_MODEL_D = obj_conf["API"]["LOGICAL_MODEL_D"]
MODE = obj_conf["API"]["MODE"]
MODE_D = obj_conf["API"]["MODE_D"]
COMPLETION_MODE = obj_conf["SYSTEM"]["COMPLETION_MODE"]
METAPIPELINE_OUTPUT_FOLDER = obj_conf["PATH"]["METAPIPELINE_OUTPUT_FOLDER"]



engine_wrapper_c = EngineWrapper(
    model=LOGICAL_MODEL_C,
    api_key=API_KEY_C,
    base_url=BASE_URL_C,
    mode=MODE,
)

engine_wrapper_d = EngineWrapper(
    model=LOGICAL_MODEL_D,
    api_key=API_KEY_D,
    base_url=BASE_URL_D,
    mode=MODE_D,
)

### Create Meta-pipeline Steps

async def check_requirements(args, id):
    check_requirements_path = (
        "check_requirements.txt" if COMPLETION_MODE else "check_requirements.yaml"
    )
    requirements_generator = GenerationStep(
        prompt_path=check_requirements_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await requirements_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/requirements_generation", id)
    return result

async def modify_requirements(args, id):
    modify_requirements_path = (
        "modify_requirements.txt" if COMPLETION_MODE else "modify_requirements.yaml"
    )
    requirements_generator = GenerationStep(
        prompt_path=modify_requirements_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await requirements_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/requirements_generation", id)
    return result

async def check_input_fields(args, id):
    check_input_fields_path = (
        "check_input_fields.txt" if COMPLETION_MODE else "check_input_fields.yaml"
    )
    check_input_fields_generator = GenerationStep(
        prompt_path=check_input_fields_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await check_input_fields_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/input_fields_generation", id)
    return result

async def modify_input_fields(args, id):
    modify_input_fields_path = (
        "modify_input_fields.txt" if COMPLETION_MODE else "modify_input_fields.yaml"
    )
    modify_input_fields_generator = GenerationStep(
        prompt_path=modify_input_fields_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await modify_input_fields_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/input_fields_generation", id)
    return result

## OUTPUT PROCESSOR
def parse_check_prompt_response(response):
    response = response.strip()
    if 'REVISE EXAMPLE' in response:
        try:
            index = int(response.split('REVISE EXAMPLE')[1].split()[0]) - 1
            return ('example', index)
        except (IndexError, ValueError):
            pass
    elif 'REVISE INPUT TEMPLATE' in response:
        return ('template', None, response)
    elif 'ADD SPECIFIC INSTRUCTION' in response:
        return ('instruction', None, response)
    elif 'NO REVISIONS NEEDED' in response:
        return False
    return False

async def check_prompt(args, id):
    check_prompt_path = (
        "check_prompt.txt" if COMPLETION_MODE else "check_prompt.yaml"
    )
    check_prompts_generator = GenerationStep(
        prompt_path=check_prompt_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
        output_processor=parse_check_prompt_response
    )
    
    result, full_output = await check_prompts_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/prompts_generation", id)
    return result

# Output processor: if FLAWED in output, return False, else return True
def parse_check_validation_response(response):
    response = response.strip()
    if 'FLAWED' in response:
        return False, response
    return True, response

async def validate_data(args, id):
    validate_data_path = (
        "validate_data.txt" if COMPLETION_MODE else "validate_data.yaml"
    )
    validate_data_generator = GenerationStep(
        prompt_path=validate_data_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
        output_processor=parse_check_validation_response,
    )
    
    result, full_output = await validate_data_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/validation_generation", id)
    res, bool = result
    return res, bool

async def rewrite_code(args, id, error=False):
    rewrite_code_path = (
        "rewrite_code.txt" if COMPLETION_MODE else "rewrite_code.yaml"
    )
    rewrite_code_generator = GenerationStep(
        prompt_path=rewrite_code_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await rewrite_code_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/code_generation", id)
    return result

async def modify_example(args, id):
    modify_example_path = (
        "modify_example.txt" if COMPLETION_MODE else "modify_example.yaml"
    )
    modify_example_generator = GenerationStep(
        prompt_path=modify_example_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_d,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await modify_example_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/example_generation", id)
    return result

async def modify_template(args, id):
    modify_template_path = (
        "modify_template.txt" if COMPLETION_MODE else "modify_template.yaml"
    )
    modify_template_generator = GenerationStep(
        prompt_path=modify_template_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await modify_template_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/template_generation", id)
    return result

async def modify_instruction(args, id):
    modify_instruction_path = (
        "modify_instruction.txt" if COMPLETION_MODE else "modify_instruction.yaml"
    )
    modify_instruction_generator = GenerationStep(
        prompt_path=modify_instruction_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await modify_instruction_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/instruction_generation", id)
    return result

async def add_example(args, id):
    add_example_path = (
        "add_example.txt" if COMPLETION_MODE else "add_example.yaml"
    )
    add_example_generator = GenerationStep(
        prompt_path=add_example_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await add_example_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/example_generation", id)
    return result

async def check_validation_functions(args, id):
    check_validation_functions_path = (
        "check_validation_functions.txt" if COMPLETION_MODE else "check_validation_functions.yaml"
    )
    check_validation_functions_generator = GenerationStep(
        prompt_path=check_validation_functions_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await check_validation_functions_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/check_validation_functions_generation", id)
    return result

async def modify_validation_functions(args, id):
    modify_validation_functions_path = (
        "modify_validation_functions.txt" if COMPLETION_MODE else "modify_validation_functions.yaml"
    )
    modify_validation_functions_generator = GenerationStep(
        prompt_path=modify_validation_functions_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await modify_validation_functions_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/modify_validation_functions_generation", id)
    return result

async def rewrite_validation_functions(args, id):
    rewrite_validation_functions_path = (
        "rewrite_validation_functions.txt" if COMPLETION_MODE else "rewrite_validation_functions.yaml"
    )
    rewrite_validation_functions_generator = GenerationStep(
        prompt_path=rewrite_validation_functions_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
    )
    
    result, full_output = await rewrite_validation_functions_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/rewrite_validation_functions_generation", id)
    return result

def evalaute_output_inspection(output):
    if "PASSES INSPECTION" in output:
        return True
    return False

async def inspect_output(args, id):
    inspect_output_path = (
        "inspect_output.txt" if COMPLETION_MODE else "inspect_output.yaml"
    )
    inspect_output_generator = GenerationStep(
        prompt_path=inspect_output_path,
        sampling_params={
            "max_tokens": 4000,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>", "\n\n\n\n\n\n"],
        },
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper_c,
        prompt_folder=obj_conf["PATH"]["META_PIPELINE_PROMPTS"],
        default_prompt_folder=obj_conf["PATH"]["META_PIPELINE_OVERRIDES"],
        output_processor=evalaute_output_inspection
    )
    
    result, full_output = await inspect_output_generator.generate(args)
    write_output_to_file(full_output, METAPIPELINE_OUTPUT_FOLDER + "/inspect_output_generation", id)
    return result


def extract_requirements_dict(str):
    str = str.split("\n")
    result = {}
    current_header = None
    
    for line in str:
        line = line.strip()
        
        if line.startswith("**") and line.endswith("**"):
            current_header = line[2:-2].strip()
            result[current_header] = []
        elif line.startswith("*"):
            if current_header:
                result[current_header].append(line[1:].strip())
    
    return result

def format_list_items(list):
    return "* " + "\n* ".join(list)

# NOTE need to capitalize "AI" in input fields somehow if it's a separate word. We'll get "Ai" otherwise.

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


def extract_functions(functions_string):
    # Regular expression pattern to match function definitions
    pattern = r'def\s+(\w+)\s*\(.*?\):\s*\n(?:\s+.*\n)*?(?=\ndef|\Z)'
    
    # Extract functions using the regular expression
    functions = re.findall(pattern, functions_string, re.MULTILINE)
    
    return functions

async def do_first_pass():
    
    # Set everything to None so that the functions can see that this is the first go
    reqs = None
    input_fields = None
    prompt = None
    
    id = make_id() # ID is set per loop
    
    # Create requirements
    modify_requirements_args = {
        "reqs": reqs,
        "task_description": obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"]
    }
    
    reqs = await modify_requirements(modify_requirements_args, id)
    reqs_dict = extract_requirements_dict(reqs)
    print(reqs_dict)
    print(reqs)
    
    system_prompt = generate_system_prompts(overall_task_desc=obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"], guidelines=reqs_dict["GUIDELINES"], objectives=reqs_dict["OBJECTIVES"], avoids=reqs_dict["AVOIDS"])
    
    print(system_prompt)
    prompt_list = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "FIRST EXAMPLE INPUT PLACEHOLDER"
        },
        {
            "role": "assistant",
            "content": "FIRST EXAMPLE OUTPUT PLACEHOLDER"
        },
        {
            "role": "user",
            "content": "SECOND EXAMPLE INPUT PLACEHOLDER"
        },
        {
            "role": "assistant",
            "content": "SECOND EXAMPLE OUTPUT PLACEHOLDER"
        },
        {
            "role": "user",
            "content": "THIRD EXAMPLE INPUT PLACEHOLDER"
        },
        {
            "role": "assistant",
            "content": "THIRD EXAMPLE OUTPUT PLACEHOLDER"
        },
        {
            "role": "user",
            "content": "ACTUAL INPUT TEMPLATE PLACEHOLDER"
        },
    ]
    
    modify_input_fields_args = {
        "input_fields": input_fields,
        "reqs": reqs,
        "task_description": obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"]
    }
    # Modify input fields
    input_fields_original = await modify_input_fields(modify_input_fields_args, id)
    
    print(input_fields)
    input_fields = extract_first_code_block(input_fields_original)
    print("\n\nInput fields extracted:\n\n", input_fields)
    
    
    increment = 0
    while increment < 5:
        context = {}
        increment += 1
        try:
            # Execute the AI-generated code
            exec(input_fields, globals(), context)

            # If the code executes successfully, validate the outputs
            validation_success, validation_output = await validate_data({
               "overall_task_desc": obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"],
                "validation_code": input_fields,
                "context": "\n".join([f"{k} = {v}" for k, v in context.items()])
            }, id + f"_{increment}")
            
            # print("\n\n---! SHOULD BE RESULT, BOOL")
            # print(validation_output, validation_success)
            # print("\n\n------\n\n")
            if validation_success:
                print("Validation successful:", context)
                break  # Exit loop if validation is successful
            else:
                # If validation fails, optionally modify the code
                # Show the context as a human-readable string
                rewrite_code_args = {
                    "overall_task_desc": obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"],
                    "validation_code": input_fields,
                    "validation_output": validation_output,
                    "context": "\n".join([f"{k} = {v}" for k, v in context.items()])
                }
                input_fields_original = await rewrite_code(rewrite_code_args, id + f"_{increment}")
                input_fields = extract_first_code_block(input_fields_original)
                print("Code rewritten due to validation failure.")
        except Exception as e:
            # Handle errors in execution
            print("Error executing code:", str(e))
            # Optionally modify the code based on the error
            
            rewrite_code_args = {
                    "overall_task_desc": obj_conf["REQUIREMENTS"]["OVERALL_TASK_DESCRIPTION"],
                    "validation_code": input_fields,
                    "validation_output": f"Python encountered a runtime error while executing the code: {str(e)}",
                    "context": "None; the code errored before context could be generated fully."
                }
            
            input_fields_original = await rewrite_code(rewrite_code_args, id + f"_{increment}")
            input_fields = extract_first_code_block(input_fields_original)
            print("Code rewritten due to error.") # Now that this is how that is done
    
    format_func = code_to_function(input_fields)
    print(input_fields)
    # Note: either do away with the non-profile fields, or filter out the profile field, since it results in duplication.
    
    all_exclusives = reqs_dict["EXCLUSIVES"]
    
    # Create first example input
    # No mutators, one exclusive
    example_input_dict = execute_code_to_dict(input_fields)
    exclusive_0 = random.choice(all_exclusives)
    example_input_dict["exclusive"] = exclusive_0
    example_input = format_func(example_input_dict)
    
    prompt_list[1]["content"] = example_input
    
    # Create second example input
    # Two mutators
    # print("\n\nDEBUG MUTATORS\n\n")
    all_mutators = reqs_dict["MUTATORS"]
    # print(all_mutators)
    # print("========================\n\n\n")
    # Select two random mutators, one exclusive (reroll once if it is a duplicate of the first exclusive)
    
    exclusive_1 = random.choice(all_exclusives)
    if exclusive_1 == exclusive_0:
        exclusive_1 = random.choice(all_exclusives)
    random_mutators_1 = " && ".join(random.sample(all_mutators, 2))
    example_input_dict_1 = execute_code_to_dict(input_fields)
    example_input_dict_1["mutators"] = random_mutators_1
    example_input_dict_1["exclusive"] = exclusive_1
    example_input_1 = format_func(example_input_dict_1)
    
    prompt_list[3]["content"] = example_input_1
    
    # Create third example input
    # Three mutators, one exclusive
    exclusive_2 = random.choice(all_exclusives)
    if exclusive_2 == exclusive_1 or exclusive_2 == exclusive_0:
        exclusive_2 = random.choice(all_exclusives)
    
    random_mutators_2 = " && ".join(random.sample(all_mutators, 3))
    example_input_dict_2 = execute_code_to_dict(input_fields)
    example_input_dict_2["mutators"] = random_mutators_2
    example_input_dict_2["exclusive"] = exclusive_2
    example_input_2 = format_func(example_input_dict_2)
    
    prompt_list[5]["content"] = example_input_2
    
    # Create input template:
    format_string = code_to_format_string(input_fields)
    prompt_list[-1]["content"] = format_string
    
    # make the prompts file
    os.makedirs(obj_conf["PATH"]["PROMPTS"], exist_ok=True)
    
    
    print(prompt_list)
    with open(obj_conf["PATH"]["PROMPTS"] + "/generate_conversation.yaml", 'w') as f:
        f.write(yaml.dump(prompt_list))
        prompt = yaml.dump(prompt_list[1:-1])
    
    # Needs to run the faker outputs once or twice in an environment of some kind, see if it makes sense, and adjust if necessary.
    
    # TODO filter out the profile field, since it results in duplication.
    
    # return
    for i in range(2,7,2):
        modify_example_args = {
            "system_prompt": system_prompt,
            "full_prompt": prompt,
            "example_to_modify": prompt_list[i],
            "input": prompt_list[i - 1]["content"],
        }
        new_example_output = await modify_example(modify_example_args, id + f"_{i}") # TODO modify the function to split the output into input and output edits
        prompt_list[i]["content"] = new_example_output
        
        # After each loop, update the prompt string and the file
        with open(obj_conf["PATH"]["PROMPTS"] + "/generate_conversation.yaml", 'w') as f:
            f.write(yaml.dump(prompt_list))
            prompt = yaml.dump(prompt_list[1:-1]) # We only show the AI the examples, not the system prompt, when it's editing things
    # The validation functions MUST also be called on the examples, to ensure that they are valid. In the test_case_strings, assert will be used to check that the outputs are as expected. The examples should be asserted True (validation functions return true or false).
        
    return input_fields, reqs_dict["MUTATORS"], reqs_dict["EXCLUSIVES"]

if __name__ == "__main__":
    asyncio.run(do_first_pass())
   