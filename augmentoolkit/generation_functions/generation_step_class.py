import re

# from .multi_turn_conversation_grammar import multi_turn_conversation_grammar
from .constants import LOGICAL_MODEL, INPUT_DIRECTORY
from .format_qatuples import format_qatuples
from .extract_name import extract_name
import random
import os
import traceback
import json
import logging

class GenerationStep:
    def __init__(self,
                 prompt_path="", # relative to the Inputs directory
                 regex=re.compile(r'.*', re.DOTALL), # take whole completion
                 sampling_params={
                     "temperature": 1,
                     "top_p": 1,
                     "max_tokens": 3000,
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
                        ]
                    },
                 completion_mode=True, # Chat vs completion mode
                 retries=0,
                 engine_wrapper=None,
                 logging_level=logging.INFO,  # Default logging level
                 output_processor=lambda x: x, # to ensure that control flow does not need to have decision code handling the outputs of the LLM, you can pass in a function to handle and modify the outputs (post regex) here. By default it's just the identity function and does nothing.
                 return_input_too=True
                ):
        self.prompt_path = prompt_path
        self.regex = regex
        self.sampling_params = sampling_params
        self.completion_mode = completion_mode
        self.retries = retries
        self.logging_level = logging_level
        self.output_processor = output_processor
        self.return_input_too = return_input_too
        if not engine_wrapper:
            raise Exception("Engine wrapper not passed in!")
        self.engine_wrapper = engine_wrapper
        logging.basicConfig(level=self.logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

    
    async def generate(self,arguments={}):
        # Current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Dynamic INPUT_DIRECTORY path (feel free to change, DragonFox, depending on what structure you have been working towards)
        full_prompt_path = os.path.join(current_dir, '..', '..', 'prompts',self.prompt_path)
        # Read file and escape all curly braces
        with open(full_prompt_path, 'r') as pf:
            prompt = pf.read()
            # Code to ensure that interpolation works, but curly braces are still allowed in the input
            # 1. Escape all curly braces
            prompt_escaped = prompt.replace('{', '{{').replace('}', '}}')
            # 2. Unescape curly braces that are associated with input keys
            for key in arguments.keys():
                prompt_escaped = prompt_escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}") # Somehow this works
            # 3. Format
            prompt_formatted = prompt_escaped.format(**arguments)
        # logging.info(f"Formatted prompt for generation: {prompt_formatted}")
        # Submit generation and return response, retrying as needed
        times_tried = 0
        if self.completion_mode:
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper.submit_completion(prompt_formatted, self.sampling_params)
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + filtered_response
                    return ret
                except Exception as e:
                    # logging.error(f"Error in Generation Step: {e}")
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")
        else:
            messages = json.loads(prompt_formatted)
            while times_tried <= self.retries:
                try:
                    response = await self.engine_wrapper.submit_chat(messages, self.sampling_params)
                    filtered_response = response.replace('"','\\"').replace("\n","\\n")#re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, "intermediate output broken in chat for now kek" #prompt_formatted + [{"role": "assistant", "content": filtered_response}]
                    return ret
                except Exception as e:
                    logging.error(f"Error in Generation Step: {e}")
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")