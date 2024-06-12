import re

# from .multi_turn_conversation_grammar import multi_turn_conversation_grammar
import random
import os
import traceback
import json
import logging

import yaml
from gen_engine_core.generation_functions.safe_formatter import safe_format


class GenerationStep:
    def __init__(
        self,
        prompt_path="",  # relative to the Inputs directory
        regex=re.compile(r".*", re.DOTALL),  # take whole completion
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
            ],
        },
        completion_mode=True,  # Chat vs completion mode
        retries=0,
        engine_wrapper=None,
        logging_level=logging.INFO,  # Default logging level
        output_processor=lambda x: x,  # to ensure that control flow does not need to have decision code handling the outputs of the LLM, you can pass in a function to handle and modify the outputs (post regex) here. By default it's just the identity function and does nothing.
        return_input_too=True,
        default_prompt_folder="prompts",
        prompt_folder="prompts",
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
        self.prompt_folder = prompt_folder
        self.default_prompt_folder = default_prompt_folder
        logging.basicConfig(
            level=self.logging_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    async def generate(self, arguments={}):
        # Current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the full path of the prompt file
        ideal_path = os.path.join(
            current_dir, "..", "..", self.prompt_folder, self.prompt_path
        )
        if os.path.exists(ideal_path):
            full_prompt_path = ideal_path
        else:
            full_prompt_path = os.path.join(
                current_dir, "..", "..", self.default_prompt_folder, self.prompt_path
            )

        with open(full_prompt_path, "r") as pf:
            prompt = pf.read()

        # Submit generation and return response, retrying as needed
        times_tried = 0
        if self.completion_mode:
            prompt_formatted = safe_format(prompt, **arguments)
            while times_tried <= self.retries:
                try:
                    response, timeout = await self.engine_wrapper.submit_completion(
                        prompt_formatted, self.sampling_params
                    )
                    filtered_response = re.search(self.regex, response).group(1)
                    ret = self.output_processor(filtered_response)
                    if self.return_input_too:
                        return ret, prompt_formatted + filtered_response
                    return ret
                except Exception as e:
                    # logging.error(f"Error in Generation Step: {e}")
                    try:
                        if not self.engine_wrapper.mode == "llamacpp":
                            print("Response:")
                            print(response)
                    except:
                        pass
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")
        else:
            messages = yaml.safe_load(prompt)
            input_messages = []
            for message in messages:
                try:
                    input_messages.append(
                        {
                            "role": message["role"],
                            "content": safe_format(message["content"], **arguments),
                        }
                    )
                except Exception as e:
                    print("Error in formatting message:", message["content"])
                    input_messages.append(
                        {"role": message["role"], "content": message["content"]}
                    )
            messages = input_messages
            while times_tried <= self.retries:
                try:
                    # strip whitespace added by yaml load
                    messages = [
                        {
                            "role": message["role"],
                            "content": message["content"].strip(),
                        }
                        for message in messages
                    ]
                    # print("\n\n\nBEGIN DEBUG")
                    # print(messages)
                    # print("END DEBUG\n\n\n")
                    response, timeout = await self.engine_wrapper.submit_chat(
                        messages, self.sampling_params
                    )
                    ret = self.output_processor(response)
                    if self.return_input_too:
                        return ret, yaml.dump(
                            messages
                            + [
                                {
                                    "role": "assistant",
                                    "content": response,
                                    "timeout": timeout,
                                }
                            ],
                            default_flow_style=False,
                        )
                    return ret
                except Exception as e:
                    logging.error(f"Error in Generation Step: {e}")
                    # print(prompt_formatted)
                    logging.error(
                        f"Above prompt resulted in error, probably the model's fault: {e}"
                    )
                    traceback.print_exc()
                    times_tried += 1
            raise Exception("Generation step failed -- too many retries!")
