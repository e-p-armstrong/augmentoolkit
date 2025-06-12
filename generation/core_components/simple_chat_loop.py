import asyncio
import logging
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from generation.core_components.chunking import count_tokens_specific_model
from jinja2 import Template
import re


def format_messages_into_string(messages, prompt_template):
    # Example of what a "prompt_template" might be (it is a str)
    # {% for message in messages %}{% if (message['role'] == 'system') %}{{message['content'] + '\n'}}{% elif (message['role'] == 'user') %}{{'Human: ' + message['content'] + ' **Finished.**' + '\n'}}{% elif message['role'] == 'assistant' %}{{'AI: ' + message['content'] + ' **Finished.**' + '\n'}}{% endif %}{% endfor %}
    # and given a list of things with user and assistant messages we format this into a string.
    template = Template(prompt_template)
    return template.render(messages=messages)


def get_stop_tokens(prompt_template):
    # Gets the bit of text that precedes the human response. And also gets the bit of text that ends off the AI response. Returns a list of strings.
    # for instance, in the following it would return [' **Finished.**', 'Human: ']
    # # {% for message in messages %}{% if (message['role'] == 'system') %}{{message['content'] + '\n'}}{% elif (message['role'] == 'user') %}{{'Human: ' + message['content'] + ' **Finished.**' + '\n'}}{% elif message['role'] == 'assistant' %}{{'AI: ' + message['content'] + ' **Finished.**' + '\n'}}{% endif %}{% endfor %}

    stop_tokens = []

    # Extract what comes after assistant messages (typically ends AI response)
    # Look for pattern like: 'AI: ' + message['content'] + ' **Finished.**' + '\n'
    assistant_pattern = r"message\['role'\]\s*==\s*'assistant'.*?\{\{\s*'[^']*'\s*\+\s*message\['content'\]\s*\+\s*'([^']+)'"
    assistant_match = re.search(assistant_pattern, prompt_template)
    if assistant_match:
        stop_tokens.append(assistant_match.group(1))

    # Extract what comes before user messages (typically starts human response)
    # Look for pattern like: 'Human: ' + message['content'] + ' **Finished.**' + '\n'
    user_pattern = r"message\['role'\]\s*==\s*'user'.*?\{\{\s*'([^']+)'\s*\+\s*message\['content'\]"
    user_match = re.search(user_pattern, prompt_template)
    if user_match:
        stop_tokens.append(user_match.group(1))

    # Remove duplicates while preserving order
    seen = set()
    unique_stop_tokens = []
    for token in stop_tokens:
        if token not in seen:
            seen.add(token)
            unique_stop_tokens.append(token)

    return unique_stop_tokens


def get_assistant_prefix(prompt_template):
    # Extracts the text that comes before message['content'] in assistant messages
    # For example, from 'AI: ' + message['content'] + ' **Finished.**', this would return 'AI: '

    # Look for pattern like: 'AI: ' + message['content'] + something
    assistant_prefix_pattern = r"message\['role'\]\s*==\s*'assistant'.*?\{\{\s*'([^']+)'\s*\+\s*message\['content'\]"
    assistant_match = re.search(assistant_prefix_pattern, prompt_template)
    if assistant_match:
        return assistant_match.group(1)

    # Fallback: look for common patterns
    if "'assistant'" in prompt_template:
        # Try to find any string that comes before content in assistant block
        fallback_pattern = r"'assistant'.*?\{\{\s*'([^']*?)'\s*\+.*?content"
        fallback_match = re.search(fallback_pattern, prompt_template)
        if fallback_match:
            return fallback_match.group(1)

    # Default fallback
    return ""


async def simple_chat_loop(
    system_prompt, prompt_template, context_length, finetune_hub_model_id
):  # sends completions and infers the end token from the constant text at the start of the human role.

    # Set logging level to ERROR to suppress verbose output
    logging.getLogger().setLevel(logging.ERROR)

    count_tokens = count_tokens_specific_model(finetune_hub_model_id)  # string -> int

    # PROBLEM IDENTIFIED do not write out thoughts still has thoguhts probably because of introduction of thought rephrase after. Has to be before.
    print(
        "You are now chatting with an LLM! Press Control+C or type 'exit' and hit enter to exit."
    )

    engine = EngineWrapper(
        api_key="Notused!We are local",
        base_url="http://127.0.0.1:8080/v1",  # Firstly, note that there IS a difference in objects returned between v1 and not. Essential to hit llama.cpp's v1 endpoint
        mode="api",
        model="itmattersnot",
    )

    stop_token = get_stop_tokens(prompt_template)
    assistant_prefix = get_assistant_prefix(prompt_template)

    print("Your stop token is:")
    print(stop_token)
    print("Your assistant prefix is:")
    print(repr(assistant_prefix))

    all_messages = [{"role": "system", "content": system_prompt}]

    # print("Your system prompt is")
    # print(system_prompt)

    while True:
        usr_string = input("LLM is ready >>> ")
        if "exit" == usr_string:
            return

        total_tokens = count_tokens(usr_string)
        for msg in all_messages:
            msg_tokens = count_tokens(msg["content"])
            total_tokens = total_tokens + msg_tokens

        shown_messages = all_messages.copy()

        while total_tokens >= context_length:
            if (
                not len(shown_messages) == 1
            ):  # if we are down to the system prompt and the most recent user message and are over, then we have a dire problem
                removed_message = shown_messages.pop(
                    1
                )  # remove the message after the system prompt
                removed_tokens = count_tokens(removed_message["content"])
                total_tokens = total_tokens - removed_tokens
            else:
                print(
                    f"\n\nNO MESSAGE SENT -- User message too long, user message + system message = {total_tokens} which is > {context_length}\n\n"
                )
                break

        usr_message = {"role": "user", "content": usr_string}
        shown_messages.append(usr_message)
        message_string = format_messages_into_string(
            shown_messages, prompt_template=prompt_template
        )

        # Add assistant prefix for generation prefilling
        prefilled_prompt = message_string + assistant_prefix

        response, timeout = await engine.submit_completion(
            prompt=prefilled_prompt,
            sampling_params={
                "temperature": 0.4,  # yes, it's not greedy. We want a good token probability distribtion for quality outputs.
                "min_p": 0.3,
                "top_p": 0.9,
                "stop": stop_token,
            },  # sampling params should really be a list of dicts param_name: value. Because that way you could control the order in a nice way.
            return_completion_only=True,
        )  # TODO make this be able to stream.
        # TODO before streaming, I wonder if we can show a spinner while we wait...

        all_messages.append(usr_message)
        assistant_message = {
            "role": "assistant",
            "content": response,  # Include prefix in stored response
        }
        all_messages.append(assistant_message)

        print(response)

        # receive input √
        # check if exit √
        # truncate shown messages if too long √
        # format into string √
        # generate output √
        # loop again back to input √
