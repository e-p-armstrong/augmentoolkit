import json
import os
import jinja2
import yaml
from generation.core_components.sharegpt_and_oai import rename_oai_messages_to_sharegpt


def create_meta_dataset(
    data_dicts: list[
        dict
    ],  # list of wide dicts over which to apply the meta datagen (extras+detail keys search and creation of data therefrom)
    meta_datagen_keys: list[str],  # list of details_keys to make data from
    meta_datagen_extras: list[
        str
    ],  # list of prompt paths (strings) to format the end keys into for additional datagen.
    input_processors: list[
        callable
    ],  # list of input processor functions that get called on the wide dict first, and help add new keys to the items that contain info we may want (for instance, multiple questions formatted into a single string).
    output_dir: str,  # file path to save the meta datagen data NOTE that output path has to be the actual path to an actual
):
    # Assert that data_dicts is not empty
    assert data_dicts, "data_dicts cannot be empty"

    # Input processors will have to have the logic to determine if they are being called on the right data dict. By default they will be called on all items in all data dicts.
    completion_lists = []
    chat_lists = []

    for data_dict in data_dicts:

        # Preview the data_dict structure (first 10 items)
        # print(f"Processing data_dict with {len(data_dict)} items")
        # preview_dict = dict(list(data_dict.items())[:10])
        # print(f"Data dict preview (first 10 items): {json.dumps(preview_dict, indent=2)}")
        completion_list = []
        chat_list = []
        for key, value in data_dict.items():
            for processor in input_processors:
                processor(
                    value
                )  # if it wants to add or mutate keys, it does so here. Otherwise, early return with nothing to avoid doing anything.
            if meta_datagen_keys:
                for detailkey in meta_datagen_keys:
                    if detailkey in value:
                        # do we create chat or completion data by default? Simple, depends on hte mode.
                        for item in value[detailkey]:  # it is a list of dicts now
                            # print("ITEM")
                            # print(item)
                            if item["completion_mode"]:
                                # create completion data
                                obj = {
                                    "full_input": item["full_input"],
                                    "full_response": item["full_response"],
                                    "detail_key": detailkey,
                                    "segments": [  # NOT ideal. Since the BOS token and EOS token need to be added for this to actually work. This will have to be done in the data processing pipeline (no LLM calls just processing of input data) for my meta model training. Since I don't have access to the tokenizer of the model I'm training at this stage.
                                        {
                                            "label": False,
                                            "text": item["full_input"],
                                        },  # we usually do not train on the input here, since the model is completing after this.
                                        {"label": True, "text": item["full_response"]},
                                    ],
                                }
                                completion_list.append(obj)
                            else:
                                # create chat data

                                # the input is the full_input, which is a list of messages. We don't actually... want to train on... the examples, do we. Hm that's another point. Well,  do we cut some of them out? Do we do it randomly where we just... no that should be done by some dat processing. We'll put all the messages in here and if we want to
                                # well that's the thing. We'd need the tokenizer and we would have to do segment-style training if we wanted to avoid overfitting on the same damn examples.
                                # do we make three datasets? Or two--one where we put the full input, examples included; and another where we just put the system prompt, the last user message, and the output?
                                # the model must be able to use few-shot examples. Arguably it should learn from them. But then it will massively overfit. No it'll be fine.
                                # no it would not be fine. They have to be masked. It should learn from the new input, and the output. And maybe the prompt. But not the examples.
                                # well that's up to the training prep pipeline, not the data collector. We take everything.
                                # so we will get to build a formatting pipeline that prepares it for training specifically. Makes sense. Interesting to have a pipeline with no LLM calls.
                                messages_to_use = item["full_input"] + [
                                    {
                                        "role": "assistant",
                                        "content": item["full_response"],
                                    }
                                ]
                                obj = {
                                    "full_input": item["full_input"],
                                    "full_response": item["full_response"],
                                    "detail_key": detailkey,
                                    "conversations": rename_oai_messages_to_sharegpt(
                                        messages_to_use
                                    ),
                                }
                                # NOTE we will need sophisticated handling of these conversation lists. Think about it. sysprompt or no, examples or no,  sometimes the input is in the sysprompt and sometimes it is in the last user message. It's about where it differs. You know I have changed my mind we should not have just a completion list and a chat list, we should have a separate list for each data dict.
                                chat_list.append(obj)

            if meta_datagen_extras:
                for extra in meta_datagen_extras:
                    # TODO format the keys at this item
                    # which includes 1. load the prompt from the "extra" path
                    # 2. safe format the values at the keys into the prompt (if the prompt is a yaml then it will be al ist of messages and we have to format the values into the text of each message; if it is .txt then we just format it into the text. We assume the inside of the yaml/txt is a jinja2 template, so we can use the safe format function.)
                    # 3. append the result to an output list. Let's say we have one output list for each data dict. But how do we name them. Well, let's just save everything to the same file, at the output path. Less of a pain with the config in the end anyway.
                    # load the prompt from the extra path
                    # if it is yaml then it is a list of messages in oai format. IF it is a txt then it is a jinja2 template.

                    # Extract the sub-keys of value
                    # Extract nested dictionary values and flatten them to the top level
                    flattened_value = value.copy()
                    keys_to_process = list(value.keys())

                    for key in keys_to_process:
                        if isinstance(value[key], dict):
                            # For each nested dictionary, add its keys to the top level with prefix
                            for subkey, subvalue in value[key].items():
                                flattened_key = f"{key}_{subkey}"
                                flattened_value[flattened_key] = subvalue

                    # Use the flattened dictionary for template rendering
                    value = flattened_value

                    with open(extra, "r") as f:
                        prompt = f.read()
                    if extra.endswith(".yaml"):
                        # set prompt to the yaml
                        prompt = yaml.safe_load(prompt)
                        # format the values into the text of each message
                        for message in prompt:
                            # set message to a jinja2 template
                            message_template = jinja2.Template(message["content"])
                            # format the values into the text of each message
                            message["content"] = message_template.render(**value)

                        prompt = rename_oai_messages_to_sharegpt(prompt)
                        chat_list.append(
                            {
                                "detail_key": extra,  # detail key will either be the detail key itself or teh extra path
                                "conversations": prompt,
                            }
                        )
                    else:
                        raise ValueError(f"Prompt {extra} is not a yaml file.")
                        # it does not make sense to support completion mode for extras. Because we'd have to just append text.
                        # # it is a jinja2 template.
                        # prompt = jinja2.Template(prompt)
                        # # format the values into the text of each message
                        # prompt = prompt.render(**value)
                        # # in this case we simply append the text, I mean we probably won't use extras for completion mode much, completion mode is mostly deprecated anyway

        completion_lists.append(completion_list)
        chat_lists.append(chat_list)

    # Check if any data was generated before saving
    any_data_generated = any(
        any(completion_list) for completion_list in completion_lists
    ) or any(any(chat_list) for chat_list in chat_lists)
    assert (
        any_data_generated
    ), "No data was generated. Both completion_lists and chat_lists are empty."

    # save the completion lists
    # to a completion_lists directory under the output_dir
    os.makedirs(os.path.join(output_dir, "completion_lists"), exist_ok=True)
    for idx, completion_list in enumerate(completion_lists):
        with open(
            os.path.join(output_dir, "completion_lists", f"{idx}.json"), "w"
        ) as f:
            json.dump(completion_list, f, indent=4)

    # save the chat lists
    # to a chat_lists directory under the output_dir
    os.makedirs(os.path.join(output_dir, "chat_lists"), exist_ok=True)
    for idx, chat_list in enumerate(chat_lists):
        with open(os.path.join(output_dir, "chat_lists", f"{idx}.json"), "w") as f:
            json.dump(chat_list, f, indent=4)


### NOTE
### The "incorporate meta datagen into a pipeline" checklist:
# checklist:
# 1. add do meta datagen as an arg to the main node
# 2. add include details to all execute pipeline steps
# 3. add detail keys to all pipelinestep inits
# 4. add the keys to the right palce in the config
# 5. save the meta data gens by adding the thing at the nd
