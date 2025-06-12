from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
import re


def custom_process_input_data(self, input_data):
    # Custom processing logic
    return input_data, {}  # Return a tuple with an empty dict for additional args


def judge_paragraph_processor(
    determination,
):  # TODO extract to separate file to avoid muddying the control flow code
    if (
        "unsuitable" in determination.lower()
        or "table of contents" in determination.lower()
    ):
        return False  # control flow has been modified to use the information it has, based on the determination of the output processors
    elif "suitable" in determination.lower():
        return True
    else:
        raise Exception("Incorrect format")


# filter_chunks = create_pipeline_step_function(pipeline_kwargs={
#             # "prompt_folder": PROMPTS,
#             # "default_prompt_folder": DEFAULT_PROMPTS,
#             "prompt_path": "filter",
#             "regex": re.compile(
#         r"Reasoning and thought process \(reason intelligently\):(.+)",
#         re.DOTALL | re.IGNORECASE,
#     ),
#             "sampling_params": {
#                 "max_tokens": 2000,
#                 "stop": [
#                     "### Response",
#                     "\n\n\n\n\n",
#                     "</s>",
#                     "# Input:",
#                     "[INST]",
#                     "### Instruction",
#                     "### Information",
#                     "## Information",
#                     "## Instruction",
#                     "Name:",
#                     "<|eot_id|>",
#                     "<|start_header_id|>",
#                     "<|end_header_id|>",
#                 ],
#                 "temperature": 0.8,
#                 # "top_k": -1,
#                 "top_p": 0.9,
#                 # "min_p": 0.6,
#             },
#             "output_file": "correction_data",
#             "intermediate_output_path": "intermediate_generations",
#             "save_path": "saved_readable_generations",
#             "result_key": "judgement",
#             # "use_stop": USE_STOP,
#             # "completion_mode": COMPLETION_MODE,
#             "output_processor": judge_paragraph_processor, # in the interface, this will be greyed out, since it has an external reference
#             "max_retries": 3,
#             "method_overrides": {
#                 'process_input_data': custom_process_input_data, # same here. Only conventional datatypes escape the cut.
#             },
#     })


def create_filter_chunks_step(
    output_file: str,
):
    filter_chunks_step = PipelineStep(
        prompt_path="filter",
        regex=re.compile(
            r"Reasoning and thought process \(reason intelligently\):(.+)",
            re.DOTALL | re.IGNORECASE,
        ),
        sampling_params={
            "max_tokens": 2000,
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
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.8,
            "top_p": 0.9,
        },
        output_file=output_file,
        intermediate_output_path="intermediate_generations",
        save_path="saved_readable_generations",
        result_key="judgement",
        details_key="judgement_details",
        output_processor=judge_paragraph_processor,
        max_retries=3,
        method_overrides={
            "process_input_data": custom_process_input_data,
        },
    )
    return filter_chunks_step


def filter_out_failed_items(item_list, key_to_check="judgement"):
    i = 0
    while i < len(item_list):
        if not item_list[i][key_to_check]:
            item_list.pop(i)
        else:
            i += 1


def filter_out_failed_items_dict(item_dict, key_to_check="judgement"):
    keys_to_remove = []
    for key, value in item_dict.items():
        # print("ITEM DICT ITEM KEY")
        # print(key)
        # print("ITEM DICT ITEM KEYS")
        # print([k for k in value.keys()])
        # print(key)
        if not value.get(key_to_check):
            # print()
            keys_to_remove.append(key)
            # print("Adding key to remove")

    for key in keys_to_remove:
        del item_dict[key]
