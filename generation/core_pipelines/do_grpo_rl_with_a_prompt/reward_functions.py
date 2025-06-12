###################### NOTE -- META HELPERS #######################################
# will contain all reward functions
import asyncio
from difflib import SequenceMatcher
import re

import yaml

from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.generation_functions.safe_formatter import safe_format
from generation.core_components.setup_components import make_relative_to_self


REWARD_FUNCTIONS_DICT = {
    # "key": func
}


def register_reward_function(name):
    """Decorator to register a reward function."""

    def decorator(func):
        REWARD_FUNCTIONS_DICT[name] = func
        return func

    return decorator


def get_funcs_for_str_list(list):
    return [REWARD_FUNCTIONS_DICT[key] for key in list]


def get_func_for_str(str):
    return REWARD_FUNCTIONS_DICT[str]


###################### NOTE -- CUSTOM HELPERS #######################################


def extract_xml_answer(text: str, tagname: str = "answer") -> str:
    """Extract answer from XML-formatted text.

    Not used for the LLM's output format actually, but for the evaluation of the LLM's output. Hence, no cot_start or cot_end.
    """

    # Remove any text between <think> and </think> tags
    pattern = r"<think>.*?</think>"
    text = re.sub(pattern, "", text, flags=re.DOTALL)

    answer = text.split(f"<{tagname}>")[-1]
    answer = answer.split(f"</{tagname}>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from hash-formatted text."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_llm_rating_numerical(completion: str, max: int = 4.1) -> str | None:
    """Extract rating from completion, handling both class-based and numeric ratings.
    Returns one of: "Bad", "Passable", "Perfect"
    Raises ValueError if rating is invalid."""

    print(
        f"Attempting to extract rating from completion:\n\n------{completion}\n\n----------"
    )

    pattern = r"<rating>(.*?)</rating>"
    match = re.finditer(pattern, completion)
    matches = list(match)
    if not matches:
        print("No rating tags found in completion")
        return None

    rating = matches[-1].group(1).strip()
    print(f"Extracted raw rating: {rating}")

    # If not, try to convert from number
    try:
        num_rating = float(rating)
        if 0 <= num_rating <= max:
            return num_rating
        else:
            error_msg = f"Invalid numeric rating: {num_rating}. Out of range."
            print(error_msg)
            raise ValueError(error_msg)
    except ValueError as e:
        error_msg = f"Invalid rating format: {rating}. Must be 'Bad', 'Passable', 'Perfect' or 0, 0.5, 1"
        print(error_msg)
        raise ValueError(error_msg) from e


def validate_cot_format(
    solution_str, cot_end=None, cot_start=None
):  # may use a different default? Who knows. Also yes, we use markdown for this rather than xml. Because I like this format more and I suspect it fits with the models we're training more. You can define your own format functions.
    """
    Determine if the solution str is of the valid format:
    - Must contain exactly one cot_end section
    - Must NOT contain any cot_start headers (prefill expected with add_generation_prompt)
    """
    # Check if string is None or empty
    if not solution_str:
        return False
    
    if cot_start == None:
        return True
    
    if cot_end == None:
        return True # vacuously true; good default for when COT is not being used

    # Check for exactly one Answer section
    answer_count = solution_str.count(
        cot_end
    )  # TODO adjust this and the sft pipelines too, to use "Final Response:". And in general, I want this to be more flexible. Oh I know -- answer tags passed in from the config file. And we'll ofc have a separate function for xml things!
    if answer_count != 1:
        return False

    # Check for forbidden Thought Process headers
    if cot_start in solution_str:
        return False

    return True


def extract_cot_answer(response, cot_end="Answer:", cot_start="Thought Process:"):
    """Extract CoT and answer, and return as a tuple."""
    parts = response.split(cot_end, 1)
    if len(parts) < 2:
        return None, None
    cot = parts[0].strip()
    # remove cot_start from cot... is what I would do but we don't have to because it's prefilled
    answer = parts[1].strip()
    return cot, answer


def check_thought_answer_similarity(
    response, threshold=90, cot_end=None, cot_start=None
):
    """Check if thought process and answer sections are overly similar. True if similarity exceeds the given threshold percentage."""
    
    if cot_end == None:
        return False # vacuously (not) true; good default for when cot is not used

    if cot_start == None:
        return False

    # Split into thought and answer sections
    parts = response.split(cot_end, 1)
    if len(parts) < 2:
        return False  # No answer section

    thought_part = parts[0].strip()
    answer_part = parts[1].strip()

    # Calculate similarity ratio between sections
    similarity = SequenceMatcher(None, thought_part, answer_part).ratio()
    return similarity >= (threshold / 100)


# honestly I need to learn defensive programming and the way of the "make it actually not fucking broken"
# you know, organizing things in a tight way, well-structured, lots of asserts, the nasa 10 rules (insofar as they apply to an interpreted language like python and to a general case not space probes)

###################### NOTE -- REWARD FUNCTIONS #######################################

## ... reward funcs go here ...
# note to self: if anything is an autofail, then the reward should all be within that function and if it fails then that one returns 0 — we don't want any logic like that to be in the actual function-calling code of the reward handler.


@register_reward_function("gsm8k_correctness")
def correctness_reward_func(
    tagname: str = "answer",
):  # the interface for the outer is just whatever kwargs are needed from the config to configure the inner function. The interface for the inner is prompt, completions, and all the kwargs that are in each object in the dataset itself.
    def inner(
        prompt, completions, answer, **kwargs
    ) -> (
        float
    ):  # NOTE that technically the augmentoolkit thing will, allow for any format of input data... wait no that is not the case because the expected length is determined by the hardcoded prompt key. But that can be adjusted.
        """Reward function for answer correctness."""
        response = completions[0]["content"]
        q = prompt[0][-1]["content"]
        extracted_response = extract_xml_answer(response, tagname=tagname)
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{response}",
            f"\nExtracted:\n{extracted_response}",
        )
        return 2.0 if extracted_response == answer else 0.0

    return inner


@register_reward_function("gsm8k_integerresponse")
def int_reward_func(tagname: str = "answer"):
    def inner(completions, **kwargs) -> float:
        """Reward function for integer responses."""
        response = completions[0]["content"]
        extracted_response = extract_xml_answer(response, tagname=tagname)
        return 0.5 if extracted_response.isdigit() else 0.0

    return inner


@register_reward_function("generic_llm_reward")
def llm_reward_func(
    system_prompt_path,
    score_types,
    temperature=0.7,
    top_p=0.9,
    format_score=0.1,
    max_tokens=5000,
    eval_llm_name=None,
    eval_llm_base_url=None,
    eval_llm_api_key=None,
    eval_llm_mode=None,
    cot_end=None,
    cot_start=None,
    **kwargs,
):  # don't worry we can make prompts relative to this script (same as doing it in the main grpo.py since it's in the same folder) and we can load the prompts earlier on in this function
    assert (
        eval_llm_name is not None
    ), "eval_llm_name must be provided for generic_llm_reward"
    assert (
        eval_llm_base_url is not None
    ), "eval_llm_base_url must be provided for generic_llm_reward"
    assert (
        eval_llm_api_key is not None
    ), "eval_llm_api_key must be provided for generic_llm_reward"
    assert (
        eval_llm_mode is not None
    ), "eval_llm_mode must be provided for generic_llm_reward"
    engine_wrapper = EngineWrapper(
        model=eval_llm_name,
        base_url=eval_llm_base_url,
        api_key=eval_llm_api_key,
        mode=eval_llm_mode,
    )
    system_prompt_path = make_relative_to_self(system_prompt_path)
    # load the prompt (yaml file)
    with open(system_prompt_path, "r") as f:
        prompt_data = yaml.safe_load(
            f
        )  # list of message objects. This is what we will use.

    async def inner(
        prompts, completions, **kwargs
    ) -> float:  # unsloth converts "prompt" to "prompts"
        """Reward function that uses an LLM to evaluate responses."""

        conversation_history = ""
        for message in prompts:  # All messages in that prompt
            role = message["role"]
            content = message["content"]
            conversation_history += f"{role.upper()}: {content}\n\n"

        # Check format first
        if not validate_cot_format(
            completions[-1]["content"], cot_end=cot_end, cot_start=cot_start
        ):
            print("Input:\n==========")
            print(prompts[-1]["content"])
            print("===================\n\n\n")
            print("Response\n============")
            print(completions)
            print("===============\n\n\n")
            print("Response failed format check - returning 0")
            return 0.0  # Failed format check gets 0

        if check_thought_answer_similarity(
            completions[-1]["content"], cot_end=cot_end, cot_start=cot_start
        ):
            return 0.1  # Penalize very similar thoughts to answers.

        # Call LLM asynchronously with retries
        max_retries = 2

        input_text = prompts[-1]["content"]
        output_text = completions[-1][
            "content"
        ]  # it's a list of one item, but still a list
        prompt_formatted = [
            {
                "role": item["role"],
                "content": safe_format(
                    item["content"],
                    input_text=input_text,
                    output_text=output_text,
                    **kwargs,
                ),
            }
            for item in prompt_data
        ]

        print("DEBUG: PROMPT FORMATTED")
        print(prompt_formatted)
        # also by this point, it will have converted the thing from sharegpt to oai api format.
        for attempt in range(max_retries + 1):
            print(f"Attempt {attempt + 1}/{max_retries + 1} to get LLM rating")
            try:
                completion, timed_out = await engine_wrapper.submit_chat(
                    messages=prompt_formatted,
                    sampling_params={
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                    },
                )
            except Exception as e:
                print(f"Error in LLM evaluation: {str(e)}")
                # Exponential backoff delay between retries
                if attempt < max_retries:  # Don't sleep on last attempt
                    delay = 2**attempt  # 1, 2, 4, 8, etc seconds
                    print(f"Sleeping for {delay} seconds before retry...")
                    await asyncio.sleep(delay)
                continue

            if timed_out:
                print(f"LLM evaluation timed out on attempt {attempt + 1}")
                continue  # we want to keep trying until our retries are used up, then go and handle any failure the same way

            print(
                "========================== ALL INFORMATION ====================================="
            )
            print(f"Ground truth:\n\n\n--------{kwargs['answer']}\n---------")
            print(f"Input:\n\n\n---------{input_text}\n------------")
            print(f"Response:\n\n\n---------{output_text}\n------------")
            print(
                f"Raw LLM completion:\n\n\n------------\n{completion}\n-------------------"
            )
            print("========================================")

            try:
                scores = {}
                for score_item in score_types:  # score_types is a list of dicts
                    ans = extract_xml_answer(
                        text=completion, tagname=score_item["name"]
                    )
                    ans = float(ans)
                    if score_item["type"] == "autofail":
                        if ans == 0:
                            return 0
                        else:
                            scores[score_item["name"]] = ans
                    if score_item["type"] == "score":
                        if ans < score_item["min"]:
                            continue  # retry eval
                        elif ans > score_item["max"]:
                            continue
                        scores[score_item["name"]] = ans
                    # "score" is default, just return ans

                final_score = 0
                for key, value in scores.items():
                    final_score += value

                return final_score

            except Exception as e:
                print(f"Error in LLM reward function: {str(e)}")
                return format_score
        return format_score

    return inner
