"""
Training script for fine-tuning LLMs using GRPO (Generative Reward Powered Optimization).
Adapted from Colab notebook to standalone Python script.

WARNING: THIS PIPELINE REQUIRES A MACHINE CAPABLE OF TRAINING LLMs. I.E., YOU WILL WANT A GPU.
"""

from difflib import SequenceMatcher
import hashlib
import os
import sys
import re
import traceback
import torch

os.environ["VLLM_USE_V1"] = "0"  # Add this before importing vllm/Unsloth
from datasets import load_dataset, Dataset, concatenate_datasets
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from generation.core_pipelines.do_grpo_rl_with_a_prompt.post_processors import (
    get_combination_function_from_name,
    get_scaling_function_from_name,
)
from generation.core_pipelines.do_grpo_rl_with_a_prompt.reward_functions import (
    get_func_for_str,
    get_funcs_for_str_list,
)
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from vllm import SamplingParams

PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig
from generation.core_pipelines.do_grpo_rl_with_a_prompt.grpo_trainer_subclass import (
    GRPOTrainerStopSequences,
)
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging
from datetime import datetime
import wandb
import threading

# Listen, here's what's required of you with this one:
# 1. take the existing code
# 2. rip out the h ardcoded sections and anything related to gsm8k,  make it all general and take things from the configs, but otherwise leave the logic intact
# 3. make the config a yaml not json
# 4. put all the reward stuff in a separate file and import it
# that's literally it.


# Initialize wandb
wandb.init(project="unsloth-grpo", mode="offline")

# though we may want to be saving all the outputs of this as a log file, which would make sense
# this does not really proceed in the normal way of a pipeline where we have a sequential set of steps which are all pipeline step classes
# instead the thing in and of itself is the model training; we just setup the model training and go
# there is also
# the data preparation
# and prompt choice
# data prep might want to be separate code that is then used here
# prompt choice is typical stuff, often used, we have the generation steps for that
# Thing is,
# hmm there's also the matter of custom reward functions that are not LLM-based.
# subclasses
# There's a reward function subclass say, and then llm reward funcs are a subclass of that, normal reward funcs are also just instances of it
# reward functions should be defined in a separate file, and then imported here
# and which ones we use should be specified in the config
#

# one thing that I have not yet sorted out in this refactor of the project is the question of dependencies. Each pipeline has its own deps. We want to be able to switch deps for each, and also, know what we have to install to run each pipeline, and know what we need when we import one.
# poetry? Alternatives?


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)

# ^^ goes in the config


class MultiDataset(Dataset):
    """Dataset class combining multiple datasets with deterministic sampling, truncation, and metadata injection."""

    def __init__(self, configs, total_rows, tokenizer, max_prompt_length):
        self.datasets = []
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        # Normalize percentages
        percentages = [config.get("percentage", 1.0) for config in configs]
        total_pct = sum(percentages)
        percentages = [p / total_pct for p in percentages]

        final_datasets = []

        for config, pct in zip(configs, percentages):
            dataset = load_general_dataset(config["path"])

            # Attach metadata
            dataset = dataset.map(
                lambda x: {
                    **x,
                    "reward_funcs": config["reward_funcs"],
                    "system_prompt": config["system_prompt"],
                    "single_turn_ratio": config["single_turn_ratio"],
                    "force_single_turn": config["force_single_turn"],
                    "path": config["path"],
                }
            )

            # Conversation format conversion
            dataset = dataset.map(
                lambda example: (
                    {
                        **example,
                        "conversations": [
                            {
                                "role": (
                                    "user"
                                    if msg["from"] == "human"
                                    else (
                                        "assistant"
                                        if msg["from"] == "gpt"
                                        else msg["from"]
                                    )
                                ),
                                "content": msg["value"],
                            }
                            for msg in example.get("conversations", [])
                        ],
                    }
                    if "conversations" in example
                    else example
                )
            )

            # Deterministic truncation and system prompt injection
            dataset = dataset.map(
                lambda conv: self._process_single_example(
                    conv,
                    force_single_turn=config["force_single_turn"],
                    single_turn_ratio=config["single_turn_ratio"],
                    system_prompt=config["system_prompt"],
                )
            )

            # remove "conversations" column
            dataset = dataset.remove_columns("conversations")

            # Filter out examples with empty prompts
            dataset = dataset.filter(lambda x: x.get("prompt", "") != "")

            # Deterministic sampling
            dataset = self._deterministic_sample(dataset, int(total_rows * pct))

            final_datasets.append(dataset)

        # Concatenate all datasets deterministically shuffled
        combined = concatenate_datasets(final_datasets)
        combined = self._deterministic_shuffle(combined)

        self.combined_dataset = combined

        # Validate final dataset
        print(f"\nFinal combined dataset size: {len(self.combined_dataset)}")
        if len(self.combined_dataset) > 0:
            first_example = self.combined_dataset[0]
            print(f"First combined example keys: {first_example.keys()}")

    def _process_single_example(
        self, conv, force_single_turn, single_turn_ratio, system_prompt
    ):
        messages = conv["conversations"]

        # Inject system prompt
        if system_prompt:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})

        # Deterministic truncation point calculation
        original_conv = json.dumps(messages, sort_keys=True) + system_prompt
        hash_int = int(hashlib.sha256(original_conv.encode()).hexdigest(), 16)
        max_hash = (1 << 256) - 1
        hash_ratio = hash_int / max_hash

        assistant_indices = [
            i for i, msg in enumerate(messages) if msg["role"] == "assistant"
        ]
        if not assistant_indices:
            return {"conversations": messages, "answer": ""}

        if force_single_turn or (hash_ratio < single_turn_ratio):
            cut_idx = assistant_indices[0]
        else:
            chosen_idx = hash_int % len(assistant_indices)
            cut_idx = assistant_indices[chosen_idx]

        truncated_conv = messages[:cut_idx]
        answer = messages[cut_idx]["content"]

        # Token-based truncation to fit max_prompt_length
        # Ensure tokenizer.chat_template is set if not already (though it should be by the time this is called)
        if (
            self.tokenizer.chat_template is None
            and self.tokenizer.default_chat_template is not None
        ):
            self.tokenizer.chat_template = self.tokenizer.default_chat_template

        if (
            not truncated_conv
        ):  # Handle cases where initial truncation results in empty list
            logging.warning(
                f"Initial turn-based truncation resulted in empty conversation for path: {conv.get('path', 'N/A')}"
            )
            current_token_ids = []
        else:
            current_token_ids = self.tokenizer.apply_chat_template(
                truncated_conv,
                add_generation_prompt=True,  # Consistent with trainer's generation step
                tokenize=True,
            )

        max_truncation_attempts = len(truncated_conv) + 5
        attempts = 0

        while (
            len(current_token_ids) > self.max_prompt_length
            and attempts < max_truncation_attempts
        ):
            attempts += 1
            if not truncated_conv:
                logging.warning(
                    "Truncated conversation became empty while trying to fit max_prompt_length."
                )
                break

            original_len_truncated_conv = len(truncated_conv)

            # Preserve system prompt if it's the first message
            if truncated_conv[0]["role"] == "system":
                if len(truncated_conv) > 1:
                    # Remove the message after the system prompt (oldest conversational turn)
                    removed_message = truncated_conv.pop(1)
                    logging.debug(
                        f"Removed message to shorten prompt (after system): {removed_message.get('role', 'N/A')} from {conv.get('path', 'N/A')}"
                    )
                else:
                    # Only system prompt is left and it's too long
                    logging.warning(
                        f"System prompt alone (approx {len(current_token_ids)} tokens) is too long for max_prompt_length={self.max_prompt_length} in {conv.get('path', 'N/A')}. "
                        f"Content snippet: {truncated_conv[0]['content'][:200]}"
                    )
                    break
            elif (
                len(truncated_conv) > 0
            ):  # No system prompt or system prompt already handled
                # Remove the oldest message
                removed_message = truncated_conv.pop(0)
                logging.debug(
                    f"Removed message to shorten prompt (no system or already handled): {removed_message.get('role', 'N/A')} from {conv.get('path', 'N/A')}"
                )
            else:  # Should be caught by `if not truncated_conv:`
                break

            if (
                len(truncated_conv) == original_len_truncated_conv
                and original_len_truncated_conv > 0
            ):
                # No message was removed, means we are stuck (e.g. single system prompt too long or truncated_conv became empty)
                if truncated_conv:  # check if still has content
                    logging.warning(
                        f"Could not shorten prompt further for {conv.get('path', 'N/A')}. Current length {len(current_token_ids)} tokens vs max {self.max_prompt_length}."
                    )
                break

            if not truncated_conv:  # If all messages were removed
                current_token_ids = []
                break

            current_token_ids = self.tokenizer.apply_chat_template(
                truncated_conv, add_generation_prompt=True, tokenize=True
            )

        if (
            attempts >= max_truncation_attempts
            and len(current_token_ids) > self.max_prompt_length
        ):
            logging.error(
                f"Failed to truncate prompt to {self.max_prompt_length} tokens after {max_truncation_attempts} attempts for {conv.get('path', 'N/A')}. Final length: {len(current_token_ids)}. Example prompt: {truncated_conv}"
            )
            # Potentially return an empty prompt or a specially marked error state if downstream can handle it.
            # For now, it will proceed with the overly long prompt, likely causing the original error.
            # Or, to be safer and prevent the error, make truncated_conv empty or raise an error.
            # Let's clear truncated_conv to prevent the error, though this loses the sample.
            truncated_conv = []  # This would avoid the error but lose data.
            answer = ""  # And its corresponding answer.
            # logging.error("Setting truncated_conv to empty to prevent downstream error.")

        return {
            "prompt": truncated_conv,  # we need to rename "conversations" to "prompt" so that it works with unsloth
            "answer": answer,
            "reward_funcs": conv["reward_funcs"],
            "path": conv["path"],
        }

    def _add_deterministic_hash_column(self, dataset):
        return dataset.map(
            lambda x: {
                "hash": hashlib.sha256(
                    json.dumps(x["prompt"], sort_keys=True).encode()
                ).hexdigest()
            }
        )

    def _deterministic_sample(self, dataset, n_samples):
        if len(dataset) <= n_samples:
            logging.warning(
                f"Dataset at {dataset[0]['path']} has fewer samples ({len(dataset)}) than requested ({n_samples}). Using all available samples."
            )
            return dataset

        dataset_with_hash = self._add_deterministic_hash_column(dataset)
        sorted_dataset = dataset_with_hash.sort("hash")
        sorted_dataset = sorted_dataset.remove_columns("hash")
        return sorted_dataset.select(range(n_samples))

    def _deterministic_shuffle(self, dataset):
        dataset_with_hash = self._add_deterministic_hash_column(dataset)
        shuffled_dataset = dataset_with_hash.sort("hash")
        shuffled_dataset = shuffled_dataset.remove_columns("hash")
        return shuffled_dataset

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]


# ^^ maybe this gets preserved? I don't want to have to rewrite EVERYTHING, but this can't be the easiest way???


def setup_environment():
    """Initialize environment and clear conflicting modules."""
    modules = list(sys.modules.keys())
    for x in modules:
        if "PIL" in x or "google" in x:
            sys.modules.pop(x)


def setup_model(
    base_model_name,
    max_seq_length=5000,
    lora_rank=64,
    lora_alpha=128,
    gpu_memory_utilization=0.7,
):
    """Initialize and configure the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        # load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    return model, tokenizer


# ^^ preserved

##################

# Dataset managing is either in this file or a separate components file that is generalized and imported here. Thing is, this will be RLLMF specific, so I am sympathetic to having it here.
# The structure's got to change though. A converter from sharegpt to oai api format is needed. And something which deterministically picks points in the sft to generate from. And in the config we will want the prompt to be customizable. Or do we have that in the prompts folder? (the prompt being used during RL). Do we have multiple? Do we specify a list of prompt prefixes added onto the damn thing inside the config? (Like a list of paths, not a list of strings or a single path inside the config)
# hmm that seems potentially promising and pretty flexible


# also we want an option, a post or preprocessor, or an input to the reward functions -- something which takes in the batch
# so taht we can do kalomaze-inspired things like scale the reward if the output is good across the whole batch
# the scores of other rewards will be good info to have.
# also the reward function manager may have to change? Like,


def load_jsonl_dataset(path: str) -> Dataset:
    """Load a dataset from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # Convert to Dataset
    return Dataset.from_list(data)


def load_general_dataset(path: str) -> Dataset:
    """Load a dataset from a file path, supporting JSONL, JSON, and Parquet formats."""
    # Get file extension
    ext = path.lower().split(".")[-1]

    if ext == "jsonl":
        return load_jsonl_dataset(path)
    elif ext == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, dict):
                data = [data]
            return Dataset.from_list(data)
    elif ext in ["parquet", "pq"]:
        return Dataset.from_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported formats are: jsonl, json, parquet"
        )


def save_high_score_response(
    prompt_messages, response_text, high_scoring_save_path, tokenizer
):  # sort of like an observer
    """Save high-scoring response to JSONL file with proper formatting."""
    try:

        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Add BOS/EOS tokens
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""

        # Create entry
        entry = {
            "segments": [
                {"label": False, "text": f"{bos}{formatted_prompt}"},
                {"label": True, "text": f"{response_text}{eos}"},
            ]
        }

        # Write to file with thread lock
        with open(high_scoring_save_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Saved high-scoring response to {high_scoring_save_path}")

    except Exception as e:
        print(f"Error saving high-score response: {str(e)}")


# all below here stays in the file. It won't all stay the same since we're changing how reward functions are classed and passed in and selected and work, and also the bits about the config will have to change too. However, the core logic will be there. Things like the base model etc. will get passed into the main function and drilled down typical ATK style.


def setup_trainer(
    model,
    tokenizer,
    dataset: MultiDataset,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=4000,
    max_completion_length=2500,
    max_steps=500,
    save_steps=100,
    save_total_limit=7,
    save_strategy="steps",
    max_grad_norm=0.1,
    output_dir="outputs",
    reward_combination_function=lambda rewards: sum(rewards),
    reward_scaling_function=lambda rewards: rewards,
    chat_template="chatml",
    lock=None,
    high_scoring_save_path=None,
    score_save_threshold=None,
    model_stop_sequences=["**Finished.**", "Human:"],
):  # reward scaling function takes a list of numbers and returns a list of numbers which may or may not be different/modified. reward combination function takes a list of numbers and combines them into a single one.
    """Configure and return the GRPO trainer."""
    training_args = GRPOConfig(
        use_vllm=True,
        # vllm_device="cuda:1", # NOTE this is SO THAT we DO GPU INFERENCE FOR THE REWARD GENERATION
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,  # was causing problems
        max_completion_length=max_completion_length,  # was causing REAL problems. Also I will need to pump up the context of the models I train because DAMN is it a bit too short with 5k. And single turn groups only up to 2 max I think. Well no.
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,
        max_grad_norm=max_grad_norm,
        report_to="wandb",
        output_dir=output_dir,
    )

    # Create a wrapper function that selects appropriate reward functions based on dataset type
    def reward_func_wrapper(
        *args, **kwargs
    ):  # I get that args is prompt, completions, and answer maybe too. But, what is, the kwargs? And why is args the answer?! How does it relate to teh dataset?

        # Log first 5000 chars of each arg
        for i, arg in enumerate(args):
            print(f"Arg {i}: {str(arg)[:5000]}")

        # Log first 5000 chars of each kwarg
        for key, value in kwargs.items():
            print(f"Kwarg {key}: {str(value)[:5000]}")

        # TODO if you want to have things operate over the whole batch, what needs to be done here is
        # 1. different function decorators on the reward functions, one for normal and one for batch
        # a modification to the get funcs for str list to first check if it is in the reward functions dict, and then check if it is in the batch reward functions dict
        # append it to the new reward_funcs_batch_list if it is in the batch dict
        # then, when we are iterating over the reward funcs, we iterate over the reward_funcs_batch_list first and do all those together
        # then we do reware funcs list as normal
        # this will allow us to support batching without having to rewrite all the logic
        # and while making it nice and easy to add new stuff to
        # also I think that (unless I am wrong) all it takes to do this right is to just... append each item to the same index in the all rewards list? Should do?!~
        # here's the thing
        # that doesn't work across multiple datasets
        # because the batch reward list would apply to all datasets regardless of origin
        # not ideal
        # so no

        expected_len = len(next(iter(kwargs.values())))

        # kwargs["engine_wrapper"] = [EngineWrapper() for _ in range(expected_len)]

        # Prepare containers for sync results and async tasks
        all_rewards = [None] * expected_len  # Final rewards list
        async_tasks = []  # List of tuples (async_task, output_idx, func_idx)

        for output_idx, r_func_configs in enumerate(kwargs["reward_funcs"]):
            specific_rewards = []

            for func_idx, f_config in enumerate(r_func_configs):
                f = get_func_for_str(f_config["name"])(
                    **f_config
                )  # does NOT create many many copies of the same function -- it points to the function in the dict in the reward_functions.py file.
                kwargs_at_idx = {
                    k: v[output_idx] if isinstance(v, list) else v
                    for k, v in kwargs.items()
                }
                try:
                    result = f(*args, **kwargs_at_idx)

                    if asyncio.iscoroutine(result):
                        # Accumulate async tasks with their indices for later reconstruction
                        async_tasks.append((result, output_idx, func_idx))
                    else:
                        if not isinstance(result, list):
                            raise ValueError(
                                "Invalid sync reward function output format; expected list."
                            )
                        if len(result) != expected_len:
                            raise ValueError(
                                f"Sync reward function {f.__name__} returned incorrect length: {len(result)} instead of {expected_len}"
                            )
                        specific_rewards.append(
                            result[output_idx]
                        )  # Only store the scalar at the correct idx
                except Exception as e:
                    print("Error in sync reward function!")
                    traceback.print_exc()
                    raise Exception(f"Error in reward function {f.__name__}: {str(e)}")

            # Temporarily store sync rewards; async rewards will be added later
            all_rewards[output_idx] = specific_rewards

        # Now handle all async tasks simultaneously
        if async_tasks:

            async def execute_async_tasks(tasks):
                coros = [task[0] for task in tasks]
                results = await asyncio.gather(*coros, return_exceptions=True)
                return results

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async_results = loop.run_until_complete(execute_async_tasks(async_tasks))

            # Rebuild async results into their proper positions
            for result, (task, output_idx, func_idx) in zip(async_results, async_tasks):
                if isinstance(result, Exception):
                    print("\n" + "!" * 80)
                    print(
                        f"WARNING: Async reward function at index {output_idx}, function {func_idx} failed! Defaulting to 0."
                    )
                    print("Error details:")
                    traceback.print_exception(
                        type(result), result, result.__traceback__
                    )
                    print("!" * 80 + "\n")
                    reward_value = 0.0
                else:
                    if not isinstance(result, (int, float)):
                        print(
                            f"WARNING: Async reward function at index {output_idx}, function {func_idx} returned non-numeric: {result}. Defaulting to 0."
                        )
                        reward_value = 0.0
                    else:
                        reward_value = result

                # Append async reward to the corresponding output_idx
                all_rewards[output_idx].append(reward_value)

        # Now combine each specific_reward list into a final scalar reward
        final_combined_rewards = []
        for idx, rewards in enumerate(all_rewards):
            combined_reward = reward_combination_function(rewards)
            kwargs_at_idx = {
                k: v[idx] if isinstance(v, list) else v for k, v in kwargs.items()
            }
            if (
                score_save_threshold is not None
                and combined_reward >= score_save_threshold
            ):  # if for some reason you're ignoring recommendations and not using a dataset with conversations, set the score save threshold immensely high.
                save_high_score_response(
                    prompt_messages=kwargs_at_idx["prompts"],
                    response_text=kwargs_at_idx["completions"],
                    high_scoring_save_path=os.path.join(
                        output_dir, high_scoring_save_path
                    ),
                    tokenizer=tokenizer,
                )
            final_combined_rewards.append(combined_reward)

        # Finally apply scaling across all combined rewards
        final_scaled_rewards = reward_scaling_function(final_combined_rewards)

        print(f"Final combined rewards: {final_scaled_rewards}")
        return final_scaled_rewards

    tokenizer.chat_template = (
        chat_template  # this is where the custom jinja2 string should apply
    )

    # Add dataset inspection before training
    print("\n===== DATASET INSPECTION =====")
    print(f"Dataset length: {len(dataset)}")

    # Inspect first 5 examples
    for i in range(min(5, len(dataset))):
        example = dataset[i]
        print(f"\nExample {i}:")
        print(f"Keys: {example.keys()}")

        if "prompt" in example:
            print(f"Prompt type: {type(example['prompt'])}")
            print(f"Prompt value: {example['prompt']}")

            # Test format_prompt on this example
            print("Testing format_prompt on this example:")
            try:
                formatted = tokenizer.apply_chat_template(example["prompt"])
                print(f"Format successful: {len(formatted) > 0}")
            except Exception as e:
                print(f"Format failed: {str(e)}")

        if "answer" in example:
            print(f"Answer type: {type(example['answer'])}")
            print(f"Answer value: {example['answer']}")

        if "reward_functions" in example:
            print(f"Metadata: {example['reward_functions']}")

    print("===== END DATASET INSPECTION =====\n")

    return GRPOTrainerStopSequences(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func_wrapper],
        args=training_args,
        train_dataset=dataset,
        stop=model_stop_sequences,
    )


# the one last thing I can think of is the need to convert things from sharegpt to oai format with role and content


def grpo_rl_pipeline(
    high_scoring_save_path,
    score_save_threshold,
    reward_combination_function,
    reward_scaling_function,
    datasets,
    base_model_name,
    max_seq_length,
    lora_rank,
    lora_alpha,
    gpu_memory_utilization,
    learning_rate,
    adam_beta1,
    adam_beta2,
    weight_decay,
    warmup_ratio,
    lr_scheduler_type,
    optim,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    num_generations,
    max_prompt_length,
    max_completion_length,
    max_steps,
    save_steps,
    save_total_limit,
    save_strategy,
    max_grad_norm,
    output_dir,
    chat_template,
    total_rows,
    **kwargs,
):
    """Main execution function."""
    print("Starting main execution")

    scalefunc = get_scaling_function_from_name(reward_scaling_function)
    combfunc = get_combination_function_from_name(reward_combination_function)

    # Initialize environment
    print("Setting up environment")
    setup_environment()

    # Setup model and tokenizer
    print("Initializing model and tokenizer")
    model, tokenizer = setup_model(
        base_model_name, max_seq_length, lora_rank, lora_alpha, gpu_memory_utilization
    )

    # Load dataset(s) from config
    print("Loading datasets from config")
    dataset = MultiDataset(datasets, total_rows, tokenizer, max_prompt_length)

    # Create full output directory path if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup and run training
    print("Setting up trainer")
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        high_scoring_save_path=high_scoring_save_path,
        score_save_threshold=score_save_threshold,
        reward_combination_function=combfunc,
        reward_scaling_function=scalefunc,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        chat_template=chat_template,
        lock=threading.Lock(),
    )

    print("Starting training")
    trainer.train()

    # Save the model
    print("Saving model")
    model.save_lora("grpo_saved_lora")


# I will have to d elete the git repo and it on github too so that the data source is not exposed
