# GRPO (Experimental)

This pipeline represents a significant departure from Augmentoolkit's standard data generation focus. Instead, it provides an experimental framework for **fine-tuning LLMs using Reinforcement Learning (RL)**, specifically employing the **GRPO (Generative Reward Powered Optimization)** algorithm. It allows you to align a base model towards desired behaviors by defining custom reward functions, including complex evaluations performed by other LLMs.

**WARNING:** This pipeline is **experimental and has sharp edges.** Basically, it's in beta. It's cool and runs, but there are no good examples yet and getting started is tricker.

**Linux only. Rent a Linux instance for cheap on Runpod if you want 

Note:
1.  **This pipeline performs model training, not just data generation.** You need a machine with a capable GPU (suitable for LoRA training) and a Linux environment so that vLLM works. You will also need to install vllm in the virtual environment used to run this project.
2.  **Customization requires Python coding.** Adding new reward functions involves editing `generation/core_pipelines/do_grpo_rl_with_a_prompt/reward_functions.py`. New LLM-based reward functions also require creating corresponding prompt files.
3.  **Defaults and best practices are still evolving.** This pipeline is provided for advanced users and researchers comfortable with RL concepts, training, and potential instability. The interface might change. Nothing is guaranteed. But if you have a cool idea for a reward function and just want to slot it in, this sure as hell beats hacking the Unsloth example at convenience.
4. **No Auto-Resume:** This pipeline due to complexity breaks from the common Augmentoolkit mainstay. 

The core idea is to generate multiple candidate responses for prompts drawn from your input dataset(s), evaluate these responses using your defined reward functions, and then use these reward signals to optimize the model's LoRA weights via GRPO.

## Unique Config Options

This pipeline's configuration (`config.yaml` or similar) is structured differently than other Augmentoolkit pipelines. It lacks standard sections like `api` or `path` at the top level. Instead, settings are grouped by function:

**`training_parameters` section:**
*   Contains hyperparameters for the Unsloth/TRL `GRPOConfig`, such as:
    *   `learning_rate`, `adam_beta1`, `adam_beta2`, `weight_decay`, `warmup_ratio`, `lr_scheduler_type`, `optim`: Standard training optimiser settings.
    *   `per_device_train_batch_size`, `gradient_accumulation_steps`: Control batching.
    *   `num_generations`: (GRPO specific) How many candidate completions to generate for each prompt during training.
    *   `max_prompt_length`, `max_completion_length`: Sequence length limits.
    *   `max_steps`: Total training steps.
    *   `save_steps`, `save_total_limit`, `save_strategy`: Checkpointing configuration.
    *   `max_grad_norm`: Gradient clipping.
    *   `output_dir`: Where training outputs (checkpoints, logs) are saved.
    *   `chat_template`: A Jinja2 string defining the chat format expected by the model being trained.
    *   `model_stop_sequences`: List of strings that should terminate generation for the model being trained.

**`model` section:**
*   `base_model_name`: Hugging Face identifier for the base model to fine-tune.
*   `lora_rank`, `lora_alpha`: LoRA adapter configuration.
*   `max_seq_length`: Maximum sequence length for the model.
*   `gpu_memory_utilization`: Target GPU memory usage (passed to Unsloth).

**`reward` section:**
*   `high_scoring_save_path`: Path to a JSONL file where high-scoring prompt/completion pairs will be saved.
*   `score_save_threshold`: Float threshold. If a completion's combined reward exceeds this, it's saved to the file above.
*   `combination_function`: (String) Name of the function used to combine multiple reward scores for a single completion into one scalar value. Defined in `post_processors.py` (e.g., `"sum"`, `"mean"`).
*   `scaling_function`: (String) Name of the function used to scale the *batch* of combined rewards before they are used for optimization. Defined in `post_processors.py` (e.g., `"identity"`).

**`datasets` section:** *(List of dataset configurations)*
*   Each item in this list defines an input dataset and how it should be processed and rewarded.
    *   `path`: Path to the dataset file (JSON, JSONL, Parquet accepted).
    *   `percentage`: Float (0.0-1.0) determining what fraction of the `dataset_settings.total_rows` should be sampled from this dataset.
    *   `system_prompt`: (String) A system prompt to be prepended to conversations from this dataset during processing.
    *   `single_turn_ratio`: Float (0.0-1.0). Probability of truncating multi-turn conversations after the first turn.
    *   `force_single_turn`: (Boolean) If `True`, always truncate after the first turn.
    *   `reward_funcs`: (List of Dicts) Defines the reward functions applied to completions generated from prompts from *this* dataset.
        *   Each dict specifies a reward function:
            *   `name`: (String) The name matching a function registered with `@register_reward_function` in `reward_functions.py`.
            *   *Any other keys*: These are passed as keyword arguments (**kwargs) to the reward function's *initializer* (the outer function that returns the actual reward function). This is how you configure specific reward functions.
            *   **LLM Reward Kwargs:** For LLM-based rewards (like the example `generic_llm_reward`), common kwargs include:
                *   `eval_llm_name`, `eval_llm_base_url`, `eval_llm_api_key`, `eval_llm_mode`: API details for the *evaluation* LLM.
                *   `system_prompt_path`: Path to the YAML file containing the prompt for the evaluation LLM.
                *   `score_types`: A list defining the expected structure of the evaluation LLM's output. Each item specifies a category (`name`), whether it's an `autofail` (score=0 immediately fails the evaluation) or a `score` (with `min`/`max` range), guiding the parsing of the LLM's response.
                *   Other kwargs like `cot_start`, `cot_end`, `temperature`, `top_p` etc. specific to the reward function's logic.

**`dataset_settings` section:**
*   `total_rows`: The total number of examples to sample *across all datasets* for the training run. Use this to control the number of optimizer steps, training time, overfitting etc.

## Model Requirements

*   **Base Model (`model.base_model_name`):** The model you intend to fine-tune using GRPO.
*   **Evaluation LLM(s) (`datasets[*].reward_funcs[*].eval_llm_*`):** (Optional) Any LLM(s) used by your LLM-based reward functions. These require separate API configuration within the `reward_funcs` list and are used for *inference only* during the reward calculation phase. Whether these are reasoning models or not, depends on the prompts you write or use. I tend to prefer reasoning models for grading.

## Input Files

The pipeline takes one or more input datasets specified in the `datasets` list in the config. Supported formats are JSON, JSONL, and Parquet. Datasets are expected to be in ShareGPT format, containing a `"conversations"` key with a list of turns (`"from"`, `"value"`). The LLM will be made to complete assistant messages from this dataset and the responses will then be graded.

## Output Files Guide

Unlike other pipelines, the main output is a trained model artifact:

*   **Trained LoRA Adapter:** Saved within the `training_parameters.output_dir`. Checkpoints are saved according to `save_strategy` and `save_steps`. The final adapter can be loaded and merged with the base model for inference.
*   `high_scoring_examples.jsonl`: (Optional) If `reward.score_save_threshold` is set, this file in the run directory will contain prompt/completion pairs that achieved high reward scores, formatted for potential use as preference data.

## Purpose of Overall Pipeline and Use Cases

The GRPO pipeline aims to democratize Reinforcement Learning for LLM alignment. By allowing users to define arbitrary reward functions – combining programmatic checks (e.g., checking for keywords, format validation) with powerful LLM-based evaluations (e.g., rating for creativity, factuality, helpfulness, non-sycophancy based on a rubric) – it enables optimization for complex, nuanced behaviors that are difficult to capture with SFT alone. And it opens this up easily for hobbyists to customize for their own needs without modifying lower-level code. You just write your functions in `reward_functions.py` and slot them in using the config file. GRPO is great for adding a "spark" of really impressive intelligence to models above a certain size threshold, and being able to optimize an objective is sometimes nicer to work with than trying to implicitly train on that objective with SFT and hope it generalizes in the desired way.

**Use Cases:**

*   **Aligning Models to Custom Rubrics:** Training models to adhere to specific stylistic guidelines, safety protocols, or quality standards defined via LLM evaluation prompts.
* **Empowering Researchers, Hobbyists, and Easy Production Customization:** With GRPO being easier to get started with, researchers can iterate faster, hobbyists can get started easier, and production applications of RL can be tested without the upfront technical investment of finding and hacking an existing lower-level RL codebase to use your custom thing.
* **Beyond GSM8k hillclimbing:** come on, guys, we can be more interesting than this *again and again and again*.
*   **Optimizing for Complex Metrics:** Rewarding behaviors like creativity, emotional expression, factual accuracy against a knowledge base, or coding efficiency.
*   **Beyond Preference Tuning:** Moving beyond simple pairwise comparisons (like DPO) to optimize against richer, multi-faceted reward signals.

Remember, this is an **experimental pipeline** requiring at least some technical expertise, computational resources (GPU, you can rent however), and willingness to debug and potentially modify the underlying code.

**REMINDER:** you must install vLLM on the virtual environment you use to run Augmentoolkit in order for this pipeline to work.

## Example Setup on a Fresh Ubuntu Runpod

Here's a list of commands that can help you get the GRPO pipeline working on a freshly created Ubuntu Runpod docker image. Note that you will need to replace placeholder values for IP addresses and ports with your specific Runpod details.

```bash
# On the new linux machine, run:
apt-get update && apt-get install -y tmux nano # Added -y to apt-get install
tmux
git config --global credential.helper store
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash linux.sh # Allow this to error; it installs some dependencies

# Replace YOUR_RUNPOD_PORT and YOUR_RUNPOD_IP with your actual Runpod details
# Also, ensure the source paths match your local file structure (like, the things that copy over your config file for instance)
# (run these next three from the root of your augmentoolkit project on your machine)
scp -P YOUR_RUNPOD_PORT -r ./generation/core_pipelines/do_grpo_rl_with_a_prompt/hidden_YOUR_CONFIG root@YOUR_RUNPOD_IP:/augmentoolkit/generation/core_pipelines/do_grpo_rl_with_a_prompt/
scp -P YOUR_RUNPOD_PORT -r ./generation/core_pipelines/do_grpo_rl_with_a_prompt/prompts/evaluation_prompts/hidden_YOUR_PROMPT.yaml root@YOUR_RUNPOD_IP:/augmentoolkit/generation/core_pipelines/do_grpo_rl_with_a_prompt/prompts/evaluation_prompts/hidden_YOUR_PROMPT.yaml
scp -P YOUR_RUNPOD_PORT -r ./inputs/hidden_YOUR_INPUT* root@YOUR_RUNPOD_IP:/augmentoolkit/inputs

# Back on the new linux machine, Install GRPO specific requirements:
source .venv/bin/activate
uv pip install -r generation/core_pipelines/do_grpo_rl_with_a_prompt/requirements.txt

# Create/Edit your super_config.yaml
nano super_config.yaml
```

Inside `super_config.yaml`, you'll need to define the pipeline order. For example:

```yaml
pipeline_order:
  - node: grpo-rl-pipeline
    config: grpo-rl-folder:your_grpo_config.yaml # Replace 'your_grpo_config.yaml' with the actual name of your GRPO config file
```

Then, log in to Hugging Face:
```bash
huggingface-cli login
```

Finally, run Augmentoolkit:
```bash
python run_augmentoolkit.py
```

Once it is finished training, login to your Hugging Face account and your weights and biases.

```
huggingface-cli login
wandb login
```

Run the wandb sync command it tells you to run. Then do, from the augmentoolkit/ dir:
```
huggingface-cli upload YourHFUsername/YourDesiredHFRepoWhereYouWantYourModelFiles outputs/
```

Then use the command line merge utility

```
python  cli_utils/merge_lora.py --base-model [the model you made the lora on top of] --adapter-model [the path to the checkpoint that you want to use (You can check weights and biases to see which is probably the best, usually 300 is pretty good)] --save-path [folder you want to save the new merged model to]


#### This is an unexplored space.

We'll need to figure out RL together, as a community effort, I think. If you either figure something out, or if you have unanswered questions, please head over to the [Discord](https://discord.gg/s6PBfsaVzu)!