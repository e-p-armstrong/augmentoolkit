# Complete Factual Generation

This is the primary **composition pipeline** in Augmentoolkit, designed to be the most straightforward way to generate the full suite of datasets needed to create a robust **domain-expert LLM**. It orchestrates several other specialized pipelines, handling everything from initial document processing to generating diverse pretraining and supervised fine-tuning (SFT) data, culminating in ready-to-use Axolotl training configuration files.

Running this single pipeline provides:

1.  **Cleaned Text:** Processes input documents, including cleaning PDFs.
2.  **Rich Pretraining Data:** Generates varied representations and inferred facts from the cleaned text using the [Representation Variation pipeline](./representation_variation.md).
3.  **Diverse SFT Data:** Creates factual question-answering datasets using multiple prompt styles (open-ended, negative, hallucination-focused, etc.) via the [Multi-Source Recall pipeline](./multi_source_facts.md), RAG-focused training data via the [RAG Data pipeline](./rag_data.md), and self-correction examples via the [Correction Data pipeline](./corrections.md).
4.  **Balanced Training Mix:** Combines the generated domain SFT data with specified generic SFT datasets, automatically balancing the token counts.
5.  **Training Configuration:** Produces Axolotl YAML configuration files for both the continued pretraining and the final supervised fine-tuning steps.

Essentially, you provide documents and configure API keys/models, and this pipeline outputs everything needed to train your expert model.

#### The common question: Why Mistral 7b as the default base model? Why something so old?

Good question! Newer models have succumbed to a bit of a metagame of training on a lot of data. Take llama 3.1 8b, a newer model in the same weight class: it is trained on about 15 trillion tokens (!). This is good for benchmarks, but often makes these models bad at being trained for new usecases -- llama, for instance, is quite set in its ways and is tough to teach new domains to. It does not pick up new knowledge as well as Mistral 7b. I believe this might be a case of "Catastrophic Overtraining" (https://arxiv.org/abs/2503.19206). Ironically, Meta AI's focus on using scale as a crutch for performance undermines them as a player in their other focus, open source.

Also, it's much cheaper to tune Mistral 7b than basically anything else, making it good for experimentation (for me), good for end users (because you don't have to spend as much money to get results). And inference with it is cheaper too. The good thing about specialist models is that you can get away with small sizes because you are directly tuning on the same task you'll have at inference time; you aren't relying on emergent properties of the model to carry your task, you're training on the task and letting the optimization function guarantee a result close to the training data.

So basically: we tune on Mistral because it's better, faster, and cheaper. And the license is nicer too.

## Config Structure

This composition's configuration (`config.yaml`) aggregates settings for the underlying pipelines it calls, plus some overall orchestration settings. Sections specific to this composition are detailed first, followed by sections configuring the sub-pipelines.

### Composition-Specific Config Sections

**`path` section:**
*   `input_dirs`: (List of Dicts) Defines the source document locations and per-directory settings.
    *   `path`: Path to the input directory.
    *   `variation_generation_counts`: How many variations the Representation Variation pipeline should generate for text from this directory. **This is an important setting.** It is one of the two settings (the other being `number_of_factual_sft_generations_to_do`) which you should use if your dataset is too small and therefore your LLM has too few optimizer steps. Increase this to make more data. You can also increase this for specific folders in your dataset to weight them higher over the others. See [this quick explanation](../README.md#note-for-when-you-start-training) for a crash course on optimizer step management.
    *   `final_system_prompt_additional_context`: (String, Optional) Text to prepend to the system prompt for SFT data generated from this directory's content.
    *   `factual_gen_subset_size_per_way`: How many chunks from this directory to use for *each* factual SFT generation type defined in `factual_sft`.
    *   `factual_gen_use_subset`: Whether to apply the subset size above.
    *   `rag_subset_size`, `rag_use_subset`: Subset settings specifically for the RAG Data pipeline for this input directory.
    *   `correction_subset_size`, `correction_use_subset`: Subset settings specifically for the Correction Data pipeline for this input directory.
*   `output_dir`: The main directory where all intermediate pipeline outputs, final datasets, and training configs will be saved.
*   `huggingface_cache_dir`: Local directory for caching downloaded Hugging Face datasets (used for generic SFT data).

**`system` section:**
*   `number_of_factual_sft_generations_to_do`: (Integer) This determines how many different factual SFT datasets we make (by running the same data through the same piplines a second, third, fourth, etc... time). **This is an important setting.** It is one of the two settings (the other being `variation_generation_counts`) which you should use if your dataset is too small and therefore your LLM has too few optimizer steps. Increase this to make a lot more data. See [this quick explanation](../README.md#note-for-when-you-start-training) for a crash course on optimizer step management.
*   `completion_mode`: (Boolean) Passed down to most sub-pipelines. `False` (chat mode) is standard.
*   `concurrency_limit`: (Integer) Maximum concurrent API calls *across all sub-pipelines*.
*   `use_stop`: (Boolean) Passed down to most sub-pipelines. Whether to use stop tokens.
*   `subset_size`: (Integer) Default subset size if a sub-pipeline's specific `use_subset` is True but its size isn't specified.
*   `use_subset`: (Boolean) Default value for sub-pipelines' `use_subset` flags if not specified per-pipeline/per-directory.
*   `what_percent_of_sft_is_pretrain`: (Float, Optional) Target percentage of the final SFT dataset (by tokens) that should consist of pretraining data. Overridden by `num_tokens_pretraining_in_sft` if both are present.
*   `num_tokens_pretraining_in_sft`: (Integer, Optional) Target absolute number of tokens from the pretraining data to include in the final SFT dataset. Takes precedence over the percentage.
*   `shared_instruction`: (String) A system prompt prepended to *all* SFT data (domain and generic) before final processing. Supports `{context_uppercase}`, `{context_lowercase}`, and `{context}` placeholders which are filled by `dataset_context`.

**`dataset_context` field:**
*   (String) A short phrase describing the domain of your input documents (e.g., "Marine Biology", "Quantum Physics"). Used to fill placeholders in system prompts.

**`final_datasaving_settings` section:**
*   `template`: (String) Specifies the chat template for formatting the final SFT data into completion format. Can be a preset (`"atk"`, `"chatml"`) or a full Jinja2 template string.
*   `template_kwargs`: (Dict) Keyword arguments passed to the Jinja2 template during rendering (if a custom template string is used).
*   `generic_dataset_paths`: (List of Dicts) Specifies generic SFT datasets to download from Hugging Face Hub and mix with the generated domain data.
    *   `path`: Hugging Face dataset identifier (e.g., `teknium/OpenHermes-2.5`).
    *   `context_to_add`: (String, Optional) Text to add to the system prompt or user prompt for this generic dataset.
    *   `context_to_add_type`: (String: `"system"`, `"human"`, `"none"`) Where to add the `context_to_add`.
*   `generic_dataset_percentages`: (List of Integers) Corresponding percentages defining the target token proportion for each generic dataset in the final mix (relative to the total domain SFT token count).
*   `max_samples_per_dataset`: (Integer) Maximum number of samples to download from each generic Hugging Face dataset.
*   `minimum_generic_sft`: (Integer) Ensure at least this many tokens of generic SFT data are included, even if percentages calculate less.

**`model_training` section:**
*   `base_model`: Hugging Face identifier for the base model to be used in the generated Axolotl configs.
*   `pretrain_hub_model_id`: Target Hugging Face Hub path for pushing the *pretraining* LoRA adapter.
*   `pretrain_hub_strategy`: Hub upload strategy for pretraining checkpoints (e.g., `"all_checkpoints"`).
*   `finetune_hub_model_id`: Target Hugging Face Hub path for pushing the *SFT* LoRA adapter.
*   `finetune_hub_strategy`: Hub upload strategy for SFT checkpoints.
*   `wandb_project`: Weights & Biases project name for logging in the Axolotl configs.
*   `is_mistral_derived_model`: (Boolean) Flag for Axolotl config generation specific to Mistral models.
*   `other_pretrain_kwargs`, `other_finetune_kwargs`: (Dicts) Allows adding arbitrary extra key-value pairs directly into the generated pretraining and SFT Axolotl configuration files, respectively.

### Sub-Pipeline Config Sections

**`pdf_cleaning` section:**
*   Configures the [PDF Cleaning Pipeline (TODO: Link)](./pdf_cleaning.md).
*   **Available Options:** `pdf_cleaning_chunk_size`, API details (`small_model`, `large_model`, modes, URLs, keys), `pdf_cleaning_use_stop`, cost estimation args.
*   **Missing/Hardcoded:** `input_dir`, `output_dir`, `prompts`, `default_prompts`, `concurrency_limit`, `completion_mode`, `use_subset`, `subset_size` are handled by the composition pipeline.

**`representation_variation` section:**
*   Configures the [Representation Variation Pipeline](./representation_variation.md).
*   **Available Options:** `representation_variation_chunk_size`, API details, `representation_variation_use_stop`, `dataset_context`, `code_variation_functions`, cost estimation args.
*   **Missing/Hardcoded:** `input_dir`, `output_dir`, `prompts`, `default_prompts`, `concurrency_limit`, `completion_mode`, `use_subset`, `subset_size`, `variation_generator_count` (set per-input-dir), `include_context_in_dataset` (hardcoded `True`), `make_inferred_facts` (run once with `False`, once with `True`).

**`factual_sft` section:**
*   Defines *multiple* configurations for running the [Multi-Source Recall Pipeline](./multi_source_facts.md).
*   Each key (e.g., `openended`, `negative`) represents a separate run with specific settings.
*   **Available Options (per key):** `prompts`, `default_prompts`, `single_turn`, `skip_question_check`, `skip_answer_relevancy_check`, `skip_answer_accuracy_check`, `skip_repair_qa_tuples`, `multi_source` (should typically be `True`).
*   **Missing/Hardcoded:** All other arguments (API, paths, system settings) are taken from the `factual_sft_settings` section or the main composition config.

**`factual_sft_settings` section:**
*   Provides the *shared* settings for all factual SFT runs defined in the `factual_sft` section.
*   Configures the [Multi-Source Recall Pipeline](./multi_source_facts.md).
*   **Available Options:** `factual_use_stop`, `factual_chunk_size`, `factual_completion_mode`, API details, cost estimation args, `final_assistant_prompts_no_rag`, `items_per_conversation`, `combine_sharegpt_target_pairs`.
*   **Missing/Hardcoded:** `input_dir`, `output_dir`, `prompts`, `default_prompts`, `concurrency_limit`, `use_subset`, `subset_size`, `final_assistant_prompts_rag`, `rag_failure_percentage` (RAG data generation is handled by the dedicated `rag_data` step).

**`rag_data` section:**
*   Configures the [RAG Data Pipeline](./rag_data.md).
*   **Available Options:** `rag_failure_percentage`, `rag_max_chunks`, formatting (`user_format`, `system_format`, `assistant_format`, `bos`), `final_assistant_prompts`, `num_items_per_group`, API details, cost estimation args, `rag_use_stop`.
*   **Missing/Hardcoded:** `input_dir`, `output_dir`, `prompts`, `default_prompts`, `concurrency_limit`, `completion_mode`, `use_subset`, `subset_size`, `chunk_size` (uses `factual_chunk_size`), `skip_filter_chunks` (uses default `False`).

**`correction_pipeline` section:**
*   Configures the [Correction Data Pipeline](./corrections.md).
*   **Available Options:** `correction_chunk_size`, API details, cost estimation args, `correction_prompt_template`, `correction_use_stop`, `correction_completion_mode`.
*   **Missing/Hardcoded:** `input_dir`, `output_dir`, `prompts`, `default_prompts`, `concurrency_limit`, `use_subset`, `subset_size`.

## Model Requirements

This pipeline requires multiple models:
*   **PDF Cleaning:** Typically a reasoning model (`small_model`) and a strong instruction model (`large_model`).
*   **Representation Variation:** Reasoning models for both small and large if doing inferred facts, otherwise instruction models.
*   **Factual SFT (Multi-Source Recall):** Strong instruction models for both small (filtering) and large (QA generation).
*   **RAG Data:** Strong instruction models for both small (filtering) and large (QA generation with context).
*   **Correction Data:** Instruction model for filtering (`small_model`), reasoning model recommended for generation (`large_model`).

Consult the linked individual pipeline docs for specifics, but generally, you'll need access to both capable instruction-following models (like Llama 3.x, DeepSeek V2/V3) and reasoning models (like Qwen/QwQ, R1 distills, DeepSeek-R1). **Alternatively**, the custom Augmentoolkit datagen model (7b) has been trained to run all steps by itself. API keys and base URLs need to be configured correctly for each pipeline section. Sensible+functional defaults are provided for base urls + model names, but if you're not generating locally you *will* have to provide API keys.

## Input Files

Standard document formats (`.txt`, `.md`, `.pdf`, `.docx`, `.jsonl` with "text" key) placed within the directories specified in `path.input_dirs[*].path`.

Multiple input dirs allows finer control over how each source of data is treated by the pipelines. If you want to emphasize one group of data that is otherwise small, you could increase its `variation_generation_counts` for instance. You can also append different things to the system prompts of instruct data made from different sources using the `final_system_prompt_additional_context`.

## Output Files Guide

The composition generates numerous intermediate folders (one for each sub-pipeline run). The most important final outputs are within the main `output_dir`:

*   **`pretraining_run/` Folder:**
    *   `*.jsonl`: Contains the combined pretraining data (original chunks, representation variations, inferred facts) generated by the Representation Variation steps for each input directory.
    *   `axolotl_pretraining_config.yaml`: The generated Axolotl config file for running continued pretraining on this data.
*   **`sft_run/` Folder:**
    *   `combined_factual_data/`: Contains the domain SFT data generated by the Multi-Source Recall steps, combined and ready for completionification.
    *   `factual_sft_completion/`: The domain SFT data after being formatted into completion records using the specified template.
    *   `generic_sft_completion/`: The downloaded and subsetted generic SFT data, also formatted into completion records.
    *   `axolotl_correction_conversations_*.json`: Correction data (segmented ShareGPT format).
    *   `axolotl_rag_conversations_*.jsonl`: RAG data (segmented ShareGPT format).
    *   `pretraining_subset_*.jsonl`: (Optional) A subset of the pretraining data, also formatted for completion, to be included in the SFT mix.
    *   `sft_training_config.yaml`: The generated Axolotl config file for running the final SFT phase using all the data in this directory.
*   **Intermediate Folders:** Directories named like `pdf_cleaning_<input_dir_name>`, `representation_variation_<input_dir_name>`, `factual_sft_<input_dir_name>_<way>_<index>`, etc., containing the intermediate outputs of each sub-pipeline run. Useful for debugging. Needed for resuming generations, so don't delete these unless you've already trained your model.

**Which files to use for training?**

1.  Use the **`pretraining_run/axolotl_pretraining_config.yaml`** file with Axolotl to perform continued pretraining on your base model using the data in `pretraining_run/`. Just copy the folder over to a machine with axolotl and a good enough GPU, and run `accelerate launch -m axolotl.cli.train axolotl_pretraining_config.yaml`
2.  Use the **`sft_run/sft_training_config.yaml`** file with Axolotl to perform supervised fine-tuning on the *result* of the pretraining step, using all the diverse SFT data compiled in `sft_run/`. Just copy the folder over to a machine with axolotl and a good enough GPU, and run `accelerate launch -m axolotl.cli.train sft_training_config.yaml`

## Purpose of Overall Pipeline and Use Cases

This pipeline automates the best-practice workflow for creating highly capable domain-expert LLMs using Augmentoolkit. It combines:

*   **Continued Pretraining:** Exposing the model to the domain's core information in varied formats.
*   **Multi-faceted SFT:** Training the model on:
    *   Recalling facts from multiple sources ([Multi-Source Recall](./multi_source_facts.md)).
    *   Handling diverse question types (open-ended, negative, vague, follow-up).
    *   Utilizing RAG context ([RAG Data](./rag_data.md)).
    *   Correcting its own mistakes ([Correction Data](./corrections.md)).
*   **Generic Data Balancing:** Ensuring the model retains general capabilities by mixing in processed generic data.

By running this single composition, users can go from raw documents to ready-to-train Axolotl configurations, significantly simplifying the process of building specialized LLMs.

**Use Cases:**

*   **The Primary Augmentoolkit Workflow:** This is the recommended pipeline for most users aiming to create a domain-expert model.
*   **Comprehensive Domain Adaptation:** Building models that not only know facts but also reason about them and handle diverse interactions within a specific domain.
*   **Reproducible Expert Model Creation:** Provides a standardized, configurable process for generating the necessary training data and configurations.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.