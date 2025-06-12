# Generic Data Rephrase

This pipeline serves a crucial and somewhat unique purpose within the Augmentoolkit ecosystem. Its primary function is to take existing, standard instruction-following datasets (in ShareGPT JSONL format) and **retrofit them with synthetic thought processes.** It reads human prompts and existing assistant answers, then uses an LLM to generate a plausible Chain-of-Thought (CoT) reasoning that *could* have led to that original answer. The output is a new dataset where each assistant turn is prefaced by this generated thought process.

This process is **essential for training effective Augmentoolkit domain experts.** Early experiments showed that training models on domain-specific SFT data (with CoT) alongside generic SFT data (without CoT) led to poor generalization. Models learned to associate the task type (domain vs. generic) with the type of answer rather than learning the underlying knowledge robustly and in a general way. Because the model essentially learned two different tasks, rather than one unified one, models were bad at conversation when being asked domain questions, even though they had been trained on plenty of conversational data.

The Generic Data Rephrase pipeline solves this by **standardizing the response format across both domain and generic samples.** Now both of them include CoT, rather than only the domain SFT having CoT. By ensuring *both* domain-specific and generic SFT data include a thought process preceding the answer, the model is forced to learn the content and reasoning patterns more deeply, rather than relying on superficial format cues. This significantly improves generalization, instruction following, and the overall robustness of the final domain-expert model.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`system` section:**
*   `cot_preface`: (String) Text added *before* the generated thought process in the final output (e.g., `"Thought Process:"`).
*   `cot_suffix`: (String) Text added *after* the generated thought process but *before* the original assistant answer in the final output (e.g., `"\nAnswer:"`).

## Model Requirements

*   **`large_model`:** Used for the core step of generating the synthetic thought process (`thought_process_addition_pipeline`). This requires a model capable of understanding the request-response pair and a section of the chat history, and generating a plausible reasoning chain that connects them. **Do not use a reasoning model.** Reasoning models (like Qwen/QwQ or R1) can sometimes struggle with this task, as they may insert their *actual* reasoning process instead of generating a *synthetic* one appropriate for the existing answer. Models like DeepSeek-V3 or the custom Augmentoolkit datagen model have shown good results.
*   **`small_model`:** Not actively used in the default flow of this pipeline as there is no initial filtering step defined. The configuration exists for framework consistency and for making smoother the likely future integration of a validation step to this pipeline.

## Input Files

The pipeline reads `.jsonl` files from the `path.input_dir`. Each line in these files should be a JSON object representing a conversation, typically in the ShareGPT format (e.g., `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`). It processes the human/gpt pairs within these conversations. ShareGPT is the same format output by the rest of Augmentoolkit and is widely accepted by training frameworks.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `*.jsonl`: For each input JSONL file, a corresponding output JSONL file is created (with potentially an underscore appended if the name conflicts). Each line in the output file is a ShareGPT conversation object (`{"conversations": [...]}`). The key difference is that the assistant (`gpt`) turns have been modified to include the synthetic thought process, formatted with the `cot_preface` and `cot_suffix`.
*   `debug_outputs/revised_generics.yaml`: Debugging output showing the intermediate LLM generations for the thought processes.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs.
*   `meta_datagen/` (Folder, optional): Contains the meta-dataset if enabled.

**Which files to use for training?**

*   Use the **output `.jsonl` files** as part of your generic SFT data mix when training an Augmentoolkit domain expert. Combine this modified generic data with your domain-specific SFT data (which already includes CoT) to train the model.

## Purpose of Overall Pipeline and Use Cases

The Generic Data Rephrase pipeline's core purpose is structural alignment: **it makes generic instruction-following data look like domain-specific, CoT-driven data.** This prevents the LLM from taking shortcuts during training by learning format-based correlations, forcing it to integrate knowledge and reasoning capabilities more effectively.

While Augmentoolkit provides links to pre-processed generic datasets on its Hugging Face page, this pipeline allows users to:
1.  Process additional generic instruction datasets they may have.
2.  Increase the size or variety of their generic data mix.
3.  Experiment with adding CoT to datasets for specific domains or tasks beyond factual recall.

**Use Cases:**

*   **Preparing Generic Data for Augmentoolkit Training:** The primary use case is processing standard instruction datasets (Alpaca, Dolly, OpenHermes, etc.) so they can be effectively combined with Augmentoolkit's domain-specific datasets during SFT.
*   **Enhancing Existing Datasets:** Adding plausible reasoning steps to any request-response dataset to potentially improve model performance on tasks requiring intermediate thought.
*   **Data Standardization:** Ensuring consistency in data format across different dataset sources before training.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.