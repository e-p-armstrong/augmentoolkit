# Representation Variation

This pipeline generates diverse representations of input text to create robust continued pretraining datasets. It first extracts core "atomic facts" from text chunks and then uses various prompts to rewrite these chunks and facts into different styles (e.g., articles, summaries, pseudocode, XML). It also applies programmatic variations like case changes. This process aims to help the model learn the underlying information in multiple formats, improving its generalization and understanding before domain-specific supervised fine-tuning (SFT).

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`system` section:**
*   `variation_generator_count`: (Integer) The number of different LLM-generated variations to create for *each* input chunk. The pipeline randomly selects prompts from the `prompts/variations` subfolder for each generation.
*   `make_inferred_facts`: (Boolean) **CRITICAL SETTING:** If `True`, the pipeline runs an additional step (`make_inferred_facts_step`) to generate facts that can be inferred *between the lines* of the text, rather than just directly stated atomic facts. **If `True`, ALL LLM steps (filtering, atomic facts, inferred facts, variations) require a reasoning model (like Qwen/QwQ or R1 or its distills) for the `small_model` AND `large_model` because the prompts use Chain-of-Thought or similar reasoning techniques.** If `False`, standard instruction-following models are sufficient for all steps.

**`dataset` section:**
*   `dataset_context`: (String) A brief description of the overall topic or domain of the input data (e.g., "Health"). This is used if `include_context_in_dataset` is `True`.
*   `include_context_in_dataset`: (Boolean) If `True`, prepends context information (like `[[[OVERALL_CONTEXT_IS -> {dataset_context}]]] Specific source: {filename}`) to each text entry in the final output dataset.
*   `code_variation_functions`: (List of Strings) A list of programmatic text variations to apply to the original chunks and the LLM-generated variations. Available options include: `"allcaps"`, `"lowercase"`, `"serialkillercase"`, `"titlecase"`, `"sentencecase"`, `"snakecase"`, `"kebabcase"`, `"camelcase"`, `"pascalcase"`, `"randomcase"`, `"invertcase"`, `"keyboard_augmentation"`. These add further diversity to the pretraining data. If you're not careful with these they can make the loss explode for certain datasets, killing the intelligence of the model trained -- generally only use these if your dataset is quite small (`keyboard_augmentation` is the least risky of all of them).
*   `additional_dataset_context`: (String, Optional) Any extra context string you want added during the LLM generation steps (passed as a template variable).

## Model Requirements

*   **`small_model`:**
    *   **If `make_inferred_facts: False`:** Used for initial chunk filtering. Requires a standard, cost-effective **instruction-following model** (e.g., Llama-3.1-8B-Instruct).
    *   **If `make_inferred_facts: True`:** Used for filtering, atomic facts, and inferred facts generation. **Requires a reasoning model** (e.g., Qwen/QwQ, an R1 distill) capable of following Chain-of-Thought style prompts.
*   **`large_model`:**
    *   **If `make_inferred_facts: False`:** Used for generating the diverse representations (`generate_variations_step`). Requires a strong **instruction-following model** (e.g., Llama-3.1-70B-Instruct, Deepseek V3).
    *   **If `make_inferred_facts: True`:** Used for generating the variations. **Requires a reasoning model** (e.g., Qwen/QwQ, an R1 distill) capable of following Chain-of-Thought style prompts.

**Summary:** The `make_inferred_facts` flag dictates whether you need reasoning models (like Qwen/QwQ) for *both* small and large models (if `True`) or just standard instruction-following models (if `False`).

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir`, including `.txt`, `.md`, `.pdf`, `.docx`, and `.jsonl` files containing a "text" key. It reads these files and splits them into smaller chunks based on `system.chunk_size`.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `final_output.jsonl`: **The main output file, intended for continued pretraining.** This is a JSONL file where each line contains a single JSON object with a `"text"` key. It includes:
    *   The original text chunks (after filtering).
    *   The extracted atomic facts for each chunk.
    *   The generated LLM variations for each chunk (number determined by `system.variation_generator_count`).
    *   Programmatic variations (e.g., capitalization, keyboard errors) applied to the original chunks and LLM variations (based on `dataset.code_variation_functions`).
    *   Optional context prefixes if `dataset.include_context_in_dataset` is `True`.
*   `debug_outputs/` (Folder): Contains YAML files logging the inputs and outputs of each pipeline step (e.g., `filter_chunks.yaml`, `synthetic_pretrain.yaml`). Essential for debugging.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs for each model.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, contains the generated meta-dataset (`meta_dataset.jsonl`).

**Which files to use for training?**

*   Use `final_output.jsonl` as **continued pretraining data**. This dataset, which includes the original source documents, forms the foundation upon which the domain-specific SFT datasets (like those from the recall pipelines) are applied.

## Purpose of Overall Pipeline and Use Cases

The Representation Variation pipeline is designed to create rich, diverse datasets for the crucial **continued pretraining** phase of building a domain-expert LLM with Augmentoolkit. By exposing the model to the core information (atomic facts) and the original text rewritten in numerous formats and styles (articles, lists, code comments, different capitalizations, simulated errors, etc.), it encourages the model to develop a deeper, more flexible understanding of the domain's concepts beyond simple surface-level pattern matching.

The optional `make_inferred_facts` mode pushes this further by training the model to reason about implications and connections within the text, not just regurgitate explicitly stated facts.

This pretraining step aims to significantly improve the effectiveness of subsequent supervised fine-tuning using datasets from pipelines like Multi-Source or Single-Source Recall. The model starts the SFT phase with a better internal representation of the domain knowledge.

**Use Cases:**

*   **Creating Base Models for Domain SFT:** Generating the foundational pretraining data needed before applying specific instruct/chat datasets.
*   **Improving Model Robustness:** Training the model on varied formats and simulated noise (like casing changes or keyboard errors) can make it less brittle and better at handling real-world text.
*   **Enhancing Deeper Understanding:** Encouraging the model to learn concepts rather than specific phrasings through exposure to diverse representations.
*   **Training Reasoning (with `make_inferred_facts: True`):** Explicitly training the model to make logical inferences based on the provided domain text.


#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.