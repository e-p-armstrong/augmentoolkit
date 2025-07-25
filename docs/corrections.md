# Correction Data (loss-masked mistakes)

This pipeline generates conversational data specifically designed to teach LLMs how to correct their own mistakes based on provided factual text. It creates scenarios where the AI first provides a flawed answer to a question derived from the text, the user points out the need for correction (implicitly or explicitly), and the AI then provides the correct answer. Crucially, the output is formatted for training frameworks like Axolotl that support loss masking, ensuring the model learns from the correction, not the initial mistake.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`system` section:**
*   `prompt_template`: (String) A Jinja2 template string that defines how the conversation turns (initial question, flawed answer, user confirmation, correct answer) are formatted into a final string before being segmented for Axolotl. This template dictates the roles, separators, and special tokens (like `bos_token`).

## Model Requirements

*   **`small_model`:** Used for the initial chunk filtering step (`filter_chunks_step`) to determine if a text chunk is suitable for generating a correction scenario. Requires a standard **instruction-following model** (e.g., Llama-3.1-8B-Instruct).
*   **`large_model`:** Used for the core generation step (`masked_conversation_creation_step`) which involves creating the initial question, the *plausible but flawed* answer, the user follow-up, and the final correct answer, all grounded in the source text. The nature of the prompts here require a **reasoning model** (like Qwen/QwQ, DeepSeek-R1 or its distills) highly recommended to generate coherent incorrect-then-correct sequences.

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir` (`.txt`, `.md`, `.pdf`, `.docx`, `.jsonl` with "text" key). It reads and chunks these files based on `system.chunk_size`.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `axolotl_correction_conversations.json`: **The main output file, ready for Axolotl training.** This is a single JSON file (not JSONL) containing a list of conversations. Each conversation object has a `"segments"` key, which holds a list of text segments. Segments corresponding to the initial user question, the user follow-up, and the final *correct* AI answer have `"label": true`. The segment containing the initial *flawed* AI answer has `"label": false`, instructing the training framework to mask the loss for that part.
*   `debug_outputs/`: Debugging outputs showing intermediate steps and LLM generations.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs for each model.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, contains the generated meta-dataset (`meta_dataset.jsonl`).

**Which files to use for training?**

*   Use `axolotl_correction_conversations.json` directly with training frameworks like Axolotl that understand the `completion_records` format with segmented data and loss masking (`label: false`). This allows the model to learn the pattern of identifying and correcting mistakes without reinforcing the flawed output itself. Note that this requires a sslightly different structure than normal training in axolotl; this pipeline is used as part of the complete factual datagen, you can see the configs generated by that pipeline as an example of how ot structure the reference to this under `datasets:` in a training config.

## Purpose of Overall Pipeline and Use Cases

The Correction Data pipeline aims to improve the reliability and truthfulness of LLMs by explicitly training them on self-correction patterns. By generating examples where a model produces an incorrect statement based on given context and then corrects itself after a user prompt, it teaches the model to:

1.  Recognize potential inaccuracies in its own output.
2.  Generate corrected responses when prompted.

The use of loss masking on the initial flawed answer is critical; it prevents the model from learning the incorrect information while still allowing it to learn the conversational flow and the process of correcting itself. IIRC, Anthropic does something similar to this with some of its SFT.

This pipeline is part of creating an Augmentoolkit domain expert.

**Use Cases:**

*   **Reducing Hallucinations:** Training models to identify and fix factual errors related to provided context.
*   **Improving Dialogue Robustness:** Making models better at handling situations where their initial response might be challenged or incorrect.
*   **Fine-tuning for Reliability:** Enhancing model trustworthiness by training on explicit correction examples.
*   **Leveraging Loss-Masking Frameworks:** Creating datasets specifically formatted for advanced training techniques available in frameworks like Axolotl.

#### Is something still on your mind?

If you have any open questions about this pipeline, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.