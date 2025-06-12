# Rag Data (preparing for enhanced recall)

This pipeline is specifically designed to train LLMs on how to effectively utilize context provided by a Retrieval-Augmented Generation (RAG) system. It generates multi-turn conversational data where the model is presented with simulated RAG results (both relevant and deliberately irrelevant chunks) alongside a question derived from a source text chunk. The goal is to teach the model to synthesize answers using the provided context when relevant, or rely on its internal knowledge (trained via other pipelines like Multi-Source Recall) when the context is irrelevant or unhelpful.

It uses a specific loss masking strategy tailored for frameworks like Axolotl, masking the provided RAG context itself and previous turns, focusing the training signal only on generating the *correct answer* given the current question and context.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`skip` section:**
*   `skip_filter_chunks`: (Boolean) If `True`, bypasses the initial LLM check that determines if a source text chunk is suitable for generating QA.

**`system` section:**
*   `rag_failure_percentage`: (Float, 0.0 to 1.0) The probability that the simulated RAG context presented to the LLM during QA generation will consist *only* of chunks unrelated to the source text chunk the question is based on.
*   `rag_max_chunks`: (Integer) The maximum number of text chunks to include in the simulated RAG context provided to the LLM.
*   `final_assistant_prompts`: (List of Strings) System prompts given to the LLM during training/inference. Must include a `{data}` placeholder, which will be filled with the stringified RAG context chunks.
*   `system_format`: (String) Jinja template for formatting the system prompt part of the conversation.
*   `user_format`: (String) Jinja template for formatting the user turn part of the conversation.
*   `assistant_format`: (String) Jinja template for formatting the assistant turn part of the conversation.
*   `bos`: (String) The beginning-of-sequence token to use (e.g., `"<s>"`).
*   `num_items_per_group`: (Integer) How many distinct QA generation contexts (each based on one source chunk + its simulated RAG results) are grouped together to create the final segmented training instances.

## Model Requirements

*   **`small_model`:** Used for the optional initial chunk filtering step (`filter_chunks_step`). Requires a standard **instruction-following model**. Reasoning models can work as well with the filter chunks step, though sometimes they mess up the output format. QWQ would not be bad here.
*   **`large_model`:** Used for the core generation steps (`rag_failed_step`, `rag_success_step`) which generate QA pairs based on the source text *and* the potentially misleading simulated RAG context. Requires a strong **instruction-following model** capable of reasoning with provided context and generating accurate answers. Reasoning models not recommended for these prompts.

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir` (`.txt`, `.md`, `.pdf`, `.docx`, `.jsonl` with "text" key). It reads and chunks these files based on `system.chunk_size`.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `axolotl_rag_conversations.jsonl`: **The main output file, ready for Axolotl training.** This is a JSONL file where each line is a structured conversation designed for loss masking. Each line contains a `"segments"` list. Within these segments:
    *   The formatted system prompt (containing the simulated RAG context) is always masked (`"label": false`).
    *   User turns (questions) and Assistant turns (answers) from *previous* QA items within the group are masked (`"label": false`).
    *   Only the user question and assistant answer for the *current* target QA item within the group are unmasked (`"label": true`).
*   `debug_outputs/`: Contains YAML files logging intermediate steps like chunk filtering (`rag_convs.yaml`) and the RAG-based QA generation (`rag_failed_convs.yaml`, `rag_success_convs.yaml`).
*   `rag_prepared_data.jsonl`: A file caching the results of the RAG simulation (which chunks were selected for each source chunk, and whether it was marked as a success or failure case). Useful for resuming runs.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs for each model.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, contains the generated meta-dataset (`meta_dataset.jsonl`).

**Which files to use for training?**

*   Use `axolotl_rag_conversations.jsonl` directly with training frameworks like Axolotl that support the `completion_records` format with segmented data and loss masking. This trains the model to generate the correct answer given the question and context, without forcing it to learn the potentially noisy context itself.

## Purpose of Overall Pipeline and Use Cases

The RAG Data pipeline focuses on teaching an LLM the *skill* of using retrieved information effectively. It simulates the scenario where a model receives context from a RAG system alongside a user query.

By generating examples with both relevant ("success") and irrelevant/misleading ("failed") context, and using a specific loss-masking strategy, it trains the model to:
1.  Identify and utilize helpful information from the provided RAG context.
2.  Ignore irrelevant or incorrect information in the context.
3.  Rely on its own internal knowledge (built through pretraining and other SFT pipelines) when the context is not useful.
4.  Integrate information from multiple retrieved chunks when appropriate.

The masking strategy is key: by masking the context and previous turns, the model is only penalized for producing the wrong answer *given the current question and context*, not for the context itself. This encourages flexible context use without overfitting to the potentially noisy retrieved chunks.

**Use Cases:**

*   **Improving RAG System Performance:** Fine-tuning the generator component of a RAG system to better utilize the retriever's output.
*   **Training Robustness to Noisy Context:** Making models less susceptible to being misled by irrelevant information retrieved by imperfect RAG systems.
*   **Enhancing Contextual Reasoning:** Teaching models to synthesize information from multiple provided snippets.
*   **Building Advanced QA Systems:** Creating models that can dynamically switch between relying on provided context and their internal knowledge base.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.