# Multi-Source Recall Factual Datagen

This pipeline generates conversational datasets focused on recalling factual information from multiple provided text sources. It leverages a Retrieval-Augmented Generation (RAG) approach using ChromaDB to find relevant context from the input documents before generating question-answer pairs and assembling them into multi-turn conversations.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`huggingface` section:**
*   `hub_path`: (String) The path on Hugging Face Hub where the dataset should be pushed (e.g., `yourusername/your-dataset-name`).
*   `private`: (Boolean) Whether the dataset repository on the Hub should be private.
*   `push_to_hub`: (Boolean) Whether to push the generated datasets to the Hugging Face Hub.

**`system` section:**
*   `conversation_instructions`: (String) Instructions given to the model when generating the conversational wrapper around QA pairs (though this pipeline primarily uses `save_plain_qatuples`).
*   `do_not_use_system_prompts`: (Boolean) If `True`, the system prompts defined in `final_assistant_prompts_no_rag` and `final_assistant_prompts_rag` will not be added to the start of the generated conversations.
*   `final_assistant_prompts_no_rag`: (List of Strings) System prompts to be randomly chosen for conversations *without* RAG context.
*   `final_assistant_prompts_rag`: (List of Strings) System prompts to be randomly chosen for conversations *with* RAG context. Must include `{data}` placeholder for the retrieved text.
*   `rag_failure_percentage`: (Float, 0.0 to 1.0) The probability that the RAG context provided in the RAG dataset (`simplified_data_rag.jsonl`) will be from a *different* source document than the one the question/answer pair is based on. This helps train the model to rely on its internal knowledge even when presented with potentially misleading context. RAG datasets from this pipeline are mostly deprecated, and it is only recommended to use data from `rag_data_pipeline` for training models to use explicitly retrieved context (here, you should stick with plain_qa_list) but if you do happen to use the RAG data made here, then this is how you can configure it.
*   `items_per_conversation`: (Integer) The number of QA pairs to group together into a single conversation turn in the final dataset.

**`scraping` section:** *(Used only if `read_files_manually` is `True` and `use_gutenberg` is `True`)*
*   `use_gutenberg`: (Boolean) Whether to scrape books from Project Gutenberg based on the `start_url`.
*   `start_url`: (String) The starting URL on Project Gutenberg to begin scraping book links.
*   `max_books`: (Integer) The maximum number of books to download.
*   `max_failures`: (Integer) The maximum number of consecutive download failures before stopping the scraping process.

## Model Requirements

*   **`small_model`:** Used for initial chunk filtering (`filter_all_questions_step`). This model evaluates if a text chunk is suitable for generating factual questions. A dedicated reasoning model (like Qwen/QwQ or an R1 distill) is *not* required. Models like Llama-3.1-8B-Instruct are appropriate.
*   **`large_model`:** Used for generating the actual question-answer pairs (`question_generation_step`). This model needs to read the source chunk and potentially related chunks (via RAG context), then generate relevant questions and accurate answers based *only* on the provided text. It requires strong **instruction-following and generation capabilities** to adhere to the format (`**QUESTION:** ... **ANSWER:** ...`). A dedicated reasoning model is *not* required. Models like Llama-3.1-70B-Instruct or Deepseek V3 are suitable choices. I find that Llama 3.3 is much worse than Llama 3.1 at following desired output formats in both this pipeline and the individual factual recall pipeline.

## Input Files

Normal text documents. txt, md, docx, pdf... even .jsonl with a "text" key works. This is not one of those pipelines that takes sharegpt JSONL as an input.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `plain_qa_list.jsonl`: **The main SFT dataset, and probably the only one you should use.** Each line is a JSON object containing a "conversations" list. Conversations consist of alternating "human" (question) and "gpt" (answer) turns. If `system.do_not_use_system_prompts` is `False`, a system prompt is added at the beginning. QA pairs are grouped based on `system.items_per_conversation`.
*   `simplified_data_rag.jsonl`: **Mostly deprecated, use the dedicated rag data pipeline instead.** An SFT dataset similar to `plain_qa_list.jsonl`, but designed for training RAG-aware models. Each conversation starts with a system prompt containing context retrieved via RAG (the `{data}` placeholder in `system.final_assistant_prompts_rag` is filled). With probability `system.rag_failure_percentage`, the provided context comes from an intentionally incorrect source document to improve robustness.
*   `debug_outputs/` (Folder): Contains yaml files logging the inputs and outputs of each pipeline step (e.g., `judge_paragraph.jsonl`, `factual_questions.jsonl`). Useful for debugging and understanding the pipeline's intermediate results.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs for each model.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, this folder contains the generated meta-dataset (`meta_dataset.jsonl`) for training models to perform the pipeline tasks.

**Which files to use for training?**

*   Use `plain_qa_list.jsonl` for standard supervised fine-tuning (SFT) to teach a model factual recall based on the provided documents.
*   Use `simplified_data_rag.jsonl` for SFT if you want the model to learn how to utilize retrieved context provided in its system prompt, including handling potentially irrelevant context.
*   The pre-training dataset is generated by combining the text chunks, typically handled by a subsequent Representation Variation pipeline or similar composition. This pipeline focuses on the SFT data.

## Purpose of Overall Pipeline and Use Cases

This is the core pipeline of Augmentoolkit. The Multi-Source Recall Factual Datagen pipeline create high-quality supervised fine-tuning data that teaches a Large Language Model to accurately answer questions about a new domain, recalling multiple relevant sources from its own weights/memory during a thought process section, before providing a final answer based on its own understanding.

Uniquely, RAG is used within the pipeline to help the model recall multiple relevant sources. This differentiates it from the original Augmentoolkit factual pipeline, which restricted models to recalling information from single chunks at a time.

The multi source facts pipeline comes with a variety of different prompt overrides. These are: openended, followup, negative, hallucination, and comparison. Openended focuses on long and detailed domain answers; followup teaches a model to respond to followup questions about an area; negative teaches a model to correct misunderstandings or contradictions about the domain; hallucination teaches a model to know what *does not exist* in the domain and to guard against hallucination; and comparison teaches the model to compare the qualities of things within the domain. Together, these enhance the reasoning and conversational ability of the model within the domain it operates in. To prevent you from having to manually chain all these together and manage a bunch of configs, the multi source facts pipeline is used extensively in the [Complete Factual Datagen](./complete_factual_datagen.md) pipeline.

Note that this pipeline does not have as many varied features of the single source pipeline, as it is newer. It lacks question/answer LLM-based validation, as well as tone adjustment via conversation generation.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.