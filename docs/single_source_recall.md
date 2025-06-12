# Single-Source Recall Factual Datagen

This pipeline is the original factual dataset generation method from earlier versions of Augmentoolkit, ported to the new framework. It focuses on creating question-answer pairs where the answer is derived from a *single* chunk of text. Unlike its multi-source counterpart, it includes more granular validation and repair steps for the generated QA pairs and an optional conversation generation step.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`huggingface` section:**
*   `hub_path`: (String) The path on Hugging Face Hub where the dataset should be pushed (e.g., `yourusername/your-dataset-name`).
*   `private`: (Boolean) Whether the dataset repository on the Hub should be private.
*   `push_to_hub`: (Boolean) Whether to push the generated datasets to the Hugging Face Hub.

**`skip` section:**
*   `skip_answer_relevancy_check`: (Boolean) If `True`, bypasses the step checking if the generated answer is relevant to the question and grounded in the text chunk.
*   `skip_repair_qa_tuples`: (Boolean) If `True`, bypasses the step where an LLM attempts to reword or fix QA pairs that failed previous checks.
*   `skip_filter_chunks`: (Boolean) If `True`, bypasses the initial LLM check that determines if a text chunk is suitable for generating questions.
*   `skip_question_check`: (Boolean) If `True`, bypasses the step checking if the generated question is valid and answerable from the text chunk.
*   `skip_conversation_generation`: (Boolean) If `True`, bypasses the final step where QA pairs are woven into a simulated conversation.
*   `skip_answer_accuracy_check`: (Boolean) If `True`, bypasses the step checking if the generated answer is factually accurate according to the text chunk.

**`system` section:**
*   `conversation_instructions`: (String) Instructions given to the large model when generating the conversational wrapper around QA pairs (if `skip.skip_conversation_generation` is `False`).
*   `double_check_counter`: (Integer) The number of times validation steps (question check, answer relevancy, answer accuracy) should be run for majority voting. Setting to 1 effectively disables majority voting.
*   `do_not_use_system_prompts`: (Boolean) If `True`, the system prompts defined in `final_assistant_prompts_no_rag` and `final_assistant_prompts_rag` will not be added to the start of the generated conversations/QA lists.
*   `final_assistant_prompts_no_rag`: (List of Strings) System prompts to be randomly chosen for conversations/QA lists *without* RAG context.
*   `final_assistant_prompts_rag`: (List of Strings) System prompts to be randomly chosen for conversations/QA lists *with* RAG context. Must include `{data}` placeholder for the (single-source) text chunk. *(Note: While RAG prompt options exist for consistency, this pipeline fundamentally operates on single chunks, unlike the multi-source pipeline's ChromaDB-based RAG)*.
*   `use_filenames`: (Boolean) Whether to include filenames in prompts, providing source context to the LLM. Affects which specific prompt files are used (e.g., `judge_paragraph_filenames` vs. `judge_paragraph_no_filenames`, `qatuples_gen_filenames` vs. `qatuples_gen_no_filenames`).
*   `rag_failure_percentage`: (Float, 0.0 to 1.0) The probability that the RAG context provided in `simplified_data_rag.jsonl` will be intentionally incorrect (using a randomly chosen chunk). Helps train model robustness but RAG data is secondary here.
*   `items_per_conversation`: (Integer) The number of QA pairs to group together into a single conversation turn in the final dataset (applies to both plain QA and generated conversations).

**`scraping` section:** *(Used only if `read_files_manually` is `True` and `use_gutenberg` is `True`)*
*   `use_gutenberg`: (Boolean) Whether to scrape books from Project Gutenberg based on the `start_url`.
*   `start_url`: (String) The starting URL on Project Gutenberg to begin scraping book links.
*   `max_books`: (Integer) The maximum number of books to download.
*   `max_failures`: (Integer) The maximum number of consecutive download failures before stopping the scraping process.

## Model Requirements

*   **`small_model`:** Used for initial chunk filtering (`filter_all_questions_step`) and potentially for the validation steps (`question_validation_step`, `answer_relevancy_validation_step`, `answer_accuracy_validation_step`) if they are not skipped. These steps involve classification or checks based on the prompt instructions. A dedicated reasoning model (like Qwen/QwQ or an R1 distill) is *not* required. Capable and cost-effective instruction-following models like Llama-3.1-8B-Instruct are appropriate.
*   **`large_model`:** Used for generating the QA pairs (`question_generation_step`), repairing QA pairs (`context_repairer_step`), and generating conversations (`conversation_generator_step`) if these steps are not skipped. Requires strong instruction-following and generation capabilities. A dedicated reasoning model is *not* required. Models like Llama-3.1-70B-Instruct are suitable choices. I find that Llama 3.3 is much worse than Llama 3.1 at following desired output formats in both this pipeline and the multi-source factual recall pipeline, so be wary with newer models that might have bad post-training or are overfit to certain metrics.

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir`, including `.txt`, `.md`, `.pdf`, `.docx`, and even `.jsonl` files containing a "text" key. It reads these files and splits them into smaller chunks based on `system.chunk_size` in the config.

## Output Files Guide

The primary outputs are saved in the `output_dir` from the config:

*   `plain_qa_list.jsonl`: **The main SFT dataset.** Each line is a JSON object containing a "conversations" list. Conversations consist of alternating "human" (question) and "gpt" (answer) turns. System prompts may be included based on `system.do_not_use_system_prompts`. QA pairs are grouped based on `system.items_per_conversation`.
*   `simplified_data_rag.jsonl`: **Generally less useful than `plain_qa_list.jsonl` for this pipeline.** Similar format, but includes a system prompt with the source text chunk `{data}` (potentially swapped out based on `rag_failure_percentage`).
*   `simplified_conversation_list.jsonl`: *(Only if `skip.skip_conversation_generation` is False)* SFT data where the QA pairs have been embedded within a generated conversational structure (human/AI chat).
*   `simplified_conversation_rag_list.jsonl`: *(Only if `skip.skip_conversation_generation` is False)* Similar to `simplified_conversation_list.jsonl`, but with RAG-style system prompts including the source chunk `{data}`.
*   `debug_outputs/` (Folder): Contains YAML files logging the inputs and outputs of each pipeline step (e.g., `judge_paragraph.yaml`, `factual_questions.yaml`, `conversations_and_questiongroups.yaml`). Essential for debugging.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs for each model.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, contains the generated meta-dataset (`meta_dataset.jsonl`).

**Which files to use for training?**

*   Use `plain_qa_list.jsonl` for standard SFT to teach factual recall from single text sources.
*   Use `simplified_conversation_list.jsonl` if you want SFT data where the facts are presented within a more natural conversational flow (requires conversation generation to be enabled).
*   The `simplified_data_rag.jsonl` and `simplified_conversation_rag_list.jsonl` files are less critical for this pipeline as it doesn't inherently perform multi-source RAG like its counterpart. They primarily exist for format consistency.
*   A pre-training dataset can be generated by combining the text chunks, typically handled by a separate pipeline or composition.

## Purpose of Overall Pipeline and Use Cases

This pipeline creates supervised fine-tuning data to teach an LLM factual information from provided documents, focusing on grounding each question and answer pair within a *single* source text chunk. It represents the original methodology of Augmentoolkit's factual generation.

It includes several LLM-driven validation steps (chunk suitability, question validity, answer relevancy, answer accuracy) and a repair step, allowing for potentially higher scrutiny of individual QA pairs compared to the newer multi-source pipeline, at the cost of speed and complexity. It also offers an optional step to embed the QA pairs into a generated conversational format, which can influence the model's tone and interactive style.

Overall I find models trained on the multi-source data much more compelling and interesting and powerful, but sometimes a single source is all you need and you REALLY want validation of the data. Also, this may be useful for spare parts for eventually adding more features to the multi source pipeline. Not all of the same steps will be required -- the single source pipeline was built when open LLMs were much worse, and is in some ways more complex than it should be due to this fact. But some (like validation) are useful.

Notably, since RAG isn't perfect, if you're concerned with RAG retrieving bad sources based on superficial relevance, this pipeline avoids that problem by not using RAG and generating questions only based on single chunks of text.

**Use Cases:**

*   **Targeted Factual Tuning:** Ideal when ensuring each generated fact is strictly traceable to a specific, isolated piece of text is crucial.
*   **High-Scrutiny Data Generation:** When the overhead of multiple validation steps is acceptable to potentially increase the quality and reliability of individual QA pairs.
*   **Controlling Conversational Style:** Using the conversation generation step (`skip.skip_conversation_generation: False`) allows for tuning the model's persona and conversational ability alongside factual recall.
*   **Foundation for Domain Expertise:** Like the multi-source pipeline, it's used to build domain-expert models, though its grounding is limited to single chunks at a time during generation.

This pipeline is often used as a component within larger compositions like `complete_factual_datagen` to leverage its high quality domain SFT generation alongside other pipelines. There is an option in `complete_factual_datagen` to use single source generation instead of multi source generation. The config files for the two pipelines strongly resemble each other and in many cases, one config can run either pipeline.

#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.