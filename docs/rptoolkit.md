# RPToolkit

RPToolkit is designed to generate synthetic roleplay (RP) chat sessions inspired by the themes, styles, and content of provided input texts. You can feed it examples of writing (like novels, fanfiction, or plays), and it will generate multi-turn RP scenarios featuring characters and settings evocative of that input. The pipeline breaks down the process into steps like identifying emotions, generating character archetypes and features, creating a scene card, writing the RP story, and even rating the generated story's quality.

This pipeline was created to support the creative RP community by enabling the generation of datasets tailored to specific genres, styles, and tastes, allowing for the training of specialized RP models.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`phases` section:** 
*   `phase_index`: (Integer) Controls where to resume/stop execution if `work_in_phases` is `True`. `0` stops after archetype generation, `1` stops after story generation, `2` completes the full pipeline including rating.
*   `work_in_phases`: (Boolean) Enables phased execution.
* See [running pipelines with multiple phases](multiple_phases.md) for an explanation of this area.

**`system` section:**
*   `emotions`: (List of Strings) A list of emotions the model can *choose from* when `pick_emotion` is `False`.
*   `include_chunk_in_prompt`: (Boolean) If `True`, the original text chunk is included in the prompt when generating the story, potentially influencing the style more directly.
*   `pick_emotion`: (Boolean) If `True`, the LLM generates a fitting emotion based on the text chunk. If `False`, the LLM chooses the best-fitting emotion from the `system.emotions` list.
*   `rp_prompt_end`: (String) Custom text appended to the end of the system prompt used for the final ShareGPT dataset format.
*   `rp_prompt_start`: (String) Custom text prepended to the start of the system prompt used for the final ShareGPT dataset format.
*   `use_min_p`: (Boolean) Whether to use Min-P sampling for story generation. Can sometimes improve creativity/quality but requires the API to support it. Local generation will always support min_p.
*   `to_include_features`: (List of Strings) Specifies which features the pipeline should attempt to extract and use when generating the scene card (e.g., "Initiating Event", "Character Traits", "Feelings"). This does not effect the prompt, but rather should be written BASED ON the prompt -- the strings here are used to validate that the LLM wrote all the categories it is prompted to write.

**`archetypes` section:**
*   `generate_archetype`: (Boolean) If `True`, the LLM generates a character archetype based on the text. If `False`, it randomly picks one from the `archetypes.archetypes` list.
*   `archetypes`: (List of Strings) A list of predefined character archetypes to choose from if `generate_archetype` is `False`.

**`scraping` section:** *(Optional feature for gathering input data)*
*   `use_lightnovelco`: (Boolean) If `True`, attempts to scrape novels from lightnovelworld.co based on the other `lnco_` settings before starting the main pipeline.
*   `lnco_base_url`: (String) Base URL for the scraper.
*   `lnco_ranking_url`: (String) URL of the ranking page to find novels.
*   `lnco_chapter_count`: (Integer) How many chapters to scrape per novel.
*   `lnco_novel_count`: (Integer) How many novels to scrape.
*   `lnco_wait_time`: (Integer) Seconds to wait between requests during scraping.
*   `lnco_max_workers`: (Integer) Number of concurrent threads for scraping.

**NSFW Content Note:** The pipeline includes prompts intended for generating NSFW content (`prompts_spicy` vs standard `prompts`), catering to adult themes if present in input texts. However, due to the sensitive nature, the publicly distributed NSFW prompts are intentionally ambiguous. **Generating coherent NSFW content matching specific tastes typically requires using the custom datagen model trained to understand what these prompts actually mean.** Note that while the standard prompts are not censored, they also won't be terribly interesting or well-paced if you try to use them for NSFW purposes.

Also, despite the spicy prompts' ambiguity, I am still concerned about including them in the main project repo, so if you want them please go over to the Discord and ask (can't include this in my company's professional GitHub!)

## Model Requirements

*   **`small_model` & `large_model`:** RPToolkit was developed before the prevalence of dedicated reasoning models. **Both `small_model` and `large_model` should be strong general-purpose instruction-following models.** The prompts used for emotion generation, feature extraction, scene card generation, story generation, and rating are *not* designed for Chain-of-Thought or other explicit reasoning structures. Using reasoning models (like Qwen/QwQ or R1) may lead to format errors or unexpected behavior. Models like Llama 3.1, DeepSeek-V3, or Command R + are appropriate.

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir` (`.txt`, `.md`, `.pdf`, `.docx`, `.jsonl` with "text" key) or can scrape text if the `scraping.use_lightnovelco` option is enabled.

## Output Files Guide

The primary outputs are saved in the `output_dir/final_outputs/` folder:

*   `full_stories_list_sharegpt.json`: All generated stories (that didn't error out) formatted in ShareGPT format, suitable for training.
*   `good_and_above_stories_list_sharegpt.json`: Stories rated as "good" or "incredible" by the rating step, in ShareGPT format.
*   `incredible_stories_list_sharegpt.json`: Only stories rated as "incredible" by the rating step, in ShareGPT format.
*   `*_complete_format.json`: Corresponding JSON files containing the full data object for each story, including intermediate steps like emotion, archetype, features, scene card, and ratings. Useful for analysis.
*   `debug_outputs/story_generation.yaml`: A large YAML file containing the detailed inputs and outputs for *all* steps (emotion, archetype, features, scene card, story, rating) for each processed chunk. Essential for debugging the depth-first execution.
*   `small_model_tokens.json` / `large_model_tokens.json`: Track token counts and estimated costs.
*   `meta_datagen/` (Folder, optional): Contains the meta-dataset if enabled.

**Which files to use for training?**

*   Use one of the `*_sharegpt.json` files (likely `good_and_above` or `incredible` for higher quality) for fine-tuning RP models. The ShareGPT format includes the system prompt (constructed from `rp_prompt_start`, the scene card, and `rp_prompt_end`) and alternating user/assistant turns.

## Purpose of Overall Pipeline and Use Cases

RPToolkit's primary purpose is to **generate synthetic roleplay data that captures the essence of input texts.** It allows users to create datasets for fine-tuning LLMs to roleplay in specific genres, styles, or with characters inspired by source material.

It employs a multi-step, depth-first generation process:
1.  Analyzes text chunks to determine key emotional themes.
2.  Optionally generates or selects a character archetype.
3.  Extracts stylistic and content features.
4.  Synthesizes a "scene card" defining the character and setting.
5.  Generates a multi-turn RP chatlog based on the scene card.
6.  Rates the generated story for quality, coherence, and rule-following.

This structured approach aims to produce diverse and thematic RP scenarios that go beyond simple text continuation, providing rich data for training creative and engaging RP models.

Notably, the RPToolkit pipeline differs from most other Augmentoolkit pipelines in that **it uses a depth first approach.** Some steps will finish all the way before others start. This can be good for keeping a finger on the pulse of the output quality as a large run proceeds, but can make progress tracking lag a bit behind where it actually is, since the tqdm progress bar will show the % of all steps completed. Also, this pipeline has the most difficult datagen tasks in all Augmentoolkit so far, and so you will likely need to use the most expensive models you got (or the custom datagen one). Back in the day, Command R+ was used. Deepseek v3 shows tentative good results in the modern era.

**Use Cases:**

*   **Training Genre-Specific RP Models:** Creating datasets based on fantasy novels, sci-fi stories, historical texts, etc., to train models that excel in those genres.
*   **Developing Character Personas:** Generating data inspired by specific character dialogues or descriptions from input texts.
*   **Capturing Writing Styles:** Fine-tuning models to adopt the stylistic nuances of specific authors or source materials.
*   **NSFW RP Model Training:** Generating datasets tailored to specific adult themes and tastes (requires appropriate input data and potentially a custom datagen model for best results with provided spicy prompts. To get the spicy prompts and sample spicy config, go over to the Discord and ask me or check related channels, I need to keep the purity of the main project and my professional GitHub intact).
*   **Creative Exploration:** Bootstrapping ideas for RP scenarios or character interactions based on existing texts.



#### Is something still on your mind?

If you have any open questions, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.