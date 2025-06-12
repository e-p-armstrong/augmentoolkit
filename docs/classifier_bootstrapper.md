# Traditional Classifier Bootstrapper

This pipeline offers a distinct capability within Augmentoolkit: it **bootstraps the training of a traditional text classifier** (like DistilBERT) using LLMs. Instead of creating data *for* an LLM, it uses LLMs to generate classification rules and labels for your raw text data based on a natural language description you provide. It then iteratively trains a standard sequence classification model (on the CPU) until it reaches a desired accuracy threshold against the LLM's labels.

## Unique Config Options

Beyond the [common configuration fields](config_common_fields.md), this pipeline has the following specific options within its `config.yaml`:

**`classification` section:**
*   `classes`: (List of Strings) The names of the classes the classifier should distinguish between (e.g., `['spam', 'not spam']`).
*   `desc`: (String) A natural language description explaining the classification task and what defines each class. This is used by the LLM to generate classification rules and initial labels.
*   `predict_on_whole_set_at_the_end`: (Boolean) If `True`, after the classifier reaches the target accuracy, it will run predictions on all remaining unlabeled data from the input set.

**`system` section:**
*   `required_accuracy`: (Float, 0.0 to 1.0) The minimum accuracy the trained classifier must achieve on an LLM-labeled test set before the iterative training process stops.

**`training` section:**
*   `max_iters`: (Integer) The maximum number of training iterations (labeling more data -> retraining classifier -> testing) to perform before stopping, even if `required_accuracy` is not met.
*   `model_path`: (String) The path or Hugging Face identifier for the base sequence classification model to be fine-tuned (e.g., `distilbert-base-uncased`).
*   `test_set_size`: (Integer) The number of text chunks to use for the LLM-labeled test set in each iteration.
*   `train_set_increment`: (Integer) The number of *additional* text chunks to label with the LLM and add to the training set in each subsequent iteration if the accuracy target is not met.
*   `train_set_size`: (Integer) The number of text chunks to label with the LLM for the *initial* training set.
*   `truncation_type`: (String, `"head-tail"` or `"head"`) How to truncate input text chunks if they exceed `system.chunk_size`. `head-tail` keeps the beginning and end, `head` just keeps the beginning.

**`meta_datagen` section:** *(Optional, for training LLMs to run the *labeling* parts)*
*   `do_meta_datagen`: (Boolean) Whether to save LLM rule generation and labeling steps.
*   `meta_datagen_keys`: (List of Strings) Specifies which keys (e.g., `label_details`, `rules_creation_details`) to include.
*   `meta_datagen_extras`: (List of Strings) Extra prompt templates for custom training examples.

## Model Requirements

*   **`small_model`:** Used for the primary labeling step (`label_creator`), where it assigns a class label to input text chunks based on the generated rules. Requires a capable **instruction-following model**. A reasoning model is *not* required.
*   **`large_model`:** Used only for the initial `rules_creator` step, where it generates classification rules based on the `classification.desc` you provide. Requires a strong **instruction-following model**.
*   **Classifier Base Model (`training.model_path`):** This is *not* an LLM, but a traditional transformer model suitable for sequence classification (e.g., `distilbert-base-uncased`, `bert-base-uncased`). This is the model that gets iteratively trained on the CPU using the LLM-generated labels.

## Input Files

The pipeline accepts various document types found within the specified `path.input_dir` (`.txt`, `.md`, `.pdf`, `.docx`, `.jsonl` with "text" key). It reads and chunks these files based on `system.chunk_size` and `training.truncation_type`.

## Output Files Guide

The primary outputs are saved in the `output_dir/classifiers/` folder:

*   `classifier_<N>/` (Folder): Contains the trained classifier model files (PyTorch binaries, config, tokenizer files) for iteration `N`. The final, best-performing classifier will be in the folder with the highest number.
*   `datasets/dataset_<N>.jsonl`: The LLM-labeled data used to train classifier iteration `N`.
*   `truth_labels_classification/llm_classifications.yaml`: Debug file showing the LLM's labeling process for the test sets used in each iteration.
*   `rules_creation_generation.yaml`: Debug file showing the LLM's generated classification rules.
*   `final_classifier_output/` (Folder, optional): If `classification.predict_on_whole_set_at_the_end` is `True`, this contains the predictions of the final classifier on the remaining unlabeled data.
*   `meta_datagen/` (Folder, optional): If `meta_datagen.do_meta_datagen` is `True`, contains the meta-dataset (`meta_dataset.jsonl`) capturing the LLM labeling steps.

**Which files to use?**

*   The main output is the **final trained classifier model** located in the `classifier_<N>/` folder with the highest `N` inside `output_dir/classifiers/`. You can load this model using Hugging Face Transformers for inference.
*   The `datasets/dataset_<N>.jsonl` files can be useful if you want to inspect or reuse the LLM-labeled data.

## Purpose of Overall Pipeline and Use Cases

This pipeline bridges the gap between large language models and traditional machine learning classifiers. Its purpose is to **rapidly create a functional text classifier for a custom task when you lack pre-labeled data.**

You provide raw text and a simple description of your desired classes. The pipeline then uses:
1.  A **large LLM** once to interpret your description and generate classification rules. This LLM is also used to create the TEST set which your small classifier competes against.
2.  A **small LLM** iteratively to label batches of your data according to those rules.
3.  A **standard transformer model** (like DistilBERT) trained iteratively on the CPU using the LLM-labeled data.

This loop continues, adding more LLM-labeled data each time, until the CPU-trained classifier performs well enough (meeting `system.required_accuracy`) when evaluated against the LLM's labels on a held-out test set.

**Use Cases:**

*   **Fast Classifier Prototyping:** Quickly build a baseline text classifier for tasks like sentiment analysis, topic classification, spam detection, etc., without needing manual data labeling.
*   **Leveraging LLM Understanding for Simple Tasks:** Use an LLM's ability to understand nuanced descriptions to bootstrap a smaller, faster, cheaper-to-run traditional classifier.
*   **Data Scarce Scenarios:** Create a usable classifier when obtaining a large, manually labeled dataset is infeasible or too time-consuming.

Models trained on this pipeline came pretty close to those that are trained with human experts (a couple percent accuracy points away), and cost about a dollar to train or less in total.

**Warning:** automatic data balancing by class not included. Welcoming a PR on this front. Despite this pipeline being highly useful for a lot of business usecases, the classifier creator seems to be the least-used pipeline in Augmentoolkit, so I have not focused on it too much.

#### Is something still on your mind?

If you have any open questions about this pipeline, feel free to head over to the [Discord](https://discord.gg/s6PBfsaVzu) and ask them! Alternatively, if you want to read tips that are useful in the areas of dataset generation and model training (but are not strictly necessary for Augmentoolkit's use, hence why they're not just on the README) you can check out this [free informal blog]((https://promptingweekly.substack.com/)) I post to.