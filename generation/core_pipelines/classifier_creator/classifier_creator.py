import asyncio
import glob
import hashlib
import json
import os
import re
import sys
import traceback
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from augmentoolkit.generation_functions import engine_wrapper_class
from augmentoolkit.generation_functions.hashing_and_ordering import hash_input_list
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.generation_functions.single_generation_step import (
    SingleGenerationStep,
)
from augmentoolkit.utils import parse_string_list
from augmentoolkit.utils.head_tail_truncate import head_tail_truncate
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.parse_bool import parse_bool
import augmentoolkit.utils.sentence_chunking_algorithm
from generation.core_components.chunking import read_all_text
from generation.core_components.filter_chunks import filter_out_failed_items_dict
from generation.core_components.meta_datagen import create_meta_dataset
from generation.core_components.setup_components import (
    make_relative_to_self,
    setup_semaphore_and_engines,
)
from redis_config import set_progress


def get_last_final_label(text):
    pattern = r"Final label: (.+)"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def format_class_list(class_list):
    result_str = ""
    for idx, item in enumerate(class_list):
        result_str += f"{idx}. {item}\n"


def get_accuracy(dataset_dict):
    total = 0
    correct = 0
    missing_labels = 0
    missing_predictions = 0

    for item in dataset_dict.values():
        label_true = item.get("label")
        label_pred = item.get("classifier_label")

        if label_true is None:
            missing_labels += 1
            continue
        if label_pred is None:
            missing_predictions += 1
            continue

        total += 1
        if label_true == label_pred:
            correct += 1

    if total == 0:
        print("No items to compare labels.")
        return False

    return correct / total


def enough_labels_same(dataset_dict, required_accuracy=0.9):
    total = 0
    correct = 0
    missing_labels = 0
    missing_predictions = 0

    for item in dataset_dict.values():
        label_true = item.get("label")
        label_pred = item.get("classifier_label")

        if label_true is None:
            missing_labels += 1
            continue
        if label_pred is None:
            missing_predictions += 1
            continue

        total += 1
        if label_true == label_pred:
            correct += 1

    if total == 0:
        print("No items to compare labels.")
        return False

    accuracy = correct / total

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Required accuracy: {required_accuracy:.2%}")
    if missing_labels:
        print(f"Warning: {missing_labels} items missing original labels.")
    if missing_predictions:
        print(f"Warning: {missing_predictions} items missing classifier predictions.")

    if accuracy >= required_accuracy:
        print("Classifier meets or exceeds the required accuracy.")
        return True
    else:
        print("Classifier does not meet the required accuracy.")
        return False


def run_classifier(
    input_dict=None, model=None, output_dir=None, output_list=None
):  # model is a pipeline
    try:
        # Get keys in consistent order and pair with outputs
        items = list(input_dict.items())
        inputs = [value["text"] for key, value in items]
        outputs = model(inputs)

        # Pair each output with its original key using zip
        for (key, value), output in zip(items, outputs):
            input_dict[key]["classifier_label"] = output
    except Exception as e:
        print(e)
        traceback.print_exc()


rules_creator = SingleGenerationStep(
    prompt_path="create_rules_for_desc",
    output_file="rules_creation_generation",
    sampling_params={
        "max_tokens": 3000,
        "stop": ["### Response", "\n\n\n\n\n\n", "<|im_end|>"],
        "temperature": 0.2,
        "top_p": 0.9,
    },
    details_key="rules_creation_details",
    result_key="result",
)


async def classifier_creator(
    large_api_key: str,
    large_base_url: str,
    large_model: str,
    small_model: str,
    small_api_key: str,
    small_base_url: str,
    classes: list[str],
    desc: str,
    predict_on_whole_set_at_the_end: bool,
    input_dir: str,
    output_dir: str,
    prompts: str,
    default_prompts: str,
    chunk_size: int,
    completion_mode: bool,
    concurrency_limit: int,
    large_mode: str,
    small_mode: str,
    required_accuracy: float,
    use_stop: bool,
    max_iters: int,
    model_path: str,
    test_set_size: int,
    train_set_increment: int,
    train_set_size: int,
    truncation_type: str,
    do_meta_datagen: bool,
    meta_datagen_keys: list[str],
    meta_datagen_extras: list[str],
    read_files_manually: bool = True,
    dataset_passed_in: list[dict[str, str]] = [],
    task_id=None,
    *args,
    **kwargs,
):

    def parse_labels(classification):
        try:
            predicted_label = get_last_final_label(classification)
        except Exception as e:
            raise Exception(
                f"Model output could not be parsed. Model was stupid. Not pipeline's fault. Probably. {e}"
            )
            traceback.print_exc()
        for idx, c in enumerate(classes):
            if c.strip() == predicted_label.strip():
                return idx
        # if we got down here, maybe see if it gave us a number:
        try:
            pred = int(predicted_label)
            classes[pred]  # test that it is not out of bounds
            return pred
        except:
            pass

        # maybe see if it gave us BOTH
        try:
            pred = predicted_label.split(" ")
            pred = pred[0]
            pred = int(pred)
            classes[pred]  # test that it is not out of bounds
            return pred
        except:
            pass

        # Handle the case where the model says something like "1. positive." or "1. positive"
        try:
            parts = predicted_label.split(".")
            if len(parts) >= 2:
                pred = int(parts[0].strip())
                label = parts[1].strip().rstrip(".")
                if classes[pred].strip().lower() == label.lower():
                    return pred
        except:
            pass

        raise Exception(
            f"\n-----\/----\nNo proper label found! Generated {classification}\n\nExtracted {predicted_label}\n\nAnd tried to match with{classes}"
        )  #

    label_creator = PipelineStep(
        prompt_path="create_labels_for_chunk",
        output_file="llm_classifications",
        sampling_params={
            "max_tokens": 3000,
            "stop": ["### Response", "\n\n\n\n\n\n", "<|im_end|>"],
            "temperature": 0.2,
            "top_p": 0.9,
        },
        output_processor=parse_labels,
        result_key="label",
        details_key="label_details",
    )

    def train_classifier(text_label_dicts, classifier_counter, output_dir):
        os.environ["WANDB_DISABLED"] = "true"
        # First, save the tuples to a file with only the relevant info so that we can load them as a dataset

        # data will be saved as json lines
        classifier_counter += 1
        os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)
        path_to_dataset = os.path.join(
            output_dir, "datasets", f"dataset_{classifier_counter}.jsonl"
        )
        with open(path_to_dataset, "w", encoding="utf-8") as f:
            for key, value in text_label_dicts.items():
                json_obj = {"text": value["text"], "label": value["label"]}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

        ### TRAINING CODE
        dataset = load_dataset("json", data_files=path_to_dataset)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            use_cpu=True,
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            # eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics,
        )

        classifier_output_path = os.path.join(
            output_dir, f"classifier_{classifier_counter}"
        )

        trainer.train()
        trainer.save_model(output_dir=classifier_output_path)
        tokenizer.save_pretrained(classifier_output_path)

        new_model = AutoModelForSequenceClassification.from_pretrained(
            classifier_output_path, num_labels=2
        )

        new_tokenizer = AutoTokenizer.from_pretrained(classifier_output_path)

        def predict(text, prediction_batch_size=100):
            outputs = []

            # Calculate the total number of batches
            total_batches = (
                len(text) + prediction_batch_size - 1
            ) // prediction_batch_size

            # Process text in batch_size groups with tqdm progress bar
            for i in tqdm(
                range(0, len(text), prediction_batch_size),
                total=total_batches,
                desc="Predicting",
            ):
                batch = text[i : i + prediction_batch_size]
                encoding = new_tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                )
                batch_outputs = new_model(**encoding)
                batch_predictions = batch_outputs.logits.argmax(-1)
                outputs.extend(batch_predictions.tolist())

            return outputs

        return predict

    if kwargs:
        print("Additional arguments provided:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    prompts = make_relative_to_self(prompts)
    default_prompts = make_relative_to_self(default_prompts)

    run_task_with_limit, engine_wrapper, engine_wrapper_large, _ = (
        setup_semaphore_and_engines(
            concurrency_limit,
            small_model,
            small_api_key,
            small_base_url,
            small_mode,
            large_model,
            large_api_key,
            large_base_url,
            large_mode,
        )
    )
    set_progress(
        task_id, progress=0.0, message="Pipeline starting; reading and truncating files"
    )

    if read_files_manually:
        dataset = read_all_text(input_dir=input_dir, chunk_size=chunk_size)
        print(f"Absolute path of input directory: {os.path.abspath(input_dir)}")
        print(f"Read {len(dataset)} chunks from {input_dir}")
    else:
        dataset = dataset_passed_in

    if truncation_type == "head-tail":
        dataset = [
            {
                "text": head_tail_truncate(item["text"], chunk_size),
                "metadata": item["metadata"],
            }
            for item in dataset
        ]
    else:
        dataset = [
            {"text": item["text"][:chunk_size], "metadata": item["metadata"]}
            for item in dataset
        ]

    if train_set_size + test_set_size > len(dataset):
        print(
            "\n\nTRAIN SET SIZE AND TEST SET SIZE TOO LARGE FOR EVEN A SINGLE CLASSIFIER TRAINING RUN GIVEN THE SIZE OF THE DATASET"
        )
        print("REDUCE TRAIN OR TEST SET SIZE, OR ADD MORE INPUT DATA")
        print(f"For reference, the total length of the chunks is {len(dataset)}")
        sys.exit(1)

    # deterministically shuffle the dataset
    dataset = sorted(dataset, key=lambda x: hashlib.md5(x["text"].encode()).hexdigest())

    set_progress(task_id, progress=0.1, message="Files read; Creating rules")
    rules = await rules_creator.run(
        input_data={"classes_desc": desc, "classes_list": format_class_list(classes)},
        engine_wrapper=engine_wrapper_large,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
    )

    rules_string = rules["result"]

    set_progress(
        task_id,
        progress=0.1,
        message="Rules created; beginning dataset generation and classifier training (this may take a while)",
    )

    # note: we don't even need to load the generated data into a dict and then go from there. We can literally create a subset, hash it into a dict, generate stuff with it using the normal pipelinestep; then on each subsequent loop iteration, we add new items to the list and hash it again. Then run over the entire list. Things that were missed the first time, will be picked up on the next loop iteration.
    # This fits with the current abstractions.

    # First iteration

    # items to sample is simply train_set_size
    sub_dataset = dataset[:train_set_size]
    dataset = dataset[train_set_size:]

    # hash the subset
    sub_dataset_dict = hash_input_list(sub_dataset, key_to_hash_with="text")

    # generate first pass of the data using the subset
    await label_creator.execute_pipeline(
        input_dict=sub_dataset_dict,
        engine_wrapper=engine_wrapper,
        default_prompt_folder=default_prompts,
        prompt_folder=prompts,
        output_dir=output_dir,
        completion_mode=completion_mode,
        use_stop=use_stop,
        rules=rules_string,
        classes=format_class_list(classes),
        rtwl=run_task_with_limit,
        include_details=do_meta_datagen,
    )

    set_progress(
        task_id,
        progress=0.15,
        message="First labels created; first classifier being trained",
    )

    # filter out failed items
    filter_out_failed_items_dict(sub_dataset_dict, key_to_check="label")

    classifier_counter = 0
    output_dir = os.path.join(output_dir, "classifiers")
    os.makedirs(output_dir, exist_ok=True)

    # Count existing classifier folders
    existing_classifiers = glob.glob(os.path.join(output_dir, "classifier_*"))
    classifier_counter = len(existing_classifiers)

    # train the first classifier
    model = train_classifier(sub_dataset_dict, classifier_counter, output_dir)
    has_passed_LLM_validation = False

    while not has_passed_LLM_validation and max_iters > 0:
        max_iters = max_iters - 1
        if dataset:
            # make the output dir
            output_dir = os.path.join(output_dir, "truth_labels_classification")
            os.makedirs(output_dir, exist_ok=True)

            # First, take out a test set
            test_set = dataset[:test_set_size]
            # and remove those same items from the dataset
            dataset = dataset[test_set_size:]

            test_set_dict = hash_input_list(test_set, key_to_hash_with="text")
            # Do LLM testing on that test set
            await label_creator.execute_pipeline(
                input_dict=test_set_dict,
                engine_wrapper=engine_wrapper,
                default_prompt_folder=default_prompts,
                prompt_folder=prompts,
                output_dir=output_dir,
                completion_mode=completion_mode,
                use_stop=use_stop,
                rules=rules_string,
                classes=format_class_list(classes),
                rtwl=run_task_with_limit,
                include_details=do_meta_datagen,
            )

            # filter out failed items
            filter_out_failed_items_dict(test_set_dict, key_to_check="label")

            # run the classifier on the test set
            run_classifier(model=model, output_dir=output_dir, input_dict=test_set_dict)

            if (
                acc := get_accuracy(test_set_dict) >= required_accuracy
            ):  # all_labels_same will have to work regardless of item order, since async. Thankfully dicts see to this now.
                has_passed_LLM_validation = True
                set_progress(
                    task_id,
                    progress=1.0,
                    message=f"Last classifier reached accuracy of {acc} (target {required_accuracy}) with a dataset {len(sub_dataset_dict.item())} items long. Good enough classifier trained! Saving...",
                )
            else:
                sub_dataset_dict = (
                    sub_dataset_dict | test_set_dict
                )  # arguably the duplication (since we modify the indicies of the dataset list and therefore two items might have a hash clash) is not a concern since we want varied data. By DESIGN you see. This is the least-used pipeline anyway. I would welcome a PR that fixes this.

                # add the next train set batch to the dataset
                new_train_samples_inputs = dataset[:train_set_increment]
                dataset = dataset[train_set_increment:]

                # hash the subset
                new_train_samples_dict = hash_input_list(
                    new_train_samples_inputs, key_to_hash_with="text"
                )

                # label it
                await label_creator.execute_pipeline(
                    input_dict=new_train_samples_dict,
                    engine_wrapper=engine_wrapper,
                    default_prompt_folder=default_prompts,
                    prompt_folder=prompts,
                    output_dir=output_dir,
                    completion_mode=completion_mode,
                    use_stop=use_stop,
                    rules=rules_string,
                    classes=format_class_list(classes),
                    rtwl=run_task_with_limit,
                    include_details=do_meta_datagen,
                )  # this won't overwrite the previous thing, it will just add keys

                # filter out failed items
                filter_out_failed_items_dict(
                    new_train_samples_dict, key_to_check="label"
                )

                # combine the new and old samples
                sub_dataset_dict = sub_dataset_dict | new_train_samples_dict

                # train the classifier
                model = train_classifier(
                    sub_dataset_dict, classifier_counter, output_dir
                )

                # increment the classifier counter
                classifier_counter += 1
                set_progress(
                    task_id,
                    progress=0.2 + 0.8 * required_accuracy / acc,
                    message=f"Last classifier reached accuracy of {acc} (target {required_accuracy}) with a dataset {len(sub_dataset_dict.item())} items long. Attempting to train another...",
                )

        else:
            print("No more data to train on. Exiting.")
            break

        if predict_on_whole_set_at_the_end:
            set_progress(
                task_id,
                progress=0.95,
                message=f"Predicting on whole set at the end using the classifier (excludes the training and test sets so far)... depending on total set size, this might take a while...",
            )
            print(
                "Predicting on entire remaining set (excludes the training and test sets so far)..."
            )
            output_dir = os.path.join(output_dir, "final_classifier_output")
            os.makedirs(output_dir, exist_ok=True)

            # hash the entire data
            entire_dataset_dict = hash_input_list(dataset, key_to_hash_with="text")

            run_classifier(
                model=model, output_dir=output_dir, input_dict=entire_dataset_dict
            )

        if do_meta_datagen:
            create_meta_dataset(
                data_dicts=[sub_dataset_dict, test_set_dict],
                output_dir=os.path.join(output_dir, "meta_datagen"),
                meta_datagen_keys=meta_datagen_keys,
                meta_datagen_extras=meta_datagen_extras,
                input_processors=[],
            )

        set_progress(task_id, progress=1.0, message="Pipeline Complete")
