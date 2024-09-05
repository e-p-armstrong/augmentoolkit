import json
import logging
import os
import re
import sys
import traceback
from tqdm import tqdm
import yaml

from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.generation_functions.pipeline_step_class import PipelineStep
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.parse_bool import parse_bool
from augmentoolkit.utils.write_output_to_file import write_output_to_file

config_path = os.environ["CONFIG_PATH"]

with open(config_path, "r") as f: # different yaml file for different pipes
        config = yaml.safe_load(f)
        
COMPLETION_MODE = parse_bool(config["SYSTEM"]["COMPLETION_MODE"])
PROMPTS_DIR = os.path.abspath(config["PATH"]["PROMPTS"])
DEFAULT_PROMPTS = os.path.abspath(config["PATH"]["DEFAULT_PROMPTS"])
OUTPUT_DIR = os.path.abspath(config["PATH"]["OUTPUT"])
USE_STOP = parse_bool(config["SYSTEM"]["STOP"])

### PROMPT FUNC: Rules Creator (this does not use the pipeline step class due to it uniquely only generating a single thing and not writing to a list)

def parse_rules(rules_str):
    return rules_str # TODO

def format_class_list(class_list):
    result_str = ""
    for idx, item in enumerate(class_list):
        result_str += f"{idx}. {item}\n"
    
    return result_str.strip()

async def create_rules(engine_wrapper=None, classes_list=None, classes_desc=None, completion_mode=False):
    prompt_path = "create_rules_for_desc"
    
    if COMPLETION_MODE:
        prompt_path += ".txt"
    else:
        prompt_path += ".yaml"
    
    rules_creator = GenerationStep(
        prompt_path=prompt_path,
        sampling_params={
            "max_tokens": 1500,
            "stop": [
                "### Response",
                "\n\n\n\n\n\n",
                "</s>",
                "# Input:",
                "[INST]",
                "### Instruction",
                "[INST",
                "<|eot_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
            ],
            "temperature": 0.2,
        },
        completion_mode=completion_mode,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging.INFO,
        output_processor=parse_rules,
        prompt_folder=config["PATH"]["PROMPTS"],
        default_prompt_folder=config["PATH"]["DEFAULT_PROMPTS"],
        use_stop=config["SYSTEM"]["STOP"]
    )
    
    classes_str = format_class_list(classes_list)
    
    try:        
        
        result, full_output = await rules_creator.generate(classes_desc=classes_desc, class_list=classes_str)
        
        id = make_id()
        
        write_output_to_file(full_output, os.path.join(config["PATH"]["OUTPUT"], "rules_creation_generation"), id) # TODO move output dir to processing
        
        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
    

###

label_path = "create_labels_for_chunk"
label_regex = r"Final label: (.+)"

# for the sake of this abstraction, "rules" shall be part of the input data and intermediate saved stuff now.

class LabelCreator(PipelineStep):
    def __init__(self):
        super().__init__(
            prompt_folder=PROMPTS_DIR,
            default_prompt_folder=DEFAULT_PROMPTS,
            prompt_path=label_path,
            regex=label_regex,
            sampling_params={
                "max_tokens": 1500,
                "stop": [
                    "### Response",
                    "\n\n\n\n\n\n",
                    "</s>",
                    "# Input:",
                    "[INST]",
                    "### Instruction",
                    "[INST",
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "<|end_header_id|>",
                ],
                "temperature": 0.2,
            },
            output_dir=OUTPUT_DIR,
            output_subdir="label_creation_generations",
            intermediate_output_path="label_creation_intermediates",
            save_path="label_creations_saved", # NOTE output processor is defined inside async function and manually applied due to special circumstances
            result_key="label",
            use_stop=USE_STOP,
            completion_mode=COMPLETION_MODE,
        )
    
    def read_previous_output(self, idx, output_list):
        return False # We do not read previous output in this step
    
    def process_input_data(self, input_data):
        input_data["classes"] = format_class_list(input_data["classes"])
        return input_data
        
    
label_creator = LabelCreator()

### PROMPT FUNC: Label Creator

def get_last_final_label(text):
    pattern = r"Final label: (.+)"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


async def create_label(idx, inp, classes=None, engine_wrapper=None, output_list=None):
    
    def parse_labels(classification):
        predicted_label = get_last_final_label(classification)
        for idx, c in enumerate(classes):
            if c.strip() == predicted_label.strip():
                return idx
        # if we got down here, maybe see if it gave us a number:
        try:
            pred = int(predicted_label)
            classes[pred] # test that it is not out of bounds
            return pred
        except:
            pass
    
        # maybe see if it gave us BOTH
        try:
            pred = predicted_label.split(" ")
            pred = pred[0]
            pred = int(pred)
            classes[pred] # test that it is not out of bounds
            return pred
        except:
            pass
        
        # Handle the case where the model says something like "1. positive." or "1. positive"
        try:
            parts = predicted_label.split(".")
            if len(parts) >= 2:
                pred = int(parts[0].strip())
                label = parts[1].strip().rstrip('.')
                if classes[pred].strip().lower() == label.lower():
                    return pred
        except:
            pass
            
        raise Exception(f"\n-----\/----\nNo proper label found! Generated {classification}\n\nExtracted {predicted_label}\n\nAnd tried to match with{classes}") #
    
    label_creator.output_processor = parse_labels
    await label_creator.run(idx=idx, input_data=inp, engine_wrapper=engine_wrapper, output_list=output_list)
    
###


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_classifier(text_label_dicts, classifier_counter, output_dir):
    
    # First, save the tuples to a file with only the relevant info so that we can load them as a dataset
    
    # data will be saved as json lines
    
    
    # TODO
    classifier_counter += 1
    os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)
    path_to_dataset = os.path.join(output_dir, "datasets", f"dataset_{classifier_counter}.jsonl")
    with open(path_to_dataset, "w") as f:
        for d in text_label_dicts:
            json_obj = {
                "text": d["paragraph"],
                "label": d["label"]
            }
            f.write(json.dumps(json_obj) + "\n")

    ### TRAINING CODE
    dataset = load_dataset("json", data_files=path_to_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["MODEL_PATH"])
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(config["TRAINING"]["MODEL_PATH"], num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        use_cpu=True
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    classifier_output_path = os.path.join(output_dir, f"classifier_{classifier_counter}")
    
    trainer.train()
    trainer.save_model(output_dir=classifier_output_path)
    tokenizer.save_pretrained(classifier_output_path)
    
    new_model = AutoModelForSequenceClassification.from_pretrained(classifier_output_path, num_labels=2)
    
    new_tokenizer = AutoTokenizer.from_pretrained(classifier_output_path)
    
    def predict(text, prediction_batch_size=100):
        outputs = []
        
        # Calculate the total number of batches
        total_batches = (len(text) + prediction_batch_size - 1) // prediction_batch_size
        
        # Process text in batch_size groups with tqdm progress bar
        for i in tqdm(range(0, len(text), prediction_batch_size), total=total_batches, desc="Predicting"):
            batch = text[i:i+prediction_batch_size]
            encoding = new_tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            batch_outputs = new_model(**encoding)
            batch_predictions = batch_outputs.logits.argmax(-1)
            outputs.extend(batch_predictions.tolist())

        return outputs
    
    return predict
    
def run_classifier(input_list=None, model=None, output_dir=None, output_list=None): # model is a pipeline
    try:
        inputs = [i[0] for i in input_list]
        outputs = model(inputs)
        
        # print("OUTPUTS DEBUG:")
        # print(outputs)
        
        # sys.exit(0) # DEBUG TODO REMOVE once we have confirmed that classifier inference is functional
        for idx, inp in enumerate(input_list):
            id = make_id()
            with open(os.path.join(output_dir, f"{id}.json"), 'w') as f:
                f.write(json.dumps({
                    "text": inp[0],
                    "label": outputs[idx]
                }))
            
            out_tup = (inp[0], inp[1], outputs[idx])
            output_list.append(out_tup)
    except Exception as e:
        print(e)
        traceback.print_exc()

def all_labels_same(truth_labels, classifier_labels, required_accuracy=1.0):
    # Create dictionaries for fast lookup
    dict1 = {text: (textname, label) for text, textname, label in truth_labels}
    dict2 = {text: (textname, label) for text, textname, label in classifier_labels}

    inconsistencies = []
    not_found = []
    consistent_count = 0
    total_count = 0

    # Check consistency and existence
    for text, (textname1, label1) in dict1.items():
        if text in dict2:
            total_count += 1
            textname2, label2 = dict2[text]
            if label1 == label2:
                consistent_count += 1
            else:
                inconsistencies.append((text, textname1, label1, textname2, label2))
        else:
            not_found.append((text, textname1, label1, "list2"))

    # Check for texts in list2 not in list1
    for text, (textname2, label2) in dict2.items():
        if text not in dict1:
            not_found.append((text, textname2, label2, "list1"))

    # Calculate accuracy
    accuracy = consistent_count / total_count if total_count > 0 else 0

    # Print results
    if inconsistencies:
        print("Inconsistent labels found:")
        for text, textname1, label1, textname2, label2 in inconsistencies:
            print(f"Text: '{text}', List1: ({textname1}, {label1}), List2: ({textname2}, {label2})")

    if not_found:
        print("\nTexts not found in both lists:")
        for text, textname, label, missing_from in not_found:
            print(f"Text: '{text}', ({textname}, {label}) not found in {missing_from}")

    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Required accuracy: {required_accuracy:.2%}")

    if accuracy >= required_accuracy and not not_found:
        print("Classifier meets or exceeds the required accuracy and all texts are present in both lists.")
        return True
    else:
        print("Classifier does not meet the required accuracy or there are missing texts.")
        return False
    
    
def save_train_set(test_label_dicts, output_dir):
    with open(output_dir, "w") as f:
        for d in test_label_dicts:
            json_obj = {
                "text": d["paragraph"],
                "label": d["label"]
            }
            f.write(json.dumps(json_obj) + "\n")
        
        
def fix_text(to_replace_arr, text):
    for tup in to_replace_arr:
        text = text.replace(tup[0], tup[1])
    return text