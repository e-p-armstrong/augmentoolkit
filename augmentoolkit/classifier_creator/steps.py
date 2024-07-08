import json
import logging
import os
import traceback
import yaml

from augmentoolkit.generation_functions.generation_step_class import GenerationStep
from augmentoolkit.utils.make_id import make_id
from augmentoolkit.utils.write_output_to_file import write_output_to_file


with open("./config_classifier.yaml", "r") as f: # different yaml file for different pipes
        config = yaml.safe_load(f)
        
COMPLETION_MODE = config["SYSTEM"]["COMPLETION_MODE"]

### PROMPT FUNC: Rules Creator

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
    
    classes_list_str = format_class_list(classes_list)
    
    try:        
        
        result, full_output = await rules_creator.generate({
            "classes_desc": classes_desc,
            "class_list": classes_list_str,
        })
        
        id = make_id()
        
        write_output_to_file(full_output, os.path.join(config["PATH"]["OUTPUT"], "rules_creation_generation"), id) # TODO move output dir to processing
        
        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
    

###


### PROMPT FUNC: Label Creator

def parse_labels(rules_str):
    return rules_str # TODO # NOTE result should probably be in format, (text, textname, labelstr)

async def create_label(idx, inp, engine_wrapper=None, output_dir=None, output_list=None, rules=None):
    prompt_path = "create_labels_for_chunk"
    inp_text = inp[0]
    
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
        completion_mode=COMPLETION_MODE,
        retries=1,
        engine_wrapper=engine_wrapper,
        logging_level=logging.INFO,
        output_processor=parse_labels,
        prompt_folder=config["PATH"]["PROMPTS"],
        default_prompt_folder=config["PATH"]["DEFAULT_PROMPTS"],
        use_stop=config["SYSTEM"]["STOP"]
    )
    
    try:
        result, full_output = await rules_creator.generate({
            "rules": rules,
            "inp_text": inp_text,
        })
        
        id = make_id()
        
        write_output_to_file(full_output, os.path.join(output_dir, "rules_creation_generation"), id)
        
        output_list.append(result)
    except Exception as e:
        print(e)
        traceback.print_exc()
    
###


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline

def train_classifier(text_label_tuples, classifier_counter, output_dir):
    
    # First, save the tuples to a file with only the relevant info so that we can load them as a dataset
    
    # data will be saved as json lines
    
    
    # TODO
    classifier_counter += 1
    os.makedirs(os.path.join(output_dir, "datasets"), exist_ok=True)
    path_to_dataset = os.path.join(output_dir, "datasets", f"dataset_{classifier_counter}.jsonl")
    with open(path_to_dataset, "w") as f:
        for tup in text_label_tuples:
            json_obj = {
                "text": tup[0],
                "label": tup[2]
            }
            f.write(json.dumps(json_obj) + "\n")

    ### TRAINING CODE
    dataset = load_dataset("json", data_files=path_to_dataset)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
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
    
    new_model = pipeline(model=classifier_output_path)
    return new_model
    
def run_classifier(idx, inp, model, output_dir, output_list): # model is a pipeline
    try:
        output = model(inp[0])[0]["label"]
        
        id = make_id()
        with open(os.path.join(output_dir, f"{id}.json")) as f:
            f.write(json.dumps({
                "text": inp[0],
                "label": output
            }))
        
        out_tup = (inp[0], inp[1], output)
        output_list.append(out_tup)
    except Exception as e:
        print(e)
        traceback.print_exc()

def all_labels_same(truth_labels, classifier_labels):
    # Create dictionaries for fast lookup
    dict1 = {text: (textname, label) for text, textname, label in truth_labels}
    dict2 = {text: (textname, label) for text, textname, label in classifier_labels}

    inconsistencies = []
    not_found = []

    # Check consistency and existence
    for text, (textname1, label1) in dict1.items():
        if text in dict2:
            textname2, label2 = dict2[text]
            if label1 != label2:
                inconsistencies.append((text, textname1, label1, textname2, label2))
        else:
            not_found.append((text, textname1, label1, "list2"))

    # Check for texts in list2 not in list1
    for text, (textname2, label2) in dict2.items():
        if text not in dict1:
            not_found.append((text, textname2, label2, "list1"))

    # Print results
    if inconsistencies:
        print("Inconsistent labels found:")
        for text, textname1, label1, textname2, label2 in inconsistencies:
            print(f"Text: '{text}', List1: ({textname1}, {label1}), List2: ({textname2}, {label2})")
        return False

    if not_found:
        print("\nTexts not found in both lists:")
        for text, textname, label, missing_from in not_found:
            print(f"Text: '{text}', ({textname}, {label}) not found in {missing_from}")
        return False # this is probably a bad fuckup somewhere. NOTE, need to add a substantial number of retries to things that 

    if not inconsistencies and not not_found:
        print("All labels are consistent and all texts are present in both lists.")
        return True
    
    
def save_train_set(test_label_tuples, output_dir):
    with open(output_dir, "w") as f:
        for tup in test_label_tuples:
            json_obj = {
                "text": tup[0],
                "label": tup[2]
            }
            f.write(json.dumps(json_obj) + "\n")
        