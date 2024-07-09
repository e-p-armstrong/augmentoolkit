import json
import math
import os
from sys import argv
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.abspath('../../augmentoolkit/utils'))

from load_dataset import load_dataset

def head_tail_truncate(text, max_length=510):
    """
    Truncate the text using the head+tail method.
    Keep the first head_length characters and the last (max_length - head_length) characters.
    """
    
    head_length = math.floor(0.2*max_length)
    tail_length = max_length - head_length
    if len(text) <= max_length:
        return text
    return text[:head_length] + text[-(tail_length - head_length):]


classifier_folder_name = argv[1]
PREDICTION_BATCH_SIZE = 10
OUTPUT_FILE_PATH = "./predicted_output.jsonl"
MAX_TEXT_LENGTH = 900 # in characters
HEAD_TAIL_TRUNCATE = True


classifier_path_dir = os.path.join("./place_classifier_folder_here", classifier_folder_name)

model = AutoModelForSequenceClassification.from_pretrained(classifier_path_dir, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(classifier_path_dir)

def predict(text, prediction_batch_size=PREDICTION_BATCH_SIZE, max_length=MAX_TEXT_LENGTH):
    outputs = []
    
    # Calculate the total number of batches
    total_batches = (len(text) + prediction_batch_size - 1) // prediction_batch_size
    
    # Process text in batch_size groups with tqdm progress bar
    for i in range(0, len(text), prediction_batch_size):
        batch = text[i:i+prediction_batch_size]
        
        if HEAD_TAIL_TRUNCATE:
            # Apply head_tail_truncate to each text in the batch
            truncated_batch = [head_tail_truncate(t, max_length) for t in batch]
        else:
            truncated_batch = [t[:max_length] for t in batch]
        
        encoding = tokenizer(truncated_batch, return_tensors='pt', padding=True, truncation=True)
        batch_outputs = model(**encoding)
        batch_predictions = batch_outputs.logits.argmax(-1)
        outputs.extend(batch_predictions.tolist())

    return outputs

dataset_dir = "./place_classification_dataset_here/"
datasets = {}

for filename in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, filename)
    if os.path.isfile(file_path):
        try:
            dataset_name = os.path.splitext(filename)[0]
            datasets[dataset_name] = load_dataset(file_path)
            print(f"Loaded dataset: {dataset_name}")
        except ValueError as e:
            print(f"Error loading {filename}: {str(e)}")

# Print summary of loaded datasets
print("\nLoaded datasets:")
for dataset_name, dataset in datasets.items():
    print(f"{dataset_name}: {len(dataset)} samples")
    print(dataset.head())
    print()

# Function to calculate accuracy
def calculate_accuracy(predictions, labels):
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels)

# Predict and calculate accuracy for each dataset
total_correct = 0
total_samples = 0

for dataset_name, dataset in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    texts = dataset['text'].tolist()
    labels = dataset['label'].tolist()
    
    predictions = []
    correct = 0
    
    # Use tqdm for the outer loop to show overall progress
    for i in tqdm(range(0, len(texts), PREDICTION_BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[i:i+PREDICTION_BATCH_SIZE]
        batch_labels = labels[i:i+PREDICTION_BATCH_SIZE]
        
        batch_predictions = predict(batch_texts, max_length=MAX_TEXT_LENGTH)
        predictions.extend(batch_predictions)
        
        batch_correct = sum(p == l for p, l in zip(batch_predictions, batch_labels))
        correct += batch_correct
        
        # Save predictions to .jsonl file
        with open(OUTPUT_FILE_PATH, 'a') as f:
            for text, pred in zip(batch_texts, batch_predictions):
                json_line = json.dumps({"text": text, "label": int(pred), "dataset": dataset_name})
                f.write(json_line + '\n')
        
        # Calculate and print current accuracy
        current_accuracy = correct / (i + len(batch_texts))
        print(f"\nCurrent accuracy ({i + len(batch_texts)}/{len(texts)}): {current_accuracy:.4f}")
    
    final_accuracy = calculate_accuracy(predictions, labels)
    print(f"\nFinal accuracy for {dataset_name}: {final_accuracy:.4f}")
    
    total_correct += correct
    total_samples += len(texts)

# Print overall accuracy
overall_accuracy = total_correct / total_samples
print(f"\nOverall accuracy across all datasets: {overall_accuracy:.4f}")