import json
import os
from sys import argv
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.abspath('../../augmentoolkit/utils'))

from load_dataset import load_dataset

classifier_folder_name = argv[1]
PREDICTION_BATCH_SIZE = 10
OUTPUT_FILE_PATH = "./predicted_output.jsonl"


classifier_path_dir = os.path.join("./place_classifier_folder_here", classifier_folder_name)

model = AutoModelForSequenceClassification.from_pretrained(classifier_path_dir, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(classifier_path_dir)

def predict(text, prediction_batch_size=PREDICTION_BATCH_SIZE):
    outputs = []
    
    # Calculate the total number of batches
    total_batches = (len(text) + prediction_batch_size - 1) // prediction_batch_size
    
    # Process text in batch_size groups with tqdm progress bar
    for i in range(0, len(text), prediction_batch_size):
        batch = text[i:i+prediction_batch_size]
        encoding = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
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
        
        batch_predictions = predict(batch_texts)
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