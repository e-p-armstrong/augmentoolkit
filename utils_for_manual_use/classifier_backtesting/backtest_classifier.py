import os
from sys import argv
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
sys.path.append(os.path.abspath('../../augmentoolkit/utils'))

from load_dataset import load_dataset

classifier_folder_name = argv[1]

classifier_path_dir = os.path.join("./place_classifier_folder_here", classifier_folder_name)

model = AutoModelForSequenceClassification.from_pretrained(classifier_path_dir, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(classifier_path_dir)

def predict(text, prediction_batch_size=100):
        outputs = []
        
        # Calculate the total number of batches
        total_batches = (len(text) + prediction_batch_size - 1) // prediction_batch_size
        
        # Process text in batch_size groups with tqdm progress bar
        for i in tqdm(range(0, len(text), prediction_batch_size), total=total_batches, desc="Predicting"):
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
    print(dataset)
    
    