import json

# Function to convert a .txt file to a .jsonl file
# Useful if you want to train on the raw text first to give it broad knowledge, then train on the Augmentoolkit dataset to teach it to answer questions on the subject
def txt_to_single_jsonl(txt_file_path, jsonl_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
        # Read the entire content of the file, preserving whitespace
        file_content = txt_file.read()
        # Create a dictionary with the entire file content
        json_obj = {"text": file_content}
    
    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        # Write the single JSON object to the .jsonl file
        jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# Example usage
txt_file_path = './raw_txt_input/on_war_clausewitz.txt'
jsonl_file_path = './on_war_clausewitz.json'
txt_to_single_jsonl(txt_file_path, jsonl_file_path)

print("Conversion completed.")
