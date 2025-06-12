from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

def merge_lora(base_model_name: str, adapter_model_name: str, save_path: str):
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load and merge model
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_model_name)
    
    model = model.merge_and_unload()
    
    # Save both model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Merged model and tokenizer saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Merge LoRA adapter with base model')
    parser.add_argument('--base-model', type=str, required=True,
                        help='Base model name or path')
    parser.add_argument('--adapter-model', type=str, required=True,
                        help='LoRA adapter model name or path')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path to save the merged model')
    
    args = parser.parse_args()
    merge_lora(args.base_model, args.adapter_model, args.save_path)

if __name__ == "__main__":
    main()
