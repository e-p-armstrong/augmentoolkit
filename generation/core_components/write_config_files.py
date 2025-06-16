import yaml
import os
from typing import List, Optional


def write_training_config(
    dataset_paths: List[dict],
    base_model: str,
    output_path: str = "training_config.yaml",
    is_mistral_derived_model: bool = False,
    wandb_project: Optional[str] = None,
    hub_model_id: Optional[str] = None,
    hub_strategy: str = "all_checkpoints",
    **kwargs
) -> str:
    """
    Generate a training configuration YAML file.

    Args:
        dataset_paths: List of dictionaries representing dataset configurations
        base_model: Name of the base model to use
        output_path: Path where the config file will be saved
        is_mistral_derived_model: Whether the model is Mistral-derived
        wandb_project: Weights & Biases project name
        hub_model_id: Hugging Face Hub model ID
        hub_strategy: Hugging Face Hub upload strategy
        **kwargs: Additional keyword arguments

    Returns:
        Path to the generated config file
    """
    # Base configuration
    config = {
        "base_model": base_model,
        "tokenizer_type": "AutoTokenizer",
        "model_type": "AutoModelForCausalLM",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "strict": False,
        "datasets": dataset_paths,
        "dataset_prepared_path": "last_run_prepared",
        "output_dir": "./model-output",
        "seed": 1337,
        "sequence_len": 5000,
        "sample_packing": True,
        "pad_to_sequence_len": False,
        "shuffle_merged_datasets": True,
        "gradient_accumulation_steps": 75,
        "micro_batch_size": 2,
        "eval_batch_size": 1,
        "num_epochs": 12,
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler": "constant",
        "learning_rate": 0.000020,
        "noisy_embedding_alpha": 5,
        "weight_decay": 0,
        "train_on_inputs": kwargs.get("train_on_inputs", False),
        "group_by_length": False,
        "bf16": True,
        "fp16": False,
        "tf32": False,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "xformers_attention": False,
        "flash_attention": True,
        "chat_template": "chatml",
        "auto_resume_from_checkpoints": False,
        "warmup_ratio": 0.1,
        "evals_per_epoch": 1,
        "eval_batch_size": 4,
        "val_set_size": 0.04,
        "saves_per_epoch": 1,
        "eval_sample_packing": False,
        "save_total_limit": 2,
        "special_tokens": {"pad_token": kwargs.get("pad_token", "<unk>")},
        "use_liger_kernel": True,
        "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        "liger_rope": True,
        "liger_rms_norm": True,
        "liger_glu_activation": True,
        "liger_layer_norm": True,
        "liger_fused_linear_cross_entropy": True,
    }

    # Update config with any additional kwargs
    config.update(kwargs)

    # Add optional configurations
    if wandb_project:
        config["wandb_project"] = wandb_project
        config["wandb_entity"] = ""
        config["wandb_watch"] = ""
        config["wandb_run_id"] = ""
        config["wandb_log_model"] = ""

    if hub_model_id:
        config["hub_model_id"] = hub_model_id
        config["hub_strategy"] = hub_strategy

    # Adjust configuration for Mistral-derived models if needed
    if is_mistral_derived_model:
        # Add any Mistral-specific configurations here if needed
        pass

    # Write configuration to file
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def create_completion_dataset(data_files: str, **kwargs) -> dict:
    """
    Create a dataset configuration for completion-formatted data
    """
    return {"path": data_files, "type": "completion", **kwargs}


def create_input_output_dataset(path: str) -> dict:
    """
    Create a dataset configuration for input-output formatted data
    """
    return {
        "path": path,
        "type": "input_output",
    }
