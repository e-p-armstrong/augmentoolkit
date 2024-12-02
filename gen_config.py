import os
import yaml
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate the config file to generate conversations based datasets in the specified language.")
parser.add_argument("--language", type=str, default = "English", help="Specify the language to overwrite in the config.yaml (e.g., 'japanese').")
args = parser.parse_args()
def overwrite_language_in_config_file(config_file_path, language):
    if not os.path.exists(config_file_path):
        print(f"Error: Config file '{config_file_path}' does not exist.")
        return

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Update the relevant sections of the config file
    system_settings = config.get("SYSTEM", {})

    if "CONVERSATION_INSTRUCTIONS" in system_settings:
        system_settings["CONVERSATION_INSTRUCTIONS"] = (
            f"For this conversation, you are generating a chat between "
            f"a generalist, generic AI assistant, and a human in {language}. "
            f"Please remember that you are generating conversations between "
            f"{language} generalist, {language} generic AI assistant, and a {language} human."
        )

    if "FINAL_ASSISTANT_PROMPTS_NO_RAG" in system_settings:
        system_settings["FINAL_ASSISTANT_PROMPTS_NO_RAG"] = [
            f"You are a {language.capitalize()} helpful AI assistant.",
            f"You are A VASTLY {language.capitalize()} intelligent ARTIFICIAL INTELLIGENCE "
            f"with DOMAIN-EXPERT KNOWLEDGE from a variety of fields.",
            f"USE your knowledge to be helpful and truthfully answer questions in "
            f"{language.capitalize()} about the world. You will generate the questions and answers "
            f"in {language.capitalize()} language."
        ]

    if "FINAL_ASSISTANT_PROMPTS_RAG" in system_settings:
        system_settings["FINAL_ASSISTANT_PROMPTS_RAG"] = [
            f'You are a {language.capitalize()} helpful AI assistant. I will share some knowledge with you:\n\n'
            '{data}',
            '{data}\n\n'
            f'You are an AI domain expert. Answer questions in {language.capitalize()}',
            f'You are an AI with vast knowledge. Here is some potentially-relevant context:\n\n'
            '{data}\n\n'
            f'Answer questions in {language.capitalize()} according to the context provided.'
        ]

    # Write the updated config back to the file
    with open(config_file_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    print(f"Updated language in config file: {config_file_path}")

def main():
    language = args.language
    overwrite_language_in_config_file('./original/config.yaml', language)

if __name__ == "__main__":
    main()
