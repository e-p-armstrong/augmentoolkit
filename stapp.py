import streamlit as st
import yaml
import os
import subprocess

# Initial version credit: Sam 


def scan_folders_for_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result = []

    for folder in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        required_files = ["steps.py", "processing.py", "__init__.py"]
        if all(os.path.isfile(os.path.join(folder_path, file)) for file in required_files):
            config_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.yaml') and 'config' in file.lower():
                        config_files.append(os.path.join(root, file))
            
            for config_file in config_files:
                relative_path = os.path.relpath(config_file, folder_path)
                result.append({
                    "folder": folder,
                    "config": relative_path
                })
    
    return result

# Load the YAML config
# def load_yaml_config():
#     with open("super_config.yaml", "r") as f:
#         return yaml.safe_load(f)

# Save the updated config to the YAML file
def save_yaml_config(data, filepath):
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

# Run the processing script and capture the output
def run_processing_script(folder_path, config_path, project_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["CONFIG_PATH"] = config_path
    env["FOLDER_PATH"] = folder_path
    result = subprocess.run(
        ["python", "run_augmentoolkit.py"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr

# Load an individual config file
def load_individual_config(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

# Save an individual config file
def save_individual_config(data, filepath):
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False)



if 'unsaved_changes_made' not in st.session_state:
    st.session_state['unsaved_changes_made'] = False
# Main Streamlit app function
def main():
    print("Prints work!")
    st.write("Session state:")
    st.write(st.session_state)
    # tab1=st.tabs(['Main'])
    # with tab1:
    st.title("Augmentoolkit Runner")
    st.write("This app allows you to run Augmentoolkit pipelines and modify configurations.")

    # Display the available pipeline options in a selectbox
    folder_configs = scan_folders_for_config()

    st.sidebar.header("Select Pipeline to Run and super_config.yaml")
    pipeline_options = [f"{config['folder']} - {config['config']}" for config in folder_configs]
    selected_pipeline = st.sidebar.selectbox("Choose a pipeline:", pipeline_options)

    # Get the selected pipeline's details
    selected_config = next((config for config in folder_configs if f"{config['folder']} - {config['config']}" == selected_pipeline), None)

    if selected_config:
        st.header("Modify Configurations")
        config_path = os.path.join(selected_config['folder'], selected_config['config'])
        st.subheader(f"Configuration: {config_path}")
        config_data = load_individual_config(config_path)
        
        modified_config = {}
        
        def set_unsaved_changes_made_true(*args, **kwargs):
            print("\n\n---Args and kwargs:")
            print(args)
            print(kwargs)
            print("Unsaved changes made!\n---")
            st.session_state['unsaved_changes_made'] = True
            print(st.session_state)
        
        for key, value in config_data.items():
            
            # Display each config parameter as editable text in a cleaner multi-line format
            modified_value = st.text_area(f"{key}:", value, height=200, on_change=set_unsaved_changes_made_true)
            modified_config[key] = modified_value
        
        if 'unsaved_changes_made' in st.session_state and st.session_state['unsaved_changes_made']:
            st.warning("Don't forget to save changes!")
        
        # Save updated configurations
        if st.button("Save Configurations"):
            save_individual_config(modified_config, config_path)
            st.success("Configurations saved successfully!")

        # Run the selected pipeline and capture the output
        if st.button("Run Selected Pipeline"):
            project_root = os.path.dirname(os.path.abspath(__file__))
            stdout, stderr = run_processing_script(selected_config['folder'], config_path, project_root)
            
            # Display the captured output in Streamlit
            if stdout:
                st.text_area("Pipeline Output", value=stdout, height=300)
            if stderr:
                st.text_area("Pipeline Error", value=stderr, height=300)
if __name__ == "__main__":
    main()