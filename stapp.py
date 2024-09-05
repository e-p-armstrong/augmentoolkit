import streamlit as st
import yaml
import os
import subprocess
import threading
import time

from augmentoolkit.utils.make_id import make_id

# Initial version credit: A guy named Sam on Fiverr
# I had to change a decent amount though so it is more collaborative


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

def run_processing_script(folder_path, config_path, project_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    env["CONFIG_PATH"] = config_path
    env["FOLDER_PATH"] = folder_path
    
    process = subprocess.Popen(
        ["python", "run_augmentoolkit.py"],
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process


def stream_output(process, output_area):
    for line in process.stdout:
        output_area.text(line.strip())
    for line in process.stderr:
        output_area.text(f"Error: {line.strip()}", unsafe_allow_html=True)

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
    st.title("Augmentoolkit Runner")
    st.write("This app allows you to run Augmentoolkit pipelines and modify configurations.")

    # Display the available pipeline options in a selectbox
    folder_configs = scan_folders_for_config()
    
    def set_unsaved_changes_made_false(*args, **kwargs):
        st.session_state['unsaved_changes_made'] = False

    st.sidebar.header("Select Pipeline to Run and super_config.yaml")
    pipeline_options = [f"{config['folder']} - {config['config']}" for config in folder_configs]
    selected_pipeline = st.sidebar.selectbox("Choose a pipeline:", pipeline_options, on_change=set_unsaved_changes_made_false)

    # Get the selected pipeline's details
    selected_config = next((config for config in folder_configs if f"{config['folder']} - {config['config']}" == selected_pipeline), None)

    if selected_config:
        st.header("Modify Configurations")
        config_path = os.path.join(selected_config['folder'], selected_config['config'])
        st.subheader(f"Configuration: {config_path}")
        config_data = load_individual_config(config_path)
        
        modified_config = {}
        
        def set_unsaved_changes_made_true(*args, **kwargs):
            st.session_state['unsaved_changes_made'] = True
            
        
        # way it worked before
        # dict with keys, values. Values were strings.
        # Now we have keys which point to a bunch of keys with their own values
        # When changing a value, we have to update that subkey in that key.
        
        
        for key, value in config_data.items():
            st.subheader(f"{key}", divider=True)
            modified_config[key] = value
            for subkey, subvalue in value.items():
                modified_value = st.text_area(f"{subkey}:", subvalue, on_change=set_unsaved_changes_made_true)
                modified_config[key][subkey] = modified_value
                
                
                # modified_config[key] = modified_value
        
        if 'unsaved_changes_made' in st.session_state and st.session_state['unsaved_changes_made']:
            st.warning("Don't forget to save changes!")
        
        # Save updated configurations
        if st.button("Save Configurations"):
            save_individual_config(modified_config, config_path)
            st.success("Configurations saved successfully!")

        if st.button("Run Selected Pipeline"):
            project_root = os.path.dirname(os.path.abspath(__file__))
            process = run_processing_script(selected_config['folder'], config_path, project_root)
            
            # Create a placeholder for the output
            output_area = st.empty()
            
            # Initialize an empty string to store the full output
            full_output = ""
            
            # Custom CSS and JavaScript for auto-scrolling and larger textarea
            custom_css = """
            <style>
            textarea {
                min-height: 500px !important;
                font-family: monospace;
            }
            </style>
            """
            
            auto_scroll_script = """
            <script>
            function scrollTextArea() {
                var textarea = document.querySelector('textarea');
                textarea.scrollTop = textarea.scrollHeight;
            }
            var observer = new MutationObserver(scrollTextArea);
            observer.observe(document.querySelector('textarea'), { childList: true, subtree: true });
            scrollTextArea();
            </script>
            """
            
            st.markdown(custom_css, unsafe_allow_html=True)
            
            # Stream the output
            for line in iter(process.stdout.readline, ''):
                full_output += line
                output_area.markdown(f"""
                {custom_css}
                <textarea style="width:100%;" readonly>{full_output}</textarea>
                {auto_scroll_script}
                """, unsafe_allow_html=True)
            
            # Wait for the process to complete
            process.wait()
            
            if process.returncode == 0:
                st.success("Pipeline completed successfully!")
            else:
                st.error("Pipeline failed. Check the output for details.")
                
if __name__ == "__main__":
    main()