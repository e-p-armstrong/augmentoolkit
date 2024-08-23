import gradio as gr
import yaml
import subprocess
import sys
import os
from gradio_log import Log
import chardet

def read_yaml_file(file_path):
    try:
        with open(file_path, 'rb') as raw_file:
            raw_data = raw_file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        with open(file_path, "r", encoding=file_encoding) as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

def scan_for_valid_scripts():
    valid_scripts = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for item in os.listdir(current_dir):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, "processing.py")):
            valid_scripts.append(item)
    return valid_scripts

def create_config_components(config):
    components = []
    for key1 in config:
        for key2 in config[key1]:
            components.append(gr.Textbox(label=f"{key1} - {key2}", value=str(config[key1][key2]), interactive=True))
    return components

def create_placeholder_components(max_components=20):
    return [gr.Textbox(visible=False) for _ in range(max_components)]

def update_config_display(folder, path):
    if os.path.exists(path):
        config = read_yaml_file(path)
        components = create_config_components(config)
        return [gr.update(visible=True), gr.update(visible=False)] + components
    else:
        return [gr.update(visible=False), gr.update(visible=True, value=f"Warning: Config file not found for {folder}")] + [gr.update(visible=False) for _ in range(20)]

def run(pipeline, *all_config_components):
    updated_configs = {}
    component_index = 0
    for item in pipeline:
        folder, config_path = item.split(':', 1)
        config = read_yaml_file(config_path)
        for key1 in config:
            for key2 in config[key1]:
                config[key1][key2] = all_config_components[component_index].value
                component_index += 1
        updated_configs[folder] = config

    for folder, config in updated_configs.items():
        try:
            env = os.environ.copy()
            env["CONFIG_PATH"] = yaml.dump(config)
            with open("log.txt", "a", encoding='utf-8') as log_file:
                subprocess.run([sys.executable, os.path.join(folder, "processing.py")], 
                               stdout=log_file, stderr=log_file, text=True, env=env, cwd=folder)
        except Exception as e:
            print(f"Error running script in {folder}: {e}")

with gr.Blocks(css="#log { padding:0; height: 0; overflow: hidden; } #log.display { padding: 10px 12px; height: auto; overflow: auto; } .gradio-container { max-width: none !important; }") as demo:
    valid_scripts = scan_for_valid_scripts()

    with gr.Row():
        log_file = os.path.abspath("log.txt")
        if not os.path.isfile(log_file):
            open(log_file, 'w').close()
        with open(log_file, "w", encoding='utf-8') as file:
            file.truncate(0)
        log_view = Log(log_file, xterm_font_size=12, dark=True, elem_id='log')

    with gr.Row():
        pipeline = gr.CheckboxGroup(choices=valid_scripts, label="Pipeline Order")
        config_paths = gr.Textbox(label="Config Paths (comma-separated)", placeholder="folder1/config.yaml, folder2/config_override.yaml")

    config_warning = gr.Markdown(visible=False)

    def update_pipeline(pipeline_order, config_paths):
        paths = [path.strip() for path in config_paths.split(',')]
        if len(pipeline_order) != len(paths):
            return gr.update(value=pipeline_order), "Error: Number of config paths doesn't match selected folders"
        return gr.update(value=pipeline_order), ""

    pipeline.change(update_pipeline, [pipeline, config_paths], [pipeline, config_warning])

    config_components = {}
    for script in valid_scripts:
        with gr.Tab(script):
            config_path = gr.Textbox(label=f"Config Path for {script}", placeholder=f"{script}/config.yaml")
            config_warning = gr.Markdown(visible=False)
            with gr.Column(visible=False) as config_display:
                config_components[script] = create_placeholder_components()
            
            def update_config(script, path):
                return update_config_display(script, path)

            config_path.change(
                update_config,
                inputs=[gr.State(script), config_path],
                outputs=[config_display, config_warning] + config_components[script]
            )

    with gr.Row():
        btn = gr.Button("Start", elem_id="start")

    btn.click(
        fn=run,
        inputs=[pipeline] + [component for script_components in config_components.values() for component in script_components],
        outputs=[],
        js='(e) => { document.querySelector("#log").classList.add("display") }'
    )

demo.launch()