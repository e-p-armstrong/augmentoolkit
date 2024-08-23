import json
import logging
import gradio as gr
import yaml
import subprocess
import sys
import os
from gradio_log import Log
import chardet

logging.basicConfig(level=logging.DEBUG)

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
        group = []
        for key2 in config[key1]:
            group.append({'path': [key1, key2], 'label': key2, 'value': config[key1][key2]})
        components.append(group)
    logging.debug(f"Created components: {components}")
    return components

def update_config_display(folder, path):
    logging.debug(f"Updating config display for {folder} with path {path}")
    if os.path.exists(path):
        # config = read_yaml_file(path)
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logging.debug(f"Config contents: {config}")
        components = create_config_components(config)
        logging.debug(f"Created components: {components}")
        return gr.update(visible=True), gr.update(visible=False), components
    else:
        logging.warning(f"Config file not found: {path}")
        return gr.update(visible=False), gr.update(visible=True, value=f"Warning: Config file not found for {folder}"), []

def run(pipeline):
    try:
        for item in pipeline:
            folder, config_path = item.split(':', 1)
            env = os.environ.copy()
            env["CONFIG_PATH"] = config_path
            with open("log.txt", "a", encoding='utf-8') as log_file:
                subprocess.run([sys.executable, os.path.join(folder, "processing.py")], 
                               stdout=log_file, stderr=log_file, text=True, env=env, cwd=folder)
    except Exception as e:
        print(f"Error: {e}")

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

    with gr.Row():
        btn = gr.Button("Start", elem_id="start")

    config_display = gr.Column(visible=False)
    config_warning = gr.Markdown(visible=False)

    def update_pipeline(pipeline_order, config_paths):
        paths = [path.strip() for path in config_paths.split(',')]
        if len(pipeline_order) != len(paths):
            return gr.update(value=pipeline_order), "Error: Number of config paths doesn't match selected folders"
        return gr.update(value=pipeline_order), ""

    pipeline.change(update_pipeline, [pipeline, config_paths], [pipeline, config_warning])

    def prepare_run(pipeline_order, config_paths):
        paths = [path.strip() for path in config_paths.split(',')]
        if len(pipeline_order) != len(paths):
            return
        pipeline = [f"{folder}:{path}" for folder, path in zip(pipeline_order, paths)]
        run(pipeline)

    btn.click(
        fn=prepare_run,
        inputs=[pipeline, config_paths],
        outputs=[],
        js='(e) => { document.querySelector("#log").classList.add("display") }'
    )

    for script in valid_scripts:
        with gr.Tab(script):
            config_path = gr.Textbox(label=f"Config Path for {script}", placeholder=f"{script}/config.yaml")
            config_components = []
            
            def update_config(script, path):
                config_display, config_warning, components = update_config_display(script, path)
                return [
                    config_display,
                    config_warning,
                    *[gr.update(visible=True, value=comp['value']) for group in components for comp in group]
                ]

            config_path.change(
                update_config,
                inputs=[gr.State(script), config_path],
                outputs=[config_display, config_warning, *config_components]
            )

            # config_path.change(
            #     update_config,
            #     inputs=[gr.State(script), config_path],
            #     outputs=[config_display, config_warning, gr.State(config_components)]
            # )

demo.launch()