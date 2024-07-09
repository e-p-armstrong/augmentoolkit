import gradio as gr
import yaml
import subprocess
import sys
import os
from gradio_log import Log
import chardet

def read_yaml_file(file_path):
    try:
        # Detect the file encoding
        with open(file_path, 'rb') as raw_file:
            raw_data = raw_file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        # Read the file with the detected encoding
        with open(file_path, "r", encoding=file_encoding) as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

config = {}
config_file_path = 'config.yaml'
config = read_yaml_file(config_file_path)
str = yaml.dump(config)
components = []
for key1 in config:
    group = []
    for key2 in config[key1]:
        group.append({'path': [key1, key2], 'label':key2, 'value':config[key1][key2]})
    components.append(group)

def changed(text, keys):
    global config
    _config = config
    for key in keys[:-1]:
        _config = _config.setdefault(key, {})
    _config[keys[-1]] = text
    return keys

def run(file):
    global config
    try:
        with open(config_file_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, allow_unicode=True)
    except Exception as e:
        print(f"Error writing config file: {e}")
        return

    try:
        env = os.environ.copy()
        with open("log.txt", "w", encoding='utf-8') as log_file:
            subprocess.run([sys.executable, "processing.py"], stdout=log_file, stderr=log_file, text=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

with gr.Blocks(css="#log { padding:0; height: 0; overflow: hidden; } #log.display { padding: 10px 12px; height: auto; overflow: auto; } .gradio-container { max-width: none !important; }") as demo:
    with gr.Row():
        log_file = os.path.abspath("log.txt")
        if not os.path.isfile(log_file):
            open(log_file, 'w').close()
        with open(log_file, "w", encoding='utf-8') as file:
            file.truncate(0)
        log_view = Log(log_file, xterm_font_size=12, dark=True, elem_id='log')
    with gr.Row():
        file = gr.File()
        btn = gr.Button("Start", elem_id="start")
        btn.click(
            fn=run,
            inputs=[file],
            outputs=[],
            js='(e) => { document.querySelector("#log").classList.add("display") }'
        )
    with gr.Row():
        for component in components:
            with gr.Column():
                for item in component:
                    t = gr.Textbox(label=item['label'], value=item['value'], interactive=True)
                    t.change(changed, [t, gr.State(value=item['path'])], [gr.State()])

demo.launch()