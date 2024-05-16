import gradio as gr
import yaml
import subprocess
import os
from gradio_log import Log

config = {}
with open('config.yaml', 'r') as file:
  config = yaml.safe_load(file)
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

def run():
  global config
  with open('config.yaml', 'w') as file:
    yaml.dump(config, file)
  try:
    with open("log.txt", "w") as log_file:
      subprocess.run(["python", "processing.py"], stdout=log_file, stderr=log_file, text=True)
  except subprocess.CalledProcessError as e:
    print(f"Error: {e}")


js = """
document.querySelector("#start").addEventListener("click", (e) => {
  console.log("clicked")
  document.querySelector("#log").classList.add("display")
})
"""

with gr.Blocks(js=js, css="#log { display: none; } #log.display { display: block; } .gradio-container { max-width: none !important; }") as demo:
  with gr.Row():
    log_file = os.path.abspath("log.txt")
    if not os.path.isfile(log_file):
      open(log_file, 'w').close()
    with open(log_file, "w") as file:
      file.truncate(0)
    log_view = Log(log_file, xterm_font_size=12, elem_id='log')
  with gr.Row():
    file = gr.File()
    btn = gr.Button("Start", elem_id="start")
    btn.click(fn=run, inputs=[], outputs=[])
  with gr.Row():
    for component in components:
      print(f"component = {component}")
      with gr.Column():
        for item in component:
          print(f"item={item}")
          t = gr.Textbox(label=item['label'], value=item['value'], interactive=True)
          t.change(changed, [t, gr.State(value=item['path'])], [gr.State()])
demo.launch()
