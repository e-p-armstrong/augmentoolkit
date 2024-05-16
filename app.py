import gradio as gr
import yaml


with open('config.yaml', 'r') as file:
  config = yaml.safe_load(file)
  str = yaml.dump(config)

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

textbox = gr.Textbox(value=str)
demo = gr.Interface(
    fn=greet,
    inputs=[textbox],
    outputs=["text"],
)

demo.launch()
