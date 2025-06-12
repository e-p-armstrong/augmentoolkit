# Basic Inference Server

This simple **utility pipeline** serves a model trained by Augmentoolkit for inference, using llama.cpp as a backend. Due to it using llama.cpp, inference will be much slower than in production (where you would want to use vllm — see the local linux start script for reference) but llama.cpp is conveniently cross-platform, so it is used for this individual chatting usecase.

The config here is incredibly simple.

- `prompt_path:` This is a path to a prompt.txt file. These files are saved at the end of a dataset generation run from Augmentoolkit; they are an example of a system prompt which that model in question was trained with, and with which it will have optimal performance.
- `template_path:` This is a path to a template.txt file. These files are saved at the end of a dataset generation run from Augmentoolkit -- point it at the file for the model you want to run. The chat template can alternatively usually be found inside the tokenizer_config.json of a model. It's in the "chat_template" key.
- `gguf_model_path:` This is the path to the GGUF MODEL FILE (not the directory) that you want to run. However, this model file should be located inside a directory with the tokenizer it was trained on. BAsically, this means that you should point this at the gguf model file inside the model directory that is saved by Augmentoolkit at the end of a raining run.
- `context_length:` The context length of the model. Typically you set this to however many tokens you trained the model with, but honestly if you need more tokens, so long as you aren't exceeding the token counts that the model was PRETRAINED with, you should be fine.
- `llama_path:` The path to your local llama.cpp install
- `port:` what port to run on
- `pipeline:` used by the interface
```