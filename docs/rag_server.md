# RAG Inference Server

This **utility pipeline** serves a model trained by Augmentoolkit for inference, using llama.cpp as a backend, with the addition of an effective RAG (Retrieval-Augmented Generation) setup. Due to it using llama.cpp, inference will be much slower than in production (where you would want to use vllm — see the local linux start script for reference) but llama.cpp is conveniently cross-platform, so it is used for this individual chatting usecase. 

This pipeline is different from the basic server in that it vectorizes the questions of the quesiton-answer pairs your model was trained on. It uses this in vector search to find paragraphs which are related to similar questions. This additional grounding can help models with details that are rarely mentioned -- and, of course, the model is trained to fall back to its parametric memory if the RAG fails.

The config here borrows heavily from the [basic server](./basic_server.md)

- `prompt_path:` This is a path to a prompt.txt file. These files are saved at the end of a dataset generation run from Augmentoolkit; they are an example of a system prompt which that model in question was trained with, and with which it will have optimal performance.
- `template_path:` This is a path to a template.txt file. These files are saved at the end of a dataset generation run from Augmentoolkit -- point it at the file for the model you want to run. The chat template can alternatively usually be found inside the tokenizer_config.json of a model.
- `gguf_model_path:` This is the path to the GGUF MODEL FILE (not the directory) that you want to run. However, this model file should be located inside a directory with the tokenizer it was trained on. BAsically, this means that you should point this at the gguf model file inside the model directory that is saved by Augmentoolkit at the end of a raining run.
- `context_length:` The context length of the model. Typically you set this to however many tokens you trained the model with, but honestly if you need more tokens, so long as you aren't exceeding the token counts that the model was PRETRAINED with, you should be fine.
- `llama_path:` The path to your local llama.cpp install
- `port:` what port to run on
- `question_chunk_size:` the maximum chunk size fr a single embedded thing. Usually questions will be a single chunk unless you set this very low. Still, you may configure it.
- `top_k:` Number of RAG documents to retrieve per query.
- `cache_dir:` Where to save the RAG dataset to when it is prepared (the dataset is cached so that future runs don't need to re-embed everything.)
- `collection_name` The name of the collection within the Chroma database. You probably don't need to touch this. Why did I make this configurable? Don't remember, but in case there was a reason, you can change this. But again, probably don't have to.
- `max_shrink_iterations:` The number of times to shrink the context of the retrieved chunks in order to make it all fit within the context limit.
- `pipeline:` this setting is used by the interface to know what pipeline this config is associated with.
```