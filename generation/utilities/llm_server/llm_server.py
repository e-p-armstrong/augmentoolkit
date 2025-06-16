# A simple inference server that serves a custom model with its prompt!
# This makes a good reference for code if you are deploying a custom model yourself for production!
# It's also a handy simple endpoint for using models in quick hcat interfaces with all the message truncation logic etc. already built in.
# If you want an openai-compatible server, just run llama.cpp/llama-server -m [pathtoyourmodel] -c [yourcontextlength]
# and make sure taht you provide your system prompt in your openai api calls.


# Requirements --
# Config file, which points to
# prompt.txt
# template.jinja
# and a .gguf model file

# it basically wraps the ismple chat interface thing + the code that starts the llama.cpp server but as a fastapi thingthat routes any non- chat/completions request to the llama.cpp server


from asyncio import Timeout
import asyncio
import os
import platform
import subprocess
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
from nltk.tokenize import word_tokenize
import nltk
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from generation.core_components.chunking import count_tokens_specific_model
from generation.core_components.simple_chat_loop import (
    format_messages_into_string,
    get_stop_tokens,
    get_assistant_prefix,
)
from redis_config import set_progress


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]


try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)


async def llm_server(
    prompt_path,
    template_path,
    gguf_model_path,
    context_length,
    llama_path="./llama.cpp",  # customizable llama.cpp path
    port=8003,
    task_id=None,
    **kwargs,
):

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Get parent directory of gguf model path
    model_dir = os.path.dirname(gguf_model_path)

    # Initialize model-specific token counting
    try:
        count_tokens = count_tokens_specific_model(
            model_dir
        )  # Assumes that tokenizer and sucha re still inside the same dir as the saved gguf model.
    except Exception as e:
        print(e)
        print(
            "\n\nYou probably deleted the tokenizer and other such things from the model directory after you got your quantized model. However, the model tokenizer is used to count tokens before sending it off to llama.cpp, so you shouldn't have done that. You can delete model files (*.safetensors) to save space, but leave the tokenizer alone."
        )
        print(
            "To fix this, re-run the datagen pipeline that produced this model. It won't re-train, but it will re-download"
        )
        raise

    # llama.cpp
    if not os.path.exists(llama_path):
        print("llama.cpp directory not found. Cloning repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", llama_path],
            check=True
        )
        subprocess.run(
            ["git", "checkout", "3cb203c89f60483e349f841684173446ed23c28f"],
            cwd=llama_path,
            check=True
        )

        # Check if llama-server exists
        llama_server_path = os.path.join(llama_path, "build", "bin", "llama-server")
        if platform.system() == "Windows":
            llama_server_path += ".exe"

        if not os.path.exists(llama_server_path):
            print("llama-server not found. Building llama.cpp...")

            # Detect if NVIDIA GPU is available
            has_nvidia_gpu = False
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=False, text=True)
                has_nvidia_gpu = result.returncode == 0
            except FileNotFoundError:
                has_nvidia_gpu = False

            # Build with appropriate flags
            build_cmd = ["cmake", "-B", "build"]
            if has_nvidia_gpu:
                build_cmd.append("-DGGML_CUDA=ON")
                print("NVIDIA GPU detected. Building with CUDA support...")
            else:
                print("No NVIDIA GPU detected. Building CPU-only version...")

            # Run cmake configure
            subprocess.run(build_cmd, cwd=llama_path, check=True)

            # Build the project
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=llama_path,
                check=True,
            )

    # Build llama-server path
    llama_server_path = os.path.join(llama_path, "build", "bin", "llama-server")
    if platform.system() == "Windows":
        llama_server_path += ".exe"

    # Start llama-server in background
    print(f"Starting llama-server with model: {gguf_model_path}")
    server_cmd = [llama_server_path, "-m", gguf_model_path, "-c", str(context_length)]
    server_process = subprocess.Popen(server_cmd)
    print(f"Started llama-server with PID: {server_process.pid}")

    try:
        # Give the server a moment to start up
        print("Waiting for llama-server to load model...")
        time.sleep(15)  # Increased wait time for model loading.

        # Check if the server process has terminated, which indicates a startup failure
        poll = server_process.poll()
        if poll is not None:
            print("\n---")
            print("ERROR: The llama-server process failed to start.")
            print(
                "This commonly happens if the model file is corrupt or incompatible with your version of llama.cpp."
            )
            print("Please check the llama-server logs above for the specific error.")
            print("---\n")
            # We raise an error to stop the script from continuing with a non-functional server
            raise RuntimeError(f"llama-server exited with code {poll}.")

        print("llama-server seems to be running.")
        engine = EngineWrapper(
            api_key="Notused!We are local",
            base_url="http://127.0.0.1:8080/v1",
            mode="api",
            model="itmattersnot",
        )
        stop_token = get_stop_tokens(template)
        assistant_prefix = get_assistant_prefix(template)

        print("Your stop token is:")
        print(stop_token)
        print("Your assistant prefix is:")
        print(repr(assistant_prefix))

        app = FastAPI(title="Augmentoolkit Custom Model API Server", version="0.1.0")

        # Add CORS middleware to handle cross-origin requests from the frontend
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify exact origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "message": "OpenAI Chat Server is running"}

        @app.post("/generate")
        async def submit_chat_to_server(request: ChatRequest):
            messages = request.messages

            # first, if messages contains a system prompt, append the prompt
            # else, create it

            if len(messages) == 0:
                raise HTTPException(400, "Empty messages!")

            if messages[0]["role"] == "system":
                messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
            else:
                # Insert system prompt if not present
                messages.insert(0, {"role": "system", "content": prompt})

            usr_string = messages.pop()["content"]

            total_tokens = count_tokens(usr_string)
            for msg in messages:
                msg_tokens = count_tokens(msg["content"])
                total_tokens = total_tokens + msg_tokens

            shown_messages = messages.copy()

            while total_tokens >= context_length:
                if (
                    not len(shown_messages) == 1
                ):  # if we are down to the system prompt and the most recent user message and are over, then we have a dire problem
                    removed_message = shown_messages.pop(
                        1
                    )  # remove the message after the system prompt
                    removed_tokens = count_tokens(removed_message["content"])
                    total_tokens = total_tokens - removed_tokens
                else:
                    print(
                        f"\n\nNO MESSAGE SENT -- User message too long, user message + system message = {total_tokens} which is > {context_length}\n\n"
                    )
                    raise HTTPException(400, "Too long user message!")

            usr_message = {"role": "user", "content": usr_string}
            shown_messages.append(usr_message)
            message_string = format_messages_into_string(
                shown_messages, prompt_template=template
            )

            # Add assistant prefix for generation prefilling
            prefilled_prompt = message_string + assistant_prefix

            response, timeout = await engine.submit_completion(
                prompt=prefilled_prompt,
                sampling_params={
                    "temperature": 0.4,  # yes, it's not greedy. We want a good token probability distribtion for quality outputs.
                    "min_p": 0.3,
                    "top_p": 0.9,
                    "stop": stop_token,
                },  # sampling params should really be a list of dicts param_name: value. Because that way you could control the order in a nice way.
                return_completion_only=True,
            )
            # TODO before streaming, I wonder if we can show a spinner while we wait...

            return response  # the string

        @app.post("/generate-stream")
        async def submit_chat_to_server_streaming(request: ChatRequest):
            """Streaming version of the /generate endpoint that returns chunks as they're generated"""
            messages = request.messages

            # first, if messages contains a system prompt, append the prompt
            # else, create it

            if len(messages) == 0:
                raise HTTPException(400, "Empty messages!")

            if messages[0]["role"] == "system":
                messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
            else:
                # Insert system prompt if not present
                messages.insert(0, {"role": "system", "content": prompt})

            usr_string = messages.pop()["content"]

            total_tokens = count_tokens(usr_string)
            for msg in messages:
                msg_tokens = count_tokens(msg["content"])
                total_tokens = total_tokens + msg_tokens

            shown_messages = messages.copy()

            while total_tokens >= context_length:
                if (
                    not len(shown_messages) == 1
                ):  # if we are down to the system prompt and the most recent user message and are over, then we have a dire problem
                    removed_message = shown_messages.pop(
                        1
                    )  # remove the message after the system prompt
                    removed_tokens = count_tokens(removed_message["content"])
                    total_tokens = total_tokens - removed_tokens
                else:
                    print(
                        f"\n\nNO MESSAGE SENT -- User message too long, user message + system message = {total_tokens} which is > {context_length}\n\n"
                    )
                    raise HTTPException(400, "Too long user message!")

            usr_message = {"role": "user", "content": usr_string}
            shown_messages.append(usr_message)
            message_string = format_messages_into_string(
                shown_messages, prompt_template=template
            )

            # Add assistant prefix for generation prefilling
            prefilled_prompt = message_string + assistant_prefix

            # Create the streaming generator
            async def generate_stream():
                try:
                    async for chunk in engine.submit_completion_streaming(
                        prompt=prefilled_prompt,
                        sampling_params={
                            "temperature": 0.4,
                            "min_p": 0.3,
                            "top_p": 0.9,
                            "stop": stop_token,
                        },
                        return_completion_only=True,
                    ):
                        yield chunk
                except Exception as e:
                    # Send error as final chunk
                    yield f"data: {{\"error\": \"{str(e)}\", \"done\": true}}\n\n"

            return StreamingResponse(
                generate_stream(), 
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                }
            )

        set_progress(
            task_id=task_id,
            progress=1.0,
            message="Server is running! Navigate over to the chat window and you can interact with your model.",
        )

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use Server API
            config = uvicorn.Config(app=app, port=port, host="0.0.0.0")
            server = uvicorn.Server(config)
            await server.serve()
        except RuntimeError:
            # No event loop running, use the synchronous API
            uvicorn.run(app, port=port, host="0.0.0.0")
        # uvicorn.run(app, port=port)

    finally:
        # Clean up the llama-server process
        if server_process:
            print("Terminating llama-server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("llama-server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Force killing llama-server...")
                server_process.kill()
                server_process.wait()
                print("llama-server force killed")
