from asyncio import Timeout
import asyncio
import os
import platform
import subprocess
import time
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
import uvicorn

from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from generation.core_components.chunking import count_tokens
from generation.core_components.simple_chat_loop import (
    format_messages_into_string,
    get_stop_tokens,
    simple_chat_loop,
)
from redis_config import set_progress


def chat(  # this is useful for chatting with trained models in the command line. It perhaps ought to stream. Perhaps we ought to
    prompt_path,
    template_path,
    gguf_model_path,
    context_length,
    modelid,  # used for the tokenizer
    llama_path="./llama.cpp",  # customizable llama.cpp path
    task_id=None,
    **kwargs,
):

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # llama.cpp
    if not os.path.exists(llama_path):
        print("llama.cpp directory not found. Cloning repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp.git"], check=True
        )
        subprocess.run(
            ["git", "checkout", "3cb203c89f60483e349f841684173446ed23c28f"],
            cwd="llama.cpp",
            check=True,
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
            subprocess.run(build_cmd, cwd=llama_path, check=True, shell=True)

            # Build the project
            subprocess.run(
                ["cmake", "--build", "build", "--config", "Release"],
                cwd=llama_path,
                check=True,
                shell=True,
            )

    # Build llama-server path
    llama_server_path = os.path.join(llama_path, "build", "bin", "llama-server")
    if platform.system() == "Windows":
        llama_server_path += ".exe"

    # Start llama-server in background
    print(f"Starting llama-server with model: {gguf_model_path}")
    server_cmd = [llama_server_path, "-m", gguf_model_path, "-c", str(context_length)]
    server_process = subprocess.Popen(server_cmd, shell=True)
    print(f"Started llama-server with PID: {server_process.pid}")

    try:
        # Give the server a moment to start up
        time.sleep(10)
        set_progress(
            task_id,
            progress=1,
            message="Simple chat loop running! Though this pipeline does not support the interface (use the openai server for that instead) so you should probably be running this through the CLI",
        )
        simple_chat_loop(
            system_prompt=prompt,
            prompt_template=template,
            context_length=context_length,
            finetune_hub_model_id="alpindale/Mistral-7B-v0.2-hf",
        )
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
