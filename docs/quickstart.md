# Quickstart

This isn't quite as "quick" as the simple start scripts shown at the start, but it's still pretty fast.

**Local inference has separate start scripts that set up the LLM for datagen locally for you. Do not use the commands immediately below if you want to do things locally.**

Here are those scripts again for reference, because they are the recommended way to start:

### MacOS (interface)
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash macos.sh # NOTE: Will attempt to install valkey via brew if not found.
```

MacOS has an option for local inference, which will set up the dataset generation model automatically. Note that currently dataset generation is very slow on MacOS due to Llama.cpp limitations. I am considering solutions like MLX and would welcome a PR here as MLX is not something I've used yet.
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash local_macos.sh
```
API inference is recommended for Mac for now.

### Linux (interface)
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
# NOTE: Requires valkey-server (or redis-server) to be installed and running. If you don't have it, the script will clone Valkey and build it from source.
# The script will check and exit if it's not found/active.
# Install example (Debian/Ubuntu): sudo apt install valkey-server && sudo systemctl start valkey-server
bash linux.sh
```

### Windows (interface)
Use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and then run the Linux command in your Linux terminal on Windows. Or use the CLI.

### Local Datagen

Linux offers fast and effective local dataset generation using vLLM. You can pick what model you want to use. 

> [!NOTE]
> 
> **On systems with less VRAM that cannot run the FP16 7b datagen model, you should give the argument "small" to the script. You will also have to change the model name to the quantized model (`Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit`) in your config.**

See the comments in this runnable thing for how the arguments to `local_linux.sh` work.
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash local_linux.sh small # runs quantized model
# bash local_linux.sh # defaults to "normal"
# bash local_linux.sh normal # runs FP16 model (fastest and highest quality but requires more powerful computer)
# bash local_linux.sh deepseek-ai/DeepSeek-R1 # or you can pass in any model name you want so long as it is on huggingface or is available locally and will work with vLLM
# bash local_linux.sh --help # get help. But you're already here, so you don't need that!
```

> [!IMPORTANT]
>
> If you have multiple GPUs, run `local_linux.sh` with the `--tensor-parallelism N` argument. N == number of GPUs you have (even). So: 1, 2, 4, 8... etc.

Mac also has local datagen, but much slower and without the parameter choice of Linux:
```bash
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
bash local_macos.sh
```

## The CLI:

> [!NOTE]
>
> To run **local models** with the CLI, all you have to do is, in another window, run some kind of **local LLM server** and then you can point the **base_url** in your dataset generation config to that local server. Using the [custom 7b dataset generation model](https://huggingface.co/Heralax/Augmentoolkit-DataSpecialist-v0.1) or its [quantized version](https://huggingface.co/Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit) is recommended. **vLLM or Aphrodite is recommended.** Ollama will be comparatively very slow.

### MacOS (CLI)
```bash
# get project
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt

# NOTE you will have to add your API key (and possibly your provider base URL, if you don't want to use deepinfra) to `./external_configs/complete_factual_datagen_example.yaml` 

# and then run the CLI
python run_augmentoolkit.py
```

### Linux (CLI)
```bash
# get project
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt

# NOTE you will have to add your API key (and possibly your provider base URL, if you don't want to use deepinfra) to `./external_configs/complete_factual_datagen_example.yaml`

# and then run the CLI
python run_augmentoolkit.py
```

### Windows (CLI)
```bash
# get project
git clone https://github.com/e-p-armstrong/augmentoolkit.git
cd augmentoolkit
python -m venv .venv
.\.venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt

# NOTE you will have to add your API key (and possibly your provider base URL, if you don't want to use deepinfra) to ./external_configs/complete_factual_datagen_example.yaml

# and then run the CLI
python run_augmentoolkit.py
```


If you REALLY want to go the manual slow way and understand everything you're doing when running the *interface*, then the process is as follows:

In one terminal window, npm install...

## Manual Interface Setup (Advanced)


Possibly, you ended up here because your platform does not have simple package management and we need Valkey.

If you prefer to run each component manually instead of using the start scripts, follow these steps. This requires managing multiple terminal windows.


**Prerequisites:**

*   Git
*   Python 3.9+
*   Node.js and npm
*   Redis (or Valkey, preferably): Huey uses Redis as a message broker. You need to install and run it separately.
    *   **macOS (using Homebrew):** `brew install valkey`, then ensure the service is running (`brew services start valkey`).
    *   **Linux (Debian/Ubuntu):** Install via package manager (`sudo apt update && sudo apt install valkey-server` or `redis-server`) and ensure the service is running (`sudo systemctl start valkey-server` or `redis-server`). The `linux.sh` script checks for this.
    *   **Windows:** Set up WSL (Windows Subsystem for Linux) and then clone Augmentoolkit there and run one of the Linux start scripts. **You'll need to manually open up the localhost url in your browser of choice after it runs**

**Steps:**

1.  **Get the Code:**
    ```bash
    git clone https://github.com/e-p-armstrong/augmentoolkit.git
    cd augmentoolkit
    ```

2.  **Terminal 1: Setup Python Environment & Run Huey Worker**
    *   Create and activate a virtual environment:
        *   macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
        *   Windows: `python -m venv .venv && .\.venv\Scripts\activate`
    *   Install dependencies (uv is recommended for speed):
        ```bash
        pip install uv
        uv pip install -r requirements.txt
        ```
    *   **Ensure Redis is running** (see Prerequisites).
    *   Start the Huey worker (leave this running):
        ```bash
        huey_consumer tasks.huey
        ```

3.  **Terminal 2: Run API Server**
    *   Navigate to the project directory: `cd augmentoolkit`
    *   Activate the virtual environment:
        *   macOS/Linux: `source .venv/bin/activate`
        *   Windows: `.\.venv\Scripts\activate`
    *   Start the Uvicorn server (leave this running):
        ```bash
        uvicorn api:app --host 0.0.0.0 --port 8000
        ```

4.  **Terminal 3: Build & Serve Frontend**
    *   Navigate to the frontend directory: `cd augmentoolkit/atk-interface`
    *   Install frontend dependencies:
        ```bash
        npm install
        ```
    *   Build the frontend application:
        ```bash
        npm run build
        ```
    *   Serve the built frontend (leave this running):
        ```bash
        # Installs serve if needed and serves the dist directory
        npx serve -s dist --listen 5173
        ```

5.  **Access the Interface:**
    Open your web browser and navigate to `http://localhost:5173`.

You should now have the API server running on port 8000, the Huey worker processing tasks via Redis, and the frontend served on port 5173. **Be sure nothing is running on these ports before running the script!** You can check if something is running on a port with the command `lsof -i :port`. So, `lsof -i :8000` for instance.

Once Augmentoolkit is running you'll probably want to start by running the [Complete Factual Datagen pipeline](/docs/complete_factual_datagen.md). There is a good started config in `external_configs/_START_HERE_complete_factual.yaml`.