#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e # Disabled for manual error checking below

# Parse command line arguments
MODEL_TYPE="normal"
TENSOR_PARALLELISM=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tensor-parallelism)
            TENSOR_PARALLELISM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [MODEL_TYPE] [OPTIONS]"
            echo ""
            echo "MODEL_TYPE options:"
            echo "  normal                   - Use default model (Heralax/Augmentoolkit-DataSpecialist-v0.1)"
            echo "  small                    - Use quantized model (Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit)"
            echo "  path/to/custom/model     - Use custom model path"
            echo ""
            echo "OPTIONS:"
            echo "  --tensor-parallelism INT - Number of tensor parallel processes for vLLM (default: 1)"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use normal model (default)"
            echo "  $0 small                              # Use quantized model"
            echo "  $0 deepseek-ai/DeepSeek-R1            # Use custom model"
            echo "  $0 normal --tensor-parallelism 2      # Use normal model with 2 tensor parallel processes"
            echo "  $0 --tensor-parallelism 4             # Use default model with 4 tensor parallel processes"
            echo ""
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
        *)
            # This is the MODEL_TYPE (positional argument)
            if [ "$MODEL_TYPE" = "normal" ]; then
                MODEL_TYPE="$1"
            else
                echo "Error: Multiple model types specified"
                echo "Use --help for usage information."
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate tensor parallelism value
if ! [[ "$TENSOR_PARALLELISM" =~ ^[0-9]+$ ]] || [ "$TENSOR_PARALLELISM" -lt 1 ]; then
    echo "Error: --tensor-parallelism must be a positive integer (got: $TENSOR_PARALLELISM)"
    exit 1
fi

echo "Starting Augmentoolkit services with model type: $MODEL_TYPE"
echo "Tensor parallelism: $TENSOR_PARALLELISM"
echo "(Use '$0 --help' to see available model options)"

# --- Get script directory and change to it ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to change directory to script location '$SCRIPT_DIR'."
    exit 1
fi
echo "Running in directory: $(pwd)"

# --- Cleanup Function ---
# Stores PIDs of background processes
HUEY_PID=""
UVICORN_PID=""
SERVE_PID="" # Though serve runs in foreground, trap might catch signals before it starts
LLAMA_SERVER_PID="" # PID for llama.cpp server
VALKEY_PID="" # PID for valkey server

cleanup() {
    echo "" # Newline after ^C
    echo "----------------------------------------"
    echo "Signal received, stopping background services..."
    
    # Function to kill a process and wait for it to terminate
    kill_and_wait() {
        local pid=$1
        local name=$2
        local timeout=9
        
        if [ -n "$pid" ] && kill -0 $pid &> /dev/null; then
            echo "Stopping $name (PID $pid)..."
            
            # First try SIGTERM (graceful shutdown)
            kill -TERM $pid 2>/dev/null
            
            # Wait for process to terminate gracefully
            local count=0
            while [ $count -lt $timeout ] && kill -0 $pid &> /dev/null; do
                sleep 1
                count=$((count + 1))
            done
            
            # If still running, force kill with SIGKILL
            if kill -0 $pid &> /dev/null; then
                echo "Process $pid ($name) didn't terminate gracefully, force killing..."
                kill -KILL $pid 2>/dev/null
                sleep 1
            fi
            
            # Final check
            if kill -0 $pid &> /dev/null; then
                echo "WARNING: Failed to stop $name (PID $pid)"
            else
                echo "$name stopped successfully."
            fi
        fi
    }
    
    # Stop processes in reverse order of startup
    kill_and_wait "$SERVE_PID" "frontend serve"
    kill_and_wait "$UVICORN_PID" "Uvicorn API server"
    kill_and_wait "$HUEY_PID" "Huey worker"
    kill_and_wait "$LLAMA_SERVER_PID" "VLLM server"
    kill_and_wait "$VALKEY_PID" "Valkey server"
    
    # Find and kill any remaining huey consumers specific to this Augmentoolkit instance
    echo "Searching for huey consumer processes in current directory ($SCRIPT_DIR)..."
    
    # Use ps with full command line and grep for huey_consumer processes in our specific directory
    # This avoids killing huey consumers from other Augmentoolkit instances or projects
    HUEY_PIDS=$(ps aux | grep -E "huey_consumer.*tasks\.huey" | grep "$SCRIPT_DIR" | grep -v grep | awk '{print $2}')
    
    if [ -n "$HUEY_PIDS" ]; then
        echo "Found huey consumer processes in current directory: $HUEY_PIDS"
        for pid in $HUEY_PIDS; do
            if kill -0 $pid &> /dev/null; then
                echo "Stopping huey consumer from current directory (PID $pid)..."
                kill -TERM $pid 2>/dev/null
                sleep 2
                if kill -0 $pid &> /dev/null; then
                    echo "Force killing huey consumer (PID $pid)..."
                    kill -KILL $pid 2>/dev/null
                fi
            fi
        done
        sleep 1
        # Final check for any remaining huey processes in our directory
        REMAINING_HUEY=$(ps aux | grep -E "huey_consumer.*tasks\.huey" | grep "$SCRIPT_DIR" | grep -v grep | awk '{print $2}')
        if [ -n "$REMAINING_HUEY" ]; then
            echo "WARNING: Some huey consumer processes in current directory may still be running: $REMAINING_HUEY"
        else
            echo "All huey consumer processes from current directory stopped successfully."
        fi
    else
        echo "No additional huey consumer processes found in current directory."
    fi
    
    # Deactivate virtual environment if active
    if command -v deactivate &> /dev/null; then
        echo "Deactivating virtual environment..."
        deactivate
    fi
    
    # Give a moment for any remaining output to flush
    sleep 1
    
    # Clear any remaining output and reset cursor
    echo "Cleanup complete."
    echo "----------------------------------------"
    
    # Ensure cursor is visible and terminal is in normal state
    printf "\033[?25h"  # Show cursor
    printf "\033[0m"    # Reset all attributes
    
    echo "" # Add extra newline for visual clarity before prompt returns
}

# Trap signals (INT: Ctrl+C, TERM: standard kill) and run cleanup function
trap cleanup SIGINT SIGTERM

# --- 1. Python Virtual Environment & Dependencies ---
echo "----------------------------------------"
echo "1 / 9. Setting up Python virtual environment and installing dependencies..."
VENV_DIR=".venv"

# Check if python3.11 command exists (preferred)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    echo "Using Python 3.11: $(python3.11 --version)"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    echo "Using Python: $(python3 --version)"
    if [[ "$PYTHON_VERSION" != "3.11" ]]; then
        echo "WARNING: Python 3.11 is recommended but not found. Using $PYTHON_VERSION instead."
        echo "If you encounter issues, please install Python 3.11 specifically."
    fi
else
    echo "ERROR: Neither python3.11 nor python3 command found."
    echo "Please install Python 3.11 (recommended) or Python 3.x and rerun the script."
    echo "On Ubuntu/Debian, you can install Python 3.11 using:"
    echo "  sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev"
    echo "On RHEL/CentOS/Fedora, you can install Python 3.11 using:"
    echo "  sudo dnf install python3.11 python3.11-devel"
    echo "Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR' using $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create Python virtual environment using $PYTHON_CMD."
        exit 1
    fi
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Python virtual environment."
    exit 1
fi
echo "Virtual environment activated. Python executable: $(which python)"

# --- Check/Install uv ---
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "'uv' command not found. Attempting to install uv using pip..."
    # Ensure pip is available (it should be with python3)
    if ! python -m pip --version &> /dev/null; then
        echo "ERROR: python -m pip command failed. Cannot install uv."
        exit 1
    fi
    python -m pip install uv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install uv using pip. Please install uv manually (e.g., 'pip install uv') and rerun the script."
        exit 1
    fi
    echo "uv installed successfully."
else
    echo "uv found."
fi
# --- End Check/Install uv ---


# Install dependencies using uv
echo "Installing Python dependencies from requirements.txt using uv..."
# Assuming uv is in PATH after the check above and venv is activated
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies using uv. Please check requirements.txt and uv logs."
    exit 1 # Cleanup trap will handle exit
fi
echo "Installing vllm using uv..."
uv pip install vllm
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install vllm using uv."
    exit 1
fi

echo "Python dependencies installed successfully."


# --- Helper Process Logging Setup ---
echo "----------------------------------------"
echo "Setting up helper process logging..."

# Create helper_process_logs directory if it doesn't exist
HELPER_LOGS_DIR="helper_process_logs"
if [ ! -d "$HELPER_LOGS_DIR" ]; then
    echo "Creating helper process logs directory: $HELPER_LOGS_DIR"
    mkdir -p "$HELPER_LOGS_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create helper process logs directory."
        exit 1
    fi
else
    echo "Helper process logs directory already exists: $HELPER_LOGS_DIR"
fi

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VALKEY_LOG_FILE="$HELPER_LOGS_DIR/valkey_server_${TIMESTAMP}.log"
LLAMA_SERVER_LOG_FILE="$HELPER_LOGS_DIR/llama_server_${TIMESTAMP}.log"
HUEY_LOG_FILE="$HELPER_LOGS_DIR/huey_worker_${TIMESTAMP}.log"
UVICORN_LOG_FILE="$HELPER_LOGS_DIR/uvicorn_api_${TIMESTAMP}.log"
FRONTEND_LOG_FILE="$HELPER_LOGS_DIR/frontend_serve_${TIMESTAMP}.log"

echo "Helper process logs will be saved to:"
echo "  Valkey server: $VALKEY_LOG_FILE"
echo "  VLLM server: $LLAMA_SERVER_LOG_FILE"
echo "  Huey worker: $HUEY_LOG_FILE"
echo "  Uvicorn API: $UVICORN_LOG_FILE"
echo "  Frontend serve: $FRONTEND_LOG_FILE"
echo "Helper process logging setup complete."

#### if REDIS_HOST skip
if [ -z "$REDIS_HOST" ]; then
    # --- 1.5 Valkey/Redis Check ---
    echo "----------------------------------------"
    echo "2 / 9. Checking for running Valkey/Redis service..."
    
    SERVER_CMD=""
    SERVICE_NAME=""
    
    # Check for valkey-server first
    if command -v valkey-server &> /dev/null; then
        SERVER_CMD="valkey-server"
        SERVICE_NAME="valkey-server"
    # Fallback to redis-server
    elif command -v redis-server &> /dev/null; then
        SERVER_CMD="redis-server"
        SERVICE_NAME="redis-server"
    else
        # Check if something is listening on the default Valkey/Redis port (6379)
        if command -v ss &> /dev/null; then
            PORT_CHECK_CMD="ss -ltn"
        else
            PORT_CHECK_CMD="netstat -ltn"
        fi
    
        if $PORT_CHECK_CMD | grep -q ':6379'; then
            echo "WARNING: Neither 'valkey-server' nor 'redis-server' found in PATH, but something is listening on port 6379."
            echo "Assuming a compatible Valkey/Redis server is running (possibly built from source or custom install)."
        else
            echo "Neither 'valkey-server' nor 'redis-server' found in PATH."
            echo "Building Valkey from source..."
            
            # Check if valkey directory already exists
            if [ ! -d "valkey" ]; then
                echo "Cloning Valkey repository..."
                git clone https://github.com/valkey-io/valkey.git
                if [ $? -ne 0 ]; then
                    echo "ERROR: Failed to clone Valkey repository. Please check your internet connection and git installation."
                    exit 1
                fi
            else
                echo "Valkey directory already exists, skipping clone."
            fi
            
            # Only build if valkey-server doesn't exist
            if [ ! -f "valkey/src/valkey-server" ]; then
                echo "Building Valkey..."
                cd valkey
                make
                if [ $? -ne 0 ]; then
                    echo "ERROR: Failed to build Valkey. Please check build dependencies (gcc, make, etc.)."
                    cd "$SCRIPT_DIR"
                    exit 1
                fi
                cd "$SCRIPT_DIR"
            else
                echo "Valkey already built, skipping build."
            fi
            
            echo "Starting Valkey server in background..."
            echo "Output will be logged to: $VALKEY_LOG_FILE"
            cd valkey/src
            ./valkey-server > "../../$VALKEY_LOG_FILE" 2>&1 &
            VALKEY_PID=$!
            cd "$SCRIPT_DIR"
            
            # Give it a moment to start/fail
            sleep 4
            if ! kill -0 $VALKEY_PID &> /dev/null; then
                echo "ERROR: Valkey server (PID $VALKEY_PID) failed to start or crashed immediately."
                echo "Check the log file for details: $VALKEY_LOG_FILE"
                exit 1
            fi
            echo "Valkey server started successfully with PID $VALKEY_PID."
            
            # Set SERVER_CMD and SERVICE_NAME for consistency with the rest of the script
            SERVER_CMD="./valkey/src/valkey-server"
            SERVICE_NAME="valkey-server"
        fi
    fi
    echo "Found $SERVER_CMD."
    
    # If we didn't just start Valkey server locally, perform service checks
    if [ -z "$VALKEY_PID" ]; then
        # Check if systemctl exists and is usable
        if command -v systemctl &> /dev/null && systemctl status &> /dev/null; then
            # systemctl seems to be working, so let's use it
            if ! systemctl is-active --quiet $SERVICE_NAME; then
                echo "WARNING: The $SERVICE_NAME service is not active according to systemctl."
                # Check if something is listening on port 6379 as fallback
                if ss -ltn | grep -q ':6379' 2>/dev/null; then
                    echo "However, something is listening on port 6379. Assuming a compatible Valkey/Redis server is running."
                else
                    echo "ERROR: The $SERVICE_NAME service is not active and nothing is listening on port 6379."
                    echo "Please start the service, e.g., 'sudo systemctl start $SERVICE_NAME' and ensure it's enabled to start on boot, e.g., 'sudo systemctl enable $SERVICE_NAME'."
                    exit 1
                fi
            else
                echo "The $SERVICE_NAME service is active."
            fi
        else
            # systemctl not available or not working, fall back to simple port check
            echo "WARNING: 'systemctl' not available/running. Checking for service via port."
            if ss -ltn | grep -q ':6379' 2>/dev/null; then
                echo "A service is listening on port 6379. Assuming a compatible Valkey/Redis server is running."
            else
                echo "ERROR: No running Valkey/Redis service detected on port 6379, and could not start one automatically as it was expected to be running."
                echo "Please start valkey-server or redis-server manually in another terminal."
                exit 1
            fi
        fi
    else
        echo "Valkey server was started by this script (PID $VALKEY_PID), skipping system service check."
    fi
    # --- End Valkey/Redis Check (Step 2) ---
fi

# --- 1.6. VLLM Setup and Server ---
echo "----------------------------------------"
echo "3 / 9. Starting VLLM server..."
echo "Output will be logged to: $LLAMA_SERVER_LOG_FILE"

# Determine which model to use based on argument
if [ "$MODEL_TYPE" = "normal" ]; then
    MODEL_NAME="Heralax/Augmentoolkit-DataSpecialist-v0.1"
    echo "Using normal model: $MODEL_NAME"
elif [ "$MODEL_TYPE" = "small" ]; then
    MODEL_NAME="Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit"
    echo "Using small (quantized) model: $MODEL_NAME"
    echo "=============================================="
    echo "!!!        IMPORTANT NOTICE        !!!"
    echo "=============================================="
    echo "You are using the quantized model."
    echo "When doing local dataset generation, you MUST use:"
    echo "  'Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit'"
    echo "in your config files instead of the default model."
    echo ""
    echo "Update your config files to use the quantized model:"
    echo "  - pdf_cleaning_small_model: Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit"
    echo "  - pdf_cleaning_large_model: Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit"
    echo "  - representation_variation_small_model: Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit"
    echo "  - representation_variation_large_model: Heralax/Augmentoolkit-DataSpecialist-gptqmodel-4bit"
    echo "  - ...etc... for any model. Find replace all cases of 'Heralax/Augmentoolkit-DataSpecialist-v0.1' with the gptq-4bit one."
    echo "=============================================="
    echo ""
else
    MODEL_NAME="$MODEL_TYPE"
    echo "Using custom model: $MODEL_NAME"
fi

# Start vllm server with output redirection
vllm serve "$MODEL_NAME" --port 8082 --max-model-len 20000 --tensor-parallel-size "$TENSOR_PARALLELISM" > "$LLAMA_SERVER_LOG_FILE" 2>&1 &
LLAMA_SERVER_PID=$!
sleep 5 # Give it a moment to start/fail

echo "VLLM server started with PID $LLAMA_SERVER_PID."
echo "Access VLLM server API at http://localhost:8082"
# --- End VLLM Setup (Step 3) ---


# --- 2. Huey Worker ---
echo "----------------------------------------"
echo "4 / 9. Starting Huey worker in the background (using venv)..."
echo "Output will be logged to: $HUEY_LOG_FILE"
# Call the huey_consumer script directly (should be in PATH from venv)
# Redirect both stdout and stderr to the log file
huey_consumer tasks.huey > "$HUEY_LOG_FILE" 2>&1 &
HUEY_PID=$!
# Give it a moment to start/fail
sleep 2
if ! kill -0 $HUEY_PID &> /dev/null; then
    echo "ERROR: Huey worker (PID $HUEY_PID) failed to start or crashed immediately."
    echo "Check the log file for details: $HUEY_LOG_FILE"
    exit 1 # Cleanup trap will handle exit
fi
echo "Huey worker started successfully with PID $HUEY_PID."


# --- 3. API Server ---
echo "----------------------------------------"
echo "5 / 9. Starting FastAPI server (Uvicorn) in the background (using venv)..."
echo "Output will be logged to: $UVICORN_LOG_FILE"
# Host 0.0.0.0 makes it accessible on the network
# Use python -m uvicorn to ensure using the venv uvicorn
# Redirect both stdout and stderr to the log file
python -m uvicorn api:app --host 0.0.0.0 --port 8000 > "$UVICORN_LOG_FILE" 2>&1 &
UVICORN_PID=$!
# Give it a moment to start/fail
sleep 2
if ! kill -0 $UVICORN_PID &> /dev/null; then
    echo "ERROR: Uvicorn server (PID $UVICORN_PID) failed to start or crashed immediately."
    echo "Check the log file for details: $UVICORN_LOG_FILE"
    # Cleanup trap will handle stopping Huey
    exit 1 # Cleanup trap will handle exit
fi
echo "Uvicorn server started successfully with PID $UVICORN_PID. Access API at http://localhost:8000"


# --- 4. NPM Check ---
echo "----------------------------------------"
echo "6 / 9. Checking for npm..."
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm command not found."
    echo "Please install Node.js and npm (e.g., via your system package manager like 'sudo apt install nodejs npm') and run this script again."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    exit 1
fi
echo "npm found."

# Check Node.js version
NODE_VERSION=$(node --version 2>/dev/null | sed 's/v//')
if [ -z "$NODE_VERSION" ]; then
    echo "ERROR: Could not determine Node.js version."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    exit 1
fi

# Extract major version number
NODE_MAJOR_VERSION=$(echo "$NODE_VERSION" | cut -d. -f1)

echo "Node.js version: v$NODE_VERSION"

# Check if Node.js version is at least 18
if [ "$NODE_MAJOR_VERSION" -lt 18 ]; then
    echo "ERROR: Node.js version $NODE_VERSION is too old."
    echo "This application requires Node.js 18.0.0 or higher (20+ recommended)."
    echo ""
    echo "To upgrade Node.js, run the following commands:"
    echo ""
    echo "1. Add NodeSource repository:"
    echo "   curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
    echo ""
    echo "2. Remove conflicting packages (if any):"
    echo "   apt-get remove -y libnode-dev libnode72"
    echo ""
    echo "3. Install Node.js 20.x:"
    echo "   apt-get install -y nodejs"
    echo ""
    echo "   OR run all steps at once:"
    echo "   curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get remove -y libnode-dev libnode72 2>/dev/null || true && apt-get install -y nodejs"
    echo ""
    echo "Alternative method using Node Version Manager (nvm):"
    echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "   source ~/.bashrc"
    echo "   nvm install 20"
    echo "   nvm use 20"
    echo ""
    echo "After upgrading, please run this script again."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    exit 1
fi


# --- 5. Frontend Dependencies ---
echo "----------------------------------------"
echo "7 / 9. Installing frontend dependencies in atk-interface..."
if [ ! -d "atk-interface" ]; then
    echo "ERROR: Directory 'atk-interface' not found in $(pwd)."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    exit 1
fi
cd atk-interface
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install npm dependencies in atk-interface. Check package.json and npm logs."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    cd "$SCRIPT_DIR" # Go back to root
    exit 1
fi
echo "Frontend dependencies installed."


# --- 6. Frontend Build ---
echo "----------------------------------------"
echo "8 / 9. Building frontend application in atk-interface..."
npm run build
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build frontend application. Check build script in package.json and logs."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    cd "$SCRIPT_DIR" # Go back to root
    exit 1
fi
echo "Frontend built successfully."


# --- 7. Frontend Serve ---
echo "----------------------------------------"
echo "9 / 9. Serving frontend application from atk-interface/dist/ in the FOREGROUND..."
echo "Output will be logged to: $FRONTEND_LOG_FILE"
# Ensure dist directory exists after build
if [ ! -d "dist" ]; then
    echo "ERROR: 'dist' directory not found in atk-interface after build."
    # Cleanup trap will handle stopping background services
    exit 1
fi


# Show summary of all log files
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "All helper processes are now running with output redirected to log files:"
if [ -n "$VALKEY_PID" ]; then
echo "  • Valkey server logs:   $VALKEY_LOG_FILE"
fi
echo "  • VLLM server logs:    $LLAMA_SERVER_LOG_FILE"
echo "  • Huey worker logs:     $HUEY_LOG_FILE"
echo "  • Uvicorn API logs:     $UVICORN_LOG_FILE"
echo "  • Frontend serve logs:  $FRONTEND_LOG_FILE"
echo ""
echo "You can monitor these logs in real-time with:"
if [ -n "$VALKEY_PID" ]; then
echo "  tail -f $VALKEY_LOG_FILE"
fi
echo "  tail -f $LLAMA_SERVER_LOG_FILE"
echo "  tail -f $HUEY_LOG_FILE"
echo "  tail -f $UVICORN_LOG_FILE"
echo "  tail -f $FRONTEND_LOG_FILE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Serve on port 5173 (matches CORS config)
# Run npx serve in the background first, then open browser once it's ready
echo "Starting serve process (output redirected to log file)..."
# Redirect both stdout and stderr to the log file - use ../ to go back to script directory
npx serve -s dist --listen 5173 > "../$FRONTEND_LOG_FILE" 2>&1 &
SERVE_PID=$!

# Wait for the serve process to be ready
echo "Waiting for frontend server to start..."
sleep 3

# Check if serve process is still running
if ! kill -0 $SERVE_PID &> /dev/null; then
    echo "ERROR: Frontend serve process failed to start or crashed immediately."
    echo "Check the log file for details: $FRONTEND_LOG_FILE"
    cleanup
    exit 1
fi

# Wait for the server to actually be listening on port 5173
echo "Checking if server is responding..."
for i in {1..10}; do
    if command -v curl &> /dev/null; then
        if curl -s http://localhost:5173 > /dev/null 2>&1; then
            break
        fi
    elif command -v wget &> /dev/null; then
        if wget -q --spider http://localhost:5173 > /dev/null 2>&1; then
            break
        fi
    else
        # Fallback: just check if something is listening on port 5173
        if command -v ss &> /dev/null; then
            if ss -ltn | grep -q ':5173'; then
                break
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -ltn | grep -q ':5173'; then
                break
            fi
        else
            # No way to check, just wait a bit more
            sleep 1
            break
        fi
    fi
    sleep 1
done

if ! kill -0 $LLAMA_SERVER_PID &> /dev/null; then
    echo "ERROR: VLLM server (PID $LLAMA_SERVER_PID) failed to start or crashed."
    echo "Check the log file for details: $LLAMA_SERVER_LOG_FILE"
    echo "The model probably wants to take too much memory. You may need to:"
    echo "  - Use 'small' argument to run quantized model: ./local_linux.sh small"
    echo "  - Use a custom model path: ./local_linux.sh path/to/your/model"
    echo "  - Use an API instead of local inference"
    cleanup
    exit 1
fi

echo "Opening interface..."

# --- Check for xdg-open ---
echo "Checking for xdg-open..."
if ! command -v xdg-open &> /dev/null; then
    echo "WARNING: 'xdg-open' command not found. Cannot automatically open browser."
    echo "Please install xdg-utils (e.g., 'sudo apt install xdg-utils' or 'sudo dnf install xdg-utils') or open http://localhost:5173 manually."
    # Not exiting, just warning
else
    echo "xdg-open found. Opening http://localhost:5173 in default browser..."
    xdg-open http://localhost:5173
fi

echo "Frontend server is now running. Press Ctrl+C to stop all services."

# Wait for the serve process to finish (it's now running in the background)
wait $SERVE_PID
SERVE_EXIT_CODE=$?
SERVE_PID="" # Clear PID as it has exited

if [ $SERVE_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Frontend serve process exited with code $SERVE_EXIT_CODE."
    echo "Check the log file for details: $FRONTEND_LOG_FILE"
    # Consider if we should stop background services here if serve crashes
    # Calling cleanup manually if serve fails
    cleanup
    exit 1 # Exit with error code
fi


# --- 8. Done (Only reached if npx serve exits gracefully, e.g., if killed separately) ---
cd "$SCRIPT_DIR" # Go back to root directory
echo "----------------------------------------"
echo "Frontend serve process finished."
echo ""
echo "Helper process log files were saved to:"
if [ -n "$VALKEY_PID" ]; then
echo "  • Valkey server logs:   $VALKEY_LOG_FILE"
fi
echo "  • VLLM server logs:    $LLAMA_SERVER_LOG_FILE"
echo "  • Huey worker logs:     $HUEY_LOG_FILE"
echo "  • Uvicorn API logs:     $UVICORN_LOG_FILE"
echo "  • Frontend serve logs:  $FRONTEND_LOG_FILE"
# Call cleanup to stop background services if serve finishes normally
cleanup

exit 0
