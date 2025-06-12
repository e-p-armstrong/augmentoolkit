#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# set -e # Disabled for manual error checking below

echo "Starting Augmentoolkit services..."

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
    
    exit 0 # Exit script after cleanup
}

# Trap signals (INT: Ctrl+C, TERM: standard kill) and run cleanup function
trap cleanup SIGINT SIGTERM

# --- 1. Python Virtual Environment & Dependencies ---
echo "----------------------------------------"
echo "1. Setting up Python virtual environment and installing dependencies..."
VENV_DIR=".venv"

# Check if python3 command exists
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 command not found. Please install Python 3."
    exit 1
fi

# --- Check/Install uv ---
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "'uv' command not found. Attempting to install uv using pip..."
    # Ensure pip is available (it should be with python3)
    if ! python3 -m pip --version &> /dev/null; then
        echo "ERROR: python3 -m pip command failed. Cannot install uv."
        exit 1
    fi
    python3 -m pip install uv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install uv using pip. Please install uv manually (e.g., 'pip install uv' or 'brew install uv') and rerun the script."
        exit 1
    fi
    echo "uv installed successfully."
else
    echo "uv found."
fi
# --- End Check/Install uv ---


# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create Python virtual environment."
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

# --- Attempt to install SSL certificates (macOS specific fix) ---
echo "Attempting to install SSL certificates for the active Python environment..."
# Get the full path to the python executable inside the venv
VENV_PYTHON_EXE=$(which python)
# Try to infer the Applications directory path (this is heuristic)
# Assumes Python was installed via the standard macOS installer
PYTHON_BASE_DIR=$(dirname "$(dirname "$VENV_PYTHON_EXE")") # Go up two levels from venv/bin/python
CERT_SCRIPT_PATH=""
# Check common installation patterns
if [ -d "$PYTHON_BASE_DIR/Install Certificates.command" ]; then
    # Might be like /Library/Frameworks/Python.framework/Versions/3.11/Install Certificates.command
    CERT_SCRIPT_PATH="$PYTHON_BASE_DIR/Install Certificates.command"
elif [ -d "/Applications/Python $(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/Install Certificates.command" ]; then
    # Check standard /Applications path using the version
    PY_VERSION_SHORT=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    CERT_SCRIPT_PATH="/Applications/Python ${PY_VERSION_SHORT}/Install Certificates.command"
fi

if [ -n "$CERT_SCRIPT_PATH" ] && [ -f "$CERT_SCRIPT_PATH" ]; then
    echo "Found certificate installation script: $CERT_SCRIPT_PATH"
    echo "Running certificate installation script..."
    # Execute the script using the venv's python
    "$CERT_SCRIPT_PATH"
    if [ $? -ne 0 ]; then
        echo "WARNING: Certificate installation script finished with a non-zero status. SSL issues might persist."
        # Continue anyway, maybe it partially worked or wasn't strictly needed
    else
        echo "Certificate installation script completed successfully."
    fi
else
    echo "WARNING: Could not find the 'Install Certificates.command' script automatically."
    echo "If you encounter SSL errors (e.g., downloading NLTK data, used in chunking), you may need to run it manually."
    echo "Look for it in your Python application folder (e.g., /Applications/Python 3.11/)."
fi
# --- End SSL Certificate Install ---


# Install dependencies using uv
echo "Installing Python dependencies from requirements.txt using uv..."
# Assuming uv is in PATH after the check above and venv is activated
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies using uv. Please check requirements.txt and uv logs."
    exit 1 # Cleanup trap will handle exit
fi
echo "Python dependencies installed successfully."

# --- Configure SSL Certificate Path ---
echo "----------------------------------------"
echo "Configuring SSL certificate path..."

# Ensure certifi is installed (uv pip install should handle this if in requirements.txt,
# but explicit check adds robustness - though uv handles it well)
# It's likely already installed as a dependency of requests/huggingface_hub etc.
# If dependency issues arise, add 'certifi' explicitly to requirements.txt

# Get the certificate path from certifi
CERT_PATH=$(python -c "import certifi; print(certifi.where())" 2>/dev/null) # Suppress stderr in case certifi isn't installed yet

if [ -n "$CERT_PATH" ] && [ -f "$CERT_PATH" ]; then
    echo "Setting SSL_CERT_FILE environment variable to: $CERT_PATH"
    export SSL_CERT_FILE="$CERT_PATH"
    echo "SSL_CERT_FILE set for this script session and its children."
    # Optional: Add a connection test here if desired
    # python -c "import requests; r = requests.get('https://huggingface.co', timeout=10); print(f'Connection test status: {r.status_code}')"
    # if [ $? -ne 0 ]; then echo "WARNING: Connection test failed after setting SSL_CERT_FILE."; fi
else
    echo "WARNING: Could not determine certificate path using certifi. Certifi might not be installed yet or there's an issue."
    echo "Attempting to proceed, but SSL errors (like CERTIFICATE_VERIFY_FAILED) may occur."
    echo "Ensure 'certifi' is listed in requirements.txt if SSL errors persist."
fi
# --- End SSL Configuration ---


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
HUEY_LOG_FILE="$HELPER_LOGS_DIR/huey_worker_${TIMESTAMP}.log"
UVICORN_LOG_FILE="$HELPER_LOGS_DIR/uvicorn_api_${TIMESTAMP}.log"
FRONTEND_LOG_FILE="$HELPER_LOGS_DIR/frontend_serve_${TIMESTAMP}.log"

echo "Helper process logs will be saved to:"
echo "  Huey worker: $HUEY_LOG_FILE"
echo "  Uvicorn API: $UVICORN_LOG_FILE"
echo "  Frontend serve: $FRONTEND_LOG_FILE"
echo "Helper process logging setup complete."


# --- 1.5 Valkey (Redis Fork) Check & Installation ---
echo "----------------------------------------"
echo "1.5. Checking for Valkey (Redis fork)..."

# First, check if something is already listening on port 6379
if command -v lsof &> /dev/null; then
    PORT_CHECK_CMD="lsof -i :6379"
elif command -v netstat &> /dev/null; then
    PORT_CHECK_CMD="netstat -an"
else
    PORT_CHECK_CMD=""
fi

if [ -n "$PORT_CHECK_CMD" ]; then
    if [ "$PORT_CHECK_CMD" = "lsof -i :6379" ]; then
        if $PORT_CHECK_CMD &> /dev/null; then
            echo "Something is already listening on port 6379. Assuming a compatible Valkey/Redis server is running."
            echo "Skipping Valkey installation and setup."
        else
            NEED_VALKEY_SETUP=true
        fi
    else
        # netstat case
        if $PORT_CHECK_CMD | grep -q ':6379'; then
            echo "Something is already listening on port 6379. Assuming a compatible Valkey/Redis server is running."
            echo "Skipping Valkey installation and setup."
        else
            NEED_VALKEY_SETUP=true
        fi
    fi
else
    echo "WARNING: Cannot check port 6379 (lsof and netstat not available). Proceeding with Valkey setup."
    NEED_VALKEY_SETUP=true
fi

if [ "$NEED_VALKEY_SETUP" = true ]; then
    # Check if brew command exists

# Check if brew command exists
if ! command -v brew &> /dev/null; then
    echo "Homebrew (brew) command not found."
    read -p "Homebrew is required to install Valkey. Install Homebrew now? (y/N) " -n 1 -r
    echo # Move to a new line after input
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Attempting to install Homebrew. This may take a few minutes and will likely ask for your administrator password..."
        # Execute the official Homebrew installation script
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Check if installation succeeded by seeing if 'brew' command is now available
        # Might require sourcing profile or opening new shell, but let's check command first
        if ! command -v brew &> /dev/null; then
            echo "ERROR: Homebrew installation appears to have failed or is not available in this shell session."
            echo "Please try installing Homebrew manually from https://brew.sh/ and run this script again, possibly in a new terminal window."
            exit 1
        else
             # Add brew to PATH for the current script execution if possible
             # This is complex as the path depends on architecture (Intel vs ARM)
             # For simplicity, we'll rely on the installer potentially modifying the profile
             # or the user running in a new shell later if 'brew' isn't found immediately after install.
             # A more robust solution might involve detecting arch and adding the known path.
            echo "Homebrew installed successfully. Proceeding..."
            # Re-check explicitly here just in case the path isn't immediately active
             if ! command -v brew &> /dev/null; then
                  echo "ERROR: Homebrew installed but 'brew' command still not found in PATH for this session."
                  echo "Please open a new terminal window and run this script again."
                  exit 1
             fi
        fi
    else
        echo "Homebrew installation declined. Valkey cannot be installed automatically."
        echo "Please install Homebrew (https://brew.sh/) and Valkey ('brew install valkey') manually and rerun the script."
        exit 1
    fi
fi
echo "Homebrew found."

# Check if valkey is installed
if ! brew list valkey &>/dev/null; then
    echo "Valkey not found via Homebrew. Attempting to install..."
    # Update brew minimally and install valkey
    # Using 'brew install' directly avoids running 'brew update' if possible,
    # but Homebrew might still update itself or core formulae.
    # We avoid 'brew upgrade' to prevent upgrading all packages.
    brew install valkey
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install Valkey using Homebrew. Please install it manually ('brew install valkey') and rerun the script."
        exit 1
    fi
    echo "Valkey installed successfully."
else
    echo "Valkey found via Homebrew."
fi

# Check if Valkey service is running, start if not
if ! brew services list | grep -q "valkey.*started"; then
    echo "Valkey service is not running. Starting it now..."
    brew services start valkey
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start Valkey service using Homebrew. Please check Homebrew services ('brew services list') and start Valkey manually if needed."
        exit 1
    fi
    # Give it a moment to start
    sleep 2
    if ! brew services list | grep -q "valkey.*started"; then
         echo "ERROR: Valkey service failed to stay running after starting. Please investigate ('brew services list', check logs)."
         exit 1
    fi
    echo "Valkey service started successfully."
else
    echo "Valkey service is already running."
fi
fi # End NEED_VALKEY_SETUP conditional
# --- End Valkey Check ---


# --- 2. Huey Worker ---
echo "----------------------------------------"
echo "2. Starting Huey worker in the background (using venv)..."
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
echo "3. Starting FastAPI server (Uvicorn) in the background (using venv)..."
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
echo "4. Checking for npm..."
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm command not found."
    echo "Please install Node.js and npm (e.g., from https://nodejs.org/) and run this script again."
    echo "Stopping background services (PIDs $HUEY_PID, $UVICORN_PID)..."
    kill $HUEY_PID $UVICORN_PID
    exit 1
fi
echo "npm found."


# --- 5. Frontend Dependencies ---
echo "----------------------------------------"
echo "5. Installing frontend dependencies in atk-interface..."
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
echo "6. Building frontend application in atk-interface..."
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
echo "7. Serving frontend application from atk-interface/dist/ in the FOREGROUND..."
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
echo "  • Huey worker logs:     $HUEY_LOG_FILE"
echo "  • Uvicorn API logs:     $UVICORN_LOG_FILE"
echo "  • Frontend serve logs:  $FRONTEND_LOG_FILE"
echo ""
echo "You can monitor these logs in real-time with:"
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
    elif command -v nc &> /dev/null; then
        if nc -z localhost 5173 > /dev/null 2>&1; then
            break
        fi
    else
        # Fallback: just check if something is listening on port 5173
        if command -v lsof &> /dev/null; then
            if lsof -i :5173 > /dev/null 2>&1; then
                break
            fi
        elif command -v netstat &> /dev/null; then
            if netstat -an | grep -q ':5173'; then
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

echo "Opening interface..."

# Open the URL in the default browser (macOS)
echo "Opening http://localhost:5173 in default browser..."
open http://localhost:5173

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
echo "  • Huey worker logs:     $HUEY_LOG_FILE"
echo "  • Uvicorn API logs:     $UVICORN_LOG_FILE"
echo "  • Frontend serve logs:  $FRONTEND_LOG_FILE"
# Call cleanup to stop background services if serve finishes normally
cleanup

exit 0