@echo off
setlocal enabledelayedexpansion

echo Starting Augmentoolkit services...

REM --- Get script directory and change to it ---
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to change directory to script location '%SCRIPT_DIR%'.
    exit /b 1
)
echo Running in directory: %CD%

REM --- Define Virtual Environment Directory ---
set "VENV_DIR=.venv"

REM --- Process Management Variables ---
set "HUEY_PID="
set "UVICORN_PID="
set "SERVE_PID="

REM --- Skip cleanup function and go to main script ---
goto :main_script

REM --- Cleanup Function ---
REM Note: Batch file cleanup on Ctrl+C is limited. Background tasks started with 'start /b'
REM might persist if the script is interrupted before the foreground task (serve) ends.
REM The cleanup function below provides better process management.

:cleanup_services
echo.
echo ----------------------------------------
echo Stopping background services...

REM Function to kill processes by command line pattern
REM Stop Uvicorn (python processes running uvicorn)
echo Stopping Uvicorn API server...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul ^| find "python.exe" 2^>nul') do (
    set "PID_TO_CHECK=%%~i"
    REM Remove quotes from PID
    set "PID_TO_CHECK=!PID_TO_CHECK:"=!"
    REM Check if this python process is running uvicorn
    for /f %%j in ('wmic process where "ProcessId=!PID_TO_CHECK!" get CommandLine /format:csv 2^>nul ^| find "uvicorn" 2^>nul') do (
        echo Terminating Uvicorn process !PID_TO_CHECK!...
        taskkill /PID !PID_TO_CHECK! /T /F >nul 2>&1
    )
)

REM Stop Huey Consumer  
echo Stopping Huey worker...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul ^| find "python.exe" 2^>nul') do (
    set "PID_TO_CHECK=%%~i"
    REM Remove quotes from PID
    set "PID_TO_CHECK=!PID_TO_CHECK:"=!"
    REM Check if this python process is running huey
    for /f %%j in ('wmic process where "ProcessId=!PID_TO_CHECK!" get CommandLine /format:csv 2^>nul ^| find "huey" 2^>nul') do (
        echo Terminating Huey process !PID_TO_CHECK!...
        taskkill /PID !PID_TO_CHECK! /T /F >nul 2>&1
    )
)

REM Stop any Node.js serve processes
echo Stopping frontend serve processes...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq node.exe" /FO CSV /NH 2^>nul ^| find "node.exe" 2^>nul') do (
    set "PID_TO_CHECK=%%~i"
    REM Remove quotes from PID
    set "PID_TO_CHECK=!PID_TO_CHECK:"=!"
    REM Check if this node process is running serve
    for /f %%j in ('wmic process where "ProcessId=!PID_TO_CHECK!" get CommandLine /format:csv 2^>nul ^| find "serve" 2^>nul') do (
        echo Terminating serve process !PID_TO_CHECK!...
        taskkill /PID !PID_TO_CHECK! /T /F >nul 2>&1
    )
)

REM Give processes time to terminate
timeout /t 2 /nobreak >nul

echo Cleanup complete.
echo Services stopped. Terminal ready for new commands.
echo ----------------------------------------
goto :eof

:main_script
REM --- 1. Python Virtual Environment & Dependencies ---
echo ----------------------------------------
echo 1. Setting up Python virtual environment and installing dependencies...

REM Check if python command exists
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python command not found. Please install Python 3 and ensure it's in your PATH.
    exit /b 1
)
echo Python found.

REM --- Check/Install uv ---
echo Checking for uv...
where uv >nul 2>nul
if errorlevel 1 (
    echo uv command not found. Attempting to install uv using pip...
    where python >nul 2>nul
    if errorlevel 1 (
         echo ERROR: python command not found. Cannot install uv.
         exit /b 1
    )
    python -m pip install uv
    if errorlevel 1 (
        echo ERROR: Failed to install uv using pip. Please install uv manually (e.g., "pip install uv") and rerun the script.
        exit /b 1
    )
    echo uv installed successfully.
) else (
    echo uv found.
)
REM --- End Check/Install uv ---

REM Create virtual environment if it doesn't exist (check for activate script)
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in '%VENV_DIR%'...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create Python virtual environment.
        exit /b 1
    )
) else (
    echo Virtual environment '%VENV_DIR%' already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate Python virtual environment.
    exit /b 1
)
echo Virtual environment activated.

REM --- Configure SSL Certificate Path ---
echo ----------------------------------------
echo Configuring SSL certificate path...

REM Get the certificate path from certifi
for /f "delims=" %%i in ('python -c "import certifi; print(certifi.where())" 2^>nul') do set "CERT_PATH=%%i"

if defined CERT_PATH (
    if exist "%CERT_PATH%" (
        echo Setting SSL_CERT_FILE environment variable to: %CERT_PATH%
        set "SSL_CERT_FILE=%CERT_PATH%"
        echo SSL_CERT_FILE set for this script session and its children.
    ) else (
        echo WARNING: Certificate path from certifi does not exist: %CERT_PATH%
    )
) else (
    echo WARNING: Could not determine certificate path using certifi. Certifi might not be installed yet.
    echo Attempting to proceed, but SSL errors may occur.
)

REM Install dependencies using uv
echo Installing Python dependencies from requirements.txt using uv...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies using uv. Please check requirements.txt and uv logs.
    exit /b 1
)
echo Python dependencies installed successfully.

REM --- Helper Process Logging Setup ---
echo ----------------------------------------
echo Setting up helper process logging...

REM Create helper_process_logs directory if it doesn't exist
set "HELPER_LOGS_DIR=helper_process_logs"
if not exist "%HELPER_LOGS_DIR%" (
    echo Creating helper process logs directory: %HELPER_LOGS_DIR%
    mkdir "%HELPER_LOGS_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create helper process logs directory.
        exit /b 1
    )
) else (
    echo Helper process logs directory already exists: %HELPER_LOGS_DIR%
)

REM Generate timestamp for log files
for /f "delims=" %%i in ('powershell -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TIMESTAMP=%%i"
set "HUEY_LOG_FILE=%HELPER_LOGS_DIR%\huey_worker_%TIMESTAMP%.log"
set "UVICORN_LOG_FILE=%HELPER_LOGS_DIR%\uvicorn_api_%TIMESTAMP%.log"
set "FRONTEND_LOG_FILE=%HELPER_LOGS_DIR%\frontend_serve_%TIMESTAMP%.log"

echo Helper process logs will be saved to:
echo   Huey worker: %HUEY_LOG_FILE%
echo   Uvicorn API: %UVICORN_LOG_FILE%
echo   Frontend serve: %FRONTEND_LOG_FILE%
echo Helper process logging setup complete.

REM --- 1.5 Valkey/Redis Check ---
echo ----------------------------------------
echo 1.5. Checking for running Valkey/Redis service...

REM Check if something is listening on port 6379
netstat -an | find ":6379" >nul 2>&1
if errorlevel 1 (
    REM Nothing listening on port 6379, need to check for and potentially install Valkey/Redis
    echo No service found listening on port 6379.
    
    REM Check if valkey-server or redis-server process is running
    tasklist /FI "IMAGENAME eq valkey-server.exe" /NH 2>nul | find /I "valkey-server.exe" > nul
    if errorlevel 1 (
        tasklist /FI "IMAGENAME eq redis-server.exe" /NH 2>nul | find /I "redis-server.exe" > nul
        if errorlevel 1 (
            echo WARNING: Valkey/Redis process not found running and port 6379 is not in use.

            REM Check for package managers
            where choco >nul 2>nul
            if not errorlevel 1 (
                echo Found Package Manager: Chocolatey (choco)
                echo Please install Valkey using Chocolatey in an ADMINISTRATOR terminal:
                echo   choco install valkey
                echo Then start the Valkey service (e.g., via services.msc or 'valkey-server --service-start')
                echo Then re-run this script.
                goto :exit_script
            )

            where scoop >nul 2>nul
            if not errorlevel 1 (
                echo Found Package Manager: Scoop
                echo Please install Valkey using Scoop:
                echo   scoop install valkey
                echo Then start the Valkey service (e.g., run 'valkey-server' in a separate terminal)
                echo Then re-run this script.
                goto :exit_script
            )

            where winget >nul 2>nul
            if not errorlevel 1 (
                echo Found Package Manager: Winget
                echo Please install Valkey/Redis using Winget:
                echo   winget install Redis.Redis
                echo Or search for a Valkey package if available.
                echo Then start the Redis service (e.g., via services.msc)
                echo Then re-run this script.
                goto :exit_script
            )

            echo No common package manager (choco, scoop, winget) found.
            echo Please install Valkey or Redis manually.
            echo Option 1: Install Chocolatey (https://chocolatey.org/install) then run 'choco install valkey'
            echo Option 2: Download from Valkey/Redis websites and follow their setup instructions.
            echo Ensure the service is running before re-running this script.
            echo See docs/quickstart.md for more details.
            goto :exit_script

        ) else (
            echo Found running process: redis-server.exe
        )
    ) else (
        echo Found running process: valkey-server.exe
    )
) else (
    echo Something is already listening on port 6379. Assuming a compatible Valkey/Redis server is running.
)
echo Valkey/Redis service appears to be available.
echo ----------------------------------------

REM --- 2. Huey Worker ---
echo ----------------------------------------
echo 2. Starting Huey worker in the background (using venv)...
echo Output will be logged to: %HUEY_LOG_FILE%

REM Check if huey_consumer is available
where huey_consumer >nul 2>nul
if errorlevel 1 (
    echo ERROR: huey_consumer command not found after activation. Check venv installation.
    exit /b 1
)

REM Use start /b to run in background without a new window, redirect output to log file
start /b "" cmd /c "huey_consumer tasks.huey > %HUEY_LOG_FILE% 2>&1"

echo Huey worker started in background. Check log file for status: %HUEY_LOG_FILE%
timeout /t 3 /nobreak > nul

REM Check if process started successfully by looking for recent log file content
if exist "%HUEY_LOG_FILE%" (
    REM Check if log file has some content (basic validation)
    for %%F in ("%HUEY_LOG_FILE%") do if %%~zF gtr 0 (
        echo Huey worker appears to have started successfully.
    ) else (
        echo WARNING: Huey worker log file is empty. Process may have failed to start.
    )
) else (
    echo WARNING: Huey worker log file not created. Process may have failed to start.
)

REM --- 3. API Server ---
echo ----------------------------------------
echo 3. Starting FastAPI server (Uvicorn) in the background (using venv)...
echo Output will be logged to: %UVICORN_LOG_FILE%

REM Host 0.0.0.0 makes it accessible on the network
REM Use start /b to run in background, redirect output to log file
start /b "" cmd /c "python -m uvicorn api:app --host 0.0.0.0 --port 8000 > %UVICORN_LOG_FILE% 2>&1"

echo Uvicorn server started in background. Access API at http://localhost:8000
echo Check log file for status: %UVICORN_LOG_FILE%
timeout /t 3 /nobreak > nul

REM Check if process started successfully by looking for recent log file content
if exist "%UVICORN_LOG_FILE%" (
    REM Check if log file has some content (basic validation)
    for %%F in ("%UVICORN_LOG_FILE%") do if %%~zF gtr 0 (
        echo Uvicorn server appears to have started successfully.
    ) else (
        echo WARNING: Uvicorn server log file is empty. Process may have failed to start.
    )
) else (
    echo WARNING: Uvicorn server log file not created. Process may have failed to start.
)

REM --- 4. NPM Check ---
echo ----------------------------------------
echo 4. Checking for npm...
where npm >nul 2>nul
if errorlevel 1 (
    echo ERROR: npm command not found.
    echo Please install Node.js and npm (e.g., from https://nodejs.org/) and ensure they are in your PATH.
    call :cleanup_services
    exit /b 1
)
echo npm found.

REM --- 5. Frontend Dependencies ---
echo ----------------------------------------
echo 5. Installing frontend dependencies in atk-interface...
if not exist "atk-interface\" (
    echo ERROR: Directory 'atk-interface' not found in %CD%.
    call :cleanup_services
    exit /b 1
)
cd atk-interface
npm install
if errorlevel 1 (
    echo ERROR: Failed to install npm dependencies in atk-interface. Check package.json and npm logs.
    call :cleanup_services
    cd "%SCRIPT_DIR%"
    exit /b 1
)
echo Frontend dependencies installed.

REM --- 6. Frontend Build ---
echo ----------------------------------------
echo 6. Building frontend application in atk-interface...
npm run build
if errorlevel 1 (
    echo ERROR: Failed to build frontend application. Check build script in package.json and logs.
    call :cleanup_services
    cd "%SCRIPT_DIR%"
    exit /b 1
)
echo Frontend built successfully.

REM --- 7. Frontend Serve ---
echo ----------------------------------------
echo 7. Serving frontend application from atk-interface/dist/ in the FOREGROUND...
echo Output will be logged to: %FRONTEND_LOG_FILE%

REM Ensure dist directory exists after build
if not exist "dist\" (
    echo ERROR: 'dist' directory not found in atk-interface after build.
    call :cleanup_services
    cd "%SCRIPT_DIR%"
    exit /b 1
)

REM Show summary of all log files
echo.
echo ===============================================================================
echo All helper processes are now running with output redirected to log files:
echo   • Huey worker logs:     %HUEY_LOG_FILE%
echo   • Uvicorn API logs:     %UVICORN_LOG_FILE%
echo   • Frontend serve logs:  %FRONTEND_LOG_FILE%
echo.
echo You can monitor these logs by opening the files in a text editor or using:
echo   type "%HUEY_LOG_FILE%"
echo   type "%UVICORN_LOG_FILE%"
echo   type "%FRONTEND_LOG_FILE%"
echo ===============================================================================
echo.

REM Start serve process in background first, then check if it's ready
echo Starting serve process (output redirected to log file)...
start /b "" cmd /c "npx serve -s dist --listen 5173 > ..\%FRONTEND_LOG_FILE% 2>&1"

REM Wait for the serve process to be ready
echo Waiting for frontend server to start...
timeout /t 5 /nobreak > nul

REM Check if server is responding by attempting to connect to port 5173
echo Checking if server is responding...
set "SERVER_READY=false"
for /l %%i in (1,1,10) do (
    netstat -an | find ":5173" >nul 2>&1
    if not errorlevel 1 (
        set "SERVER_READY=true"
        goto :server_check_done
    )
    timeout /t 1 /nobreak > nul
)
:server_check_done

if "%SERVER_READY%"=="false" (
    echo WARNING: Frontend server may not have started properly. Check log file: %FRONTEND_LOG_FILE%
    echo Continuing anyway...
)

echo Opening interface...

REM Open the URL in the default browser (Windows)
echo Opening http://localhost:5173 in default browser...
start "" http://localhost:5173

echo Frontend server is now running. Press Ctrl+C to stop all services.
echo.
echo Helper process log files are being updated in real-time:
echo   • Huey worker logs:     %HUEY_LOG_FILE%
echo   • Uvicorn API logs:     %UVICORN_LOG_FILE%
echo   • Frontend serve logs:  %FRONTEND_LOG_FILE%

REM Wait for user input to keep the script running and allow manual cleanup
echo.
echo Press any key to stop all services and exit...
pause >nul

REM --- 8. Done ---
cd "%SCRIPT_DIR%"
echo ----------------------------------------
echo Stopping all services...

call :cleanup_services

echo.
echo Helper process log files were saved to:
echo   • Huey worker logs:     %HUEY_LOG_FILE%
echo   • Uvicorn API logs:     %UVICORN_LOG_FILE%
echo   • Frontend serve logs:  %FRONTEND_LOG_FILE%

echo Exiting script.

REM Deactivate virtual environment if desired (optional)
REM if defined VIRTUAL_ENV (
REM     echo Deactivating virtual environment...
REM     call deactivate
REM )

goto :end_script

REM Add a label for exiting cleanly due to missing dependencies
:exit_script
echo Exiting script due to missing dependencies.
endlocal
exit /b 1

:end_script
endlocal
exit /b 0
