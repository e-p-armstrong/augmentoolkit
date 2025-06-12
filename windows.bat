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

REM --- Cleanup Functionality Notes ---
REM Batch file cleanup on Ctrl+C is limited. Background tasks started with 'start /b'
REM might persist if the script is interrupted before the foreground task (serve) ends.
REM You may need to manually stop python.exe (Uvicorn) and huey_consumer processes
REM using Task Manager in such cases.

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
    echo 'uv' command not found. Attempting to install uv using pip...
    where python >nul 2>nul
    if errorlevel 1 (
         echo ERROR: python command not found. Cannot install uv.
         exit /b 1
    )
    python -m pip install uv
    if errorlevel 1 (
        echo ERROR: Failed to install uv using pip. Please install uv manually (e.g., 'pip install uv') and rerun the script.
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

REM Install dependencies using uv
echo Installing Python dependencies from requirements.txt using uv...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies using uv. Please check requirements.txt and uv logs.
    exit /b 1
)
echo "Python dependencies installed successfully."


REM --- 1.5 Valkey/Redis Check ---
echo ----------------------------------------
echo 1.5. Checking for running Valkey/Redis service...

REM Check if valkey-server or redis-server process is running
tasklist /FI "IMAGENAME eq valkey-server.exe" /NH | find /I "valkey-server.exe" > nul
if errorlevel 1 (
    tasklist /FI "IMAGENAME eq redis-server.exe" /NH | find /I "redis-server.exe" > nul
    if errorlevel 1 (
        echo WARNING: Valkey/Redis process (valkey-server.exe or redis-server.exe) not found running.

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
            echo Please install Valkey using Scoop in an ADMINISTRATOR terminal:
            echo   scoop install valkey
            echo Then start the Valkey service (e.g., run 'valkey-server' in a separate terminal)
            echo Then re-run this script.
            goto :exit_script
        )

        where winget >nul 2>nul
        if not errorlevel 1 (
            echo Found Package Manager: Winget
            echo Please install Valkey/Redis using Winget in an ADMINISTRATOR terminal:
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
echo Valkey/Redis service appears to be running.
echo ----------------------------------------


REM --- 2. Huey Worker ---
echo ----------------------------------------
echo 2. Starting Huey worker in the background (using venv)...
REM Use start /b to run in background without a new window
where huey_consumer >nul 2>nul
if errorlevel 1 (
    echo ERROR: huey_consumer command not found after activation. Check venv installation.
    exit /b 1
)
start /b "" huey_consumer tasks.huey
echo Huey worker started in background. Check console output/logs for status.
timeout /t 2 /nobreak > nul


REM --- 3. API Server ---
echo ----------------------------------------
echo 3. Starting FastAPI server (Uvicorn) in the background (using venv)...
REM Host 0.0.0.0 makes it accessible on the network
start /b "" python -m uvicorn api:app --host 0.0.0.0 --port 8000
echo Uvicorn server started in background. Access API at http://localhost:8000
timeout /t 2 /nobreak > nul


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
REM Ensure dist directory exists after build
if not exist "dist\" (
    echo ERROR: 'dist' directory not found in atk-interface after build.
    call :cleanup_services
    cd "%SCRIPT_DIR%"
    exit /b 1
)

REM Open the URL in the default browser (Windows)
echo Opening http://localhost:5173 in default browser...
start "" http://localhost:5173

REM Serve on port 5173 (matches CORS config)
REM Run npx serve in the foreground - Script will wait here
echo Starting serve process in the foreground...
npx serve -s dist --listen 5173
set SERVE_EXIT_CODE=%ERRORLEVEL%

if %SERVE_EXIT_CODE% neq 0 (
    echo WARNING: Frontend serve process exited with code %SERVE_EXIT_CODE%.
    call :cleanup_services
    cd "%SCRIPT_DIR%"
    exit /b 1
)


REM --- 8. Done ---
cd "%SCRIPT_DIR%"
echo ----------------------------------------
echo Frontend serve process finished.

call :cleanup_services

REM Note: Background Huey and Uvicorn tasks should now be stopped.
echo Exiting script.

REM Deactivate virtual environment if desired (optional)
REM if defined VIRTUAL_ENV (
REM     echo Deactivating virtual environment...
REM     call deactivate
REM )

REM Add a label for exiting cleanly
:exit_script
echo Exiting script due to Valkey/Redis requirement.
endlocal
exit /b 1

REM Cleanup subroutine for consistent process termination
:cleanup_services
echo ----------------------------------------
echo Stopping background services...

REM Stop Uvicorn (python processes running uvicorn)
echo Stopping Uvicorn API server...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| find "python.exe"') do (
    for /f "tokens=*" %%j in ('tasklist /FI "PID eq %%~i" /FO LIST ^| find /I "uvicorn"') do (
        echo Terminating Uvicorn process %%~i...
        taskkill /PID %%~i /T /F >nul 2>&1
    )
)

REM Stop Huey Consumer  
echo Stopping Huey worker...
for /f "tokens=2 delims=," %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| find "python.exe"') do (
    for /f "tokens=*" %%j in ('tasklist /FI "PID eq %%~i" /FO LIST ^| find /I "huey"') do (
        echo Terminating Huey process %%~i...
        taskkill /PID %%~i /T /F >nul 2>&1
    )
)

REM Fallback: Kill any remaining python processes that might be our services
echo Attempting cleanup of any remaining service processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *huey*" >nul 2>&1

REM Give processes time to terminate
timeout /t 2 /nobreak >nul

echo Cleanup complete.
echo Services stopped. Terminal ready for new commands.
echo ----------------------------------------
goto :eof

REM Ensure the original end of script is reachable if Valkey/Redis IS running
REM The main script logic should continue after section 1.5 if no exit occurred.
