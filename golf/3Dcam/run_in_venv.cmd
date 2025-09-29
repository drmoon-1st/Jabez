@echo off
rem Simple CMD launcher to create venv, install requirements, and run the client without PowerShell
setlocal enabledelayedexpansion

rem Resolve script folder
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set VENV_DIR=%SCRIPT_DIR%\.venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set PIP_EXE=%VENV_DIR%\Scripts\pip.exe

echo Running from %SCRIPT_DIR%

if not exist "%PYTHON_EXE%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Ensure Python is on PATH.
        pause
        exit /b 1
    )
)

echo Upgrading pip and installing requirements...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PIP_EXE%" install -r requirements.txt
"%PIP_EXE%" install requests

echo Starting client...
"%PYTHON_EXE%" "%SCRIPT_DIR%realsense_pack_and_upload.py"

endlocal
pause