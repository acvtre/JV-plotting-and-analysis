@echo off
REM Batch file wrapper for plot_jv_curves.py to handle drag and drop

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Run the Python script with all dragged files as arguments
python "%SCRIPT_DIR%plot_jv_curves.py" %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
