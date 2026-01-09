@echo off
setlocal enabledelayedexpansion

REM Define languages
set languages="zu" "xh" "nr" "ss"

REM Loop through each language
for %%C in (%languages%) do (
    echo ============================
    echo Starting Language: %%~C
    echo ============================

    python infer_error.py --language %%~C

    echo Finished Language: %%~C
    echo Sleeping for 60 seconds before next...
    timeout /t 60 >nul
)

echo All multilingual languages completed.
