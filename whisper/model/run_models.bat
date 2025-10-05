@echo off
setlocal enabledelayedexpansion

REM Explicitly define valid multilingual combinations in quotes
set combos="zu ss" "xh ss" "nr ss" "zu xh ss" "zu nr ss" "xh nr ss" "zu xh nr ss"

REM Loop through each combination
for %%C in (%combos%) do (
    echo ============================
    echo Starting Combination: %%~C
    echo ============================

    python multilingual.py --languages %%~C

    echo Finished Combination: %%~C
    echo Sleeping for 60 seconds before next...
    timeout /t 60 >nul
)

echo All multilingual combinations completed.
