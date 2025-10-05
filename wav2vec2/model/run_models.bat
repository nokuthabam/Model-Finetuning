@echo off
setlocal enabledelayedexpansion

REM Explicitly define valid multilingual combinations in quotes
set combos= "xh nbl" "zu nbl ssw" "xh nbl ssw" "zu xh nbl ssw"

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
