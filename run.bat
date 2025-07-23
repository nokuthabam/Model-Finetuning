@echo off
setlocal enabledelayedexpansion

REM Set base path to your script
set SCRIPT_DIR=D:\Model-Finetuning\wav2vec2\model



REM List of language codes
set LANGUAGES=zu xh ssw nbl

for %%L in (%LANGUAGES%) do (
    echo ===========================
    echo Training wav2vec2 for language %%L
    echo ===========================

    %PYTHON_EXE% "%SCRIPT_DIR%\train_asr.py" --language %%L

    if errorlevel 1 (
        echo Failed training for language %%L
    )
)

echo All trainings finished.
pause
