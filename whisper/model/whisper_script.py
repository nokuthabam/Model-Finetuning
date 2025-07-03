import os
from whisper import load_model
import pandas as pd

# Configuration
WHISPER_MODEL_NAME = "turbo"

def load_whisper_model():
    """
    load the whisper model
    """
    model = load_model(WHISPER_MODEL_NAME)
    return model

def transcribe_audio(model, audio_path):
    """
    Transcribe audio using the whisper model.
    """
    result = model.transcribe(audio_path, language="en", verbose=True)
    return result

def fine_tune_model(model, audio_path, text)L
    