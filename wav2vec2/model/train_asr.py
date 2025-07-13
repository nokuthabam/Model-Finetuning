import json
import torch
import torchaudio
import re
from pathlib import Path
from datasets import DataseDict, load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    DataCollatorCTCWithPadding
)
import evaluate

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

processor = None 


def load_jsonl_dataset(train_file, test_file):
    """
    Load JSONL dataset for training and testing.
    """
    train_dataset = load_dataset("json", data_files=train_file)["train"]
    test_dataset = load_dataset("json", data_files=test_file)["train"]
    return DataseDict({"train": train_dataset, "test": test_dataset})


def load_and_merge(language_code):

    # Lwazi
    lwazi_train = DATA_DIR / f"{language_code}_train.json"
    lwazi_test = DATA_DIR / f"{language_code}_test.json"

    # Common Voice
    cv_train = DATA_DIR / f"commonvoice_{language_code}_train.json"