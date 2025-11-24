import argparse
from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "error_analysis"
LOG_DIR = BASE_DIR / "logs/error_analysis"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_MAP = {
    "zu": "zulu",
    "xh": "xhosa",
    "ss": "siswati",
    "nr": "ndebele"
}

def setup_logging(lang_code):
    log_file = LOG_DIR / f"{lang_code}_whisper_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_matching_model_dirs(lang_code):
    """
    Return all Whisper model subdirectories that contain the given language code.
    """
    return sorted([
        model_dir for model_dir in MODEL_DIR.iterdir()
        if model_dir.is_dir() and lang_code in model_dir.name and "whisper" in model_dir.name
    ])


def load_model(model_path, logger):
    logger.info(f"Loading model from {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="swahili", task="transcribe")
    model.eval()
    logger.info(f"Model loaded successfully from {model_path.name}")
    return model, processor


def transcribe_audio(model, processor, audio_path, device):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    inputs = processor(waveform.squeeze(0), 
                       return_tensors="pt",
                       sampling_rate=16000,
                       return_attention_mask=True
                       ).to(device)
    generation_config = model.generation_config
    generation_config.language = "sw"
    generation_config.task = "transcribe"
    generation_config.num_beams = 5
    generation_config.do_sample = False
    generation_config.temperature = 0.0
    generation_config.no_repeat_ngram_size = 3
    generation_config.length_penalty = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 0
    generation_config.suppress_tokens = []  # prevent forced token suppression
    predicted_ids = model.generate(
        inputs["input_features"],
        generation_config=generation_config
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.lower()


def run_inference(language_code, logger):
    language = LANGUAGE_MAP.get(language_code, language_code)
    data_file = DATA_DIR / f"{language}_unseen.json"

    with open(data_file, 'r') as f:
        lines = f.readlines()

    lines = lines[:500]  # Limit to 500 for consistency
    logger.info(f"Loaded {len(lines)} samples for {language}")

    model_dirs = get_matching_model_dirs(language_code)
    logger.info(f"Found {len(model_dirs)} Whisper models for {language_code}: {[m.name for m in model_dirs]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_dir in model_dirs:
        logger.info(f"Processing with model: {model_dir.name}")
        model, processor = load_model(model_dir, logger)
        model.to(device)

        results = []
        for line in tqdm(lines, desc=f"Running inference with {model_dir.name}"):
            data = json.loads(line)
            audio_path = data["audio"]
            reference = data["transcription"]
            hypothesis = transcribe_audio(model, processor, audio_path, device)
            results.append({
                "audio": audio_path,
                "reference": reference,
                "hypothesis": hypothesis
            })

        output_file = OUTPUT_DIR / f"{language}_{model_dir.name}_results.json"
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        logger.info(f"Saved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for all Whisper models trained on a specific language")
    parser.add_argument("--language", type=str, required=True, help="Language code (zu, xh, ss, nr)")
    args = parser.parse_args()

    logger = setup_logging(args.language)
    try:
        run_inference(args.language, logger)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
