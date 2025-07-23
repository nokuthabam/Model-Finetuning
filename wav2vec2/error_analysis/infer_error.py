import argparse
from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import logging
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "error_analysis"
LOG_DIR = BASE_DIR / "logs/error_analysis"
LANGUAGE_MAP = {
    "zu": "zulu",
    "xh": "xhosa",
    "ssw": "siswati",
    "nbl": "ndebele"
}
LANGUAGE_MODEL_MAP = {
    "zu": MODEL_DIR / "zu_wav2vec2",
    "xh": MODEL_DIR / "xh_wav2vec2",
    "ssw": MODEL_DIR / "ssw_wav2vec2",
    "nbl": MODEL_DIR / "nbl_wav2vec2"
}

def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_file = LOG_DIR / f"{language_code}_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger
    


def load_model(lang_code, logger):
    """
    Load the finetuned Wav2Vec2 model and processor for the specified language code.
    """
    model_path = LANGUAGE_MODEL_MAP.get(lang_code)
    logger.info(f"Loading model from {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    logger.info(f"Model loaded successfully for {lang_code}")
    return model, processor


def transcribe_audio(model, processor, audio_path, logger):
    """
    Transcribe audio using the Wav2Vec2 model.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription.lower()

def run_inference(language_code, logger):
    language = LANGUAGE_MAP.get(language_code)
    unseen_data_path = DATA_DIR / f"{language}_unseen.json"
    output_path = OUTPUT_DIR / f"{language}_inference_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(language_code, logger)
    logger.info(f"Running inference for {language}")

    devices = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(devices)

    with open(unseen_data_path, 'r') as f:
        lines = f.readlines()
    logger.info(f"Loaded {len(lines)} entries from {unseen_data_path}")
    # Limit the number of lines for quick testing
    lines = lines[:500]  
    results = []

    for line in tqdm(lines, desc=f"Running inference for {language}"):
        entry = json.loads(line)
        audio_path = entry["audio_path"]
        reference = entry["transcript"]
        hypothesis = transcribe_audio(model, processor, audio_path, logger)
        results.append({
            "audio_path": audio_path,
            "reference": reference,
            "hypothesis": hypothesis
        })
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Inference results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on unseen data for Wav2Vec2 model.")
    parser.add_argument("--language_code", type=str, choices=LANGUAGE_MAP.keys(), required=True,
                        help="Language code for the unseen dataset.")
    args = parser.parse_args()
    logger = setup_logging(args.language_code)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting inference for {LANGUAGE_MAP[args.language_code]} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_inference(args.language_code, logger)
    logger.info(f"Inference completed for {LANGUAGE_MAP[args.language_code]}. Results saved to {OUTPUT_DIR / f'{args.language_code}_inference_results.json'}.")