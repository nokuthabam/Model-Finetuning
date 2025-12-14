import argparse
from pathlib import Path
import json
from tqdm import tqdm
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import logging
from datetime import datetime
import torchaudio.functional as F
import torchaudio.transforms as T
import re

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
LANGUAGE_MODEL_MAP = {
    "zu": MODEL_DIR / "zu_whisper",
    "xh": MODEL_DIR / "xh_whisper",
    "ss": MODEL_DIR / "ss_whisper",
    "nr": MODEL_DIR / "nr_whisper"
}
CHARS_TO_REMOVE_REGEX = r'[\,\?\.\!\-\;\:\"\“\%\‘\”\'\…\•\°\(\)\=\*\/\`\ː\’]'


def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_file = LOG_DIR / f"{language_code}_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


def load_model(lang_code, logger):
    """
    Load the finetuned Whisper model and processor for the specified language code.
    """
    model_path = LANGUAGE_MODEL_MAP.get(lang_code)
    logger.info(f"Loading model from {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="swahili", task="transcribe")
    model.eval()
    logger.info(f"Model loaded successfully for {lang_code} from {model_path}")
    return model, processor


def trim_silence_from_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Normalize to avoid issues
    waveform = waveform / waveform.abs().max()

    # Apply VAD
    trimmed_waveform = F.vad(waveform, sample_rate=sample_rate)
    return trimmed_waveform


def clean_text(text: str) -> str:
    """
    Cleans and normalizes transcript text for ASR.
    - Lowercases text
    - Removes noise markers (e.g., [n], [s], [um], [uh], [sos], [silence])
    - Removes any remaining bracketed tokens like [laughs], [noise]
    - Removes unwanted punctuation and special symbols
    - Normalizes spaces and trims ends
    """

    if not isinstance(text, str):
        return ""

    # Lowercase and strip whitespace
    text = text.lower().strip()

    # Remove known noise/filler markers explicitly
    text = re.sub(
        r"\[(n|s|um|uh|sos|silence)\]", " ",
        text
    )

    # Remove any other bracketed annotations, e.g., [laughs], [noise], [inaudible]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove unwanted punctuation but keep accents, apostrophes, and hyphens
    text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transcribe_audio(model, processor, audio_path, device):
    """
    Transcribe audio file using the Whisper model.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Trim any silence from the beginning and end
    # waveform = trim_silence_from_waveform(waveform, sample_rate=16000)

    inputs = processor(waveform.squeeze(0),
                       return_tensors="pt",
                       sampling_rate=16000,
                       return_attention_mask=True
                       ).to(device)
    #forced_decoder_ids = processor.get_decoder_prompt_ids(language="sw", task="transcribe")
    generation_config = model.generation_config
    generation_config.language = "sw"
    generation_config.task = "transcribe"
    generation_config.num_beams = 5
    generation_config.do_sample = False
    # generation_config.temperature = 0.0
    generation_config.no_repeat_ngram_size = 3
    generation_config.length_penalty = 1.0
    # generation_config.top_p = 1.0
    # generation_config.top_k = 0
    generation_config.suppress_tokens = []  # prevent forced token suppression

    # Run generation with tuned decoding parameters
    predicted_ids = model.generate(
        inputs["input_features"],
        generation_config=generation_config
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.lower()


def run_inference(language_code, logger):
    """
    Run inference on all audio files in the specified language directory.
    """
    language = LANGUAGE_MAP.get(language_code, language_code)
    nchlt_unseen_data = DATA_DIR / f"nchlt_{language}_test.json"
    lwazi_unseen_data = DATA_DIR / f"{language}_unseen.json"
    output_path = OUTPUT_DIR / f"{language}_inference_results.json"

    model, processor = load_model(language_code, logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Print model summary
    with open(nchlt_unseen_data, 'r') as f:
        lines = f.readlines()
    
    with open(lwazi_unseen_data, 'r') as f:
        lines += f.readlines()

    # Jumble the lines to ensure a mix of datasets
    import random
    random.shuffle(lines)
    # Limit to first 500 lines for quicker inference during testing
    lines = lines[:1500]
    logger.info(f"Running inference for {language} on {len(lines)} audio files")
    results = []
    for line in tqdm(lines, desc=f"Processing {language} audio files"):
        data = json.loads(line)
        audio_path = data["audio"]
        reference = data["transcription"]
        reference = clean_text(reference)
        hypothesis = transcribe_audio(model, processor, audio_path, device)
        hypothesis = clean_text(hypothesis)
        results.append({
            "audio": audio_path,
            "reference": reference,
            "hypothesis": hypothesis
        })

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"Error analysis report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for error analysis on Whisper model.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., zu, xh, ss, nr)")
    args = parser.parse_args()

    language_code = args.language
    logger = setup_logging(language_code)
    
    try:
        run_inference(language_code, logger)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise