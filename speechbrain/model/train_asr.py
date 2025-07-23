import os

import speechbrain
os.environ["SB_FETCHING_STRATEGY"] = "copy"

import types
import sys

_FAKE_SB_MODULES = [
    "speechbrain.integrations",
    "speechbrain.integrations.k2_fsa",
    "speechbrain.k2_integration",
    "speechbrain.integrations.huggingface",
    "speechbrain.integrations.huggingface.wordemb",
    "speechbrain.integrations.huggingface.wav2vec2",
    "speechbrain.integrations.huggingface.interface",
]

for mod in _FAKE_SB_MODULES:
    sys.modules[mod] = types.ModuleType(mod)

import speechbrain.utils.importutils as sb_importutils

original_ensure_module = sb_importutils.LazyModule.ensure_module

def safe_ensure_module(self, stacklevel=1):
    try:
        return original_ensure_module(self, stacklevel)
    except ImportError as e:
        print(f"[WARN] Lazy import failed for {self.target}, faking module.")
        sys.modules[self.target] = types.ModuleType(self.target)
        self.lazy_module = sys.modules[self.target]
        return self.lazy_module

sb_importutils.LazyModule.ensure_module = safe_ensure_module
"""
This patch prevents SpeechBrain's LazyModule import mechanism from failing
when certain modules are not available, allowing the script to run without raising ImportError.
The patch inject fake module placeholders into sys.modules for the specified modules.
It overrides the `ensure_module` method of `LazyModule` to catch ImportError exceptions"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
"""
Suppressing UserWarnings to avoid cluttering the output with warnings
"""
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
from speechbrain import Stage
from speechbrain.core import Brain
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.dataio.dataio import load_data_csv, merge_csvs
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.checkpoints import Checkpointer
from hyperpyyaml import load_hyperpyyaml

from speechbrain.inference import EncoderDecoderASR
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from torch.nn import Linear
from speechbrain.processing.features import STFT, Filterbank, Deltas, InputNormalization, spectral_magnitude

# Set FFT and filterbank params
n_fft = 400  # Match this in both STFT and Filterbank
win_length = 25
hop_length = 10
n_mels = 512
compute_stft = STFT(sample_rate=16000, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
compute_fbanks = Filterbank(n_mels=n_mels)
normalizer = InputNormalization()

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_DIR = BASE_DIR / "manifests"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR = BASE_DIR / "logs"
MODEL_SAVE_DIR = BASE_DIR / "model"
MODEL_NAME = "speechbrain/asr-transformer-transformerlm-librispeech"

LANGUAGE_MAP = {
    "zu"  : "zulu",
    "xh"  : "xhosa",
    "ss"  : "siswati",
    "nr"  : "ndebele",
}

def setup_logging(language_code):
    """
    Set up logging configuration.
    """
    log_dir = LOG_DIR / language_code
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{language_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            file_handler,
            stream_handler
        ]
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)


def load_hparams(language_code):
    """
    Load hyperparameters from YAML file.
    """
    yaml_file = BASE_DIR / "model" / "hyperparams" / "train.yaml"
    lang_name = LANGUAGE_MAP.get(language_code, language_code)
    overrides = {
        "lang_id": lang_name,
        "train_csv_file": str(MANIFEST_DIR / f"{lang_name}_train.csv"),
        "valid_csv_file": str(MANIFEST_DIR / f"{lang_name}_test.csv"),
        "output_root": str(MODEL_SAVE_DIR / f"{lang_name}_speechbrain"),
        "text_file": str(MANIFEST_DIR / f"{lang_name}_text.txt"),
    }
    with open(yaml_file, 'r') as file:
        hparams = load_hyperpyyaml(file, overrides=overrides)

    return hparams


class ASRBrain(Brain):
    def compute_forward(self, batch, stage):
        """
        Forward pass for the ASR model.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        token_bos = batch.tokens_bos
        if isinstance(token_bos, tuple):
            token_bos = token_bos[0]

        enc_out, _ = self.modules.encoder(wavs)

        token_bos = token_bos.to(enc_out.device)  # ensure same device
        token_emb = self.modules.embedding(token_bos)  # [time, batch, emb_size]

        predictions = self.modules.decoder(enc_out, token_emb)
        return predictions, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """
        Compute the loss for the ASR model.
        """
        preds, wav_lens = predictions
        tokens, token_lens = batch.tokens
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        if isinstance(token_lens, tuple):
            token_lens = token_lens[0]
        
        wav_lens = wav_lens.to(preds.device)  # ensure same device
        input_lens = torch.clamp((wav_lens * preds.size(1)), max=preds.size(1)).long()
        token_lens = token_lens.long()
        preds = preds.transpose(0, 1)  # [time, batch, output_neurons]
        # print("🧪 preds.shape:", preds.shape)
        # print("🧪 tokens.shape:", tokens.shape)
        # print("🧪 input_lens.shape:", input_lens.shape)
        # print("🧪 token_lens.shape:", token_lens.shape)
        batch_size = preds.shape[1]
        # print("✅ batch_size:", batch_size)
        assert input_lens.shape[0] == batch_size
        assert token_lens.shape[0] == batch_size
        assert tokens.shape[0] == batch_size
        loss = self.modules.ctc_loss(preds, tokens, input_lens, token_lens)

        if stage != Stage.TRAIN:
            predicted_words = self.hparams.tokenizer.decode_ids(torch.argmax(preds, dim=-1))
            target_words = self.hparams.tokenizer.decode_ids(tokens)
            self.error_stats.append(batch.id, predicted_words, target_words)
        return loss
    
    def on_stage_start(self, stage, epoch=None):

        if stage != Stage.TRAIN:
            self.error_stats = ErrorRateStats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage != Stage.TRAIN:
            wer = self.error_stats.compute_wer()
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"loss": stage_loss},
                valid_stats={"wer": wer}
            )
            with open(self.hparas["wer_output_dir"], "a") as f:
                f.write(f"{stage.name} WER: {wer:.4f}\n")




def train_model(language_code):
    """
    Train the ASR model for the specified language code.
    """
    logger = setup_logging(language_code)
    logger.info(f"Starting training for {language_code}...")
    hparams = load_hparams(language_code)
    print("✅ DEBUG: Text file for tokenizer:", hparams["text_file"])
    print("✅ DEBUG: Output dir:", hparams["output_root"])
    print(f"DEBUG: n_fft={n_fft}, win_length={win_length}, hop_length={hop_length}")

    tokenizer = SentencePiece(
        model_dir=hparams["output_root"],
        vocab_size=hparams["output_neurons"],
        text_file=hparams["text_file"],
    )
    hparams["tokenizer"] = tokenizer
    bos_index = tokenizer.sp.encode("<bos>", out_type=int)[0]  # BOS token
    eos_index = tokenizer.sp.encode("<eos>", out_type=int)[0]  # EOS token

    def audio_pipeline(wav):
        """
        Audio processing pipeline.
        """
        sig = speechbrain.dataio.dataio.read_audio(wav)
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)  # Ensure mono audio
        
        stft = compute_stft(sig) # shape [batch, time, freq, 2]
        magnitudes = spectral_magnitude(stft) # shape [batch, time, freq]
        fbanks = compute_fbanks(magnitudes) # shape [batch, time, n_mels]
        feats = normalizer(fbanks) 
        return feats.squeeze(0)  # Remove batch dimension

    def text_pipeline(transcript):
        """
        Text processing pipeline.
        """
        tokens_list = tokenizer.sp.encode(transcript, out_type=int)
        yield torch.LongTensor(tokens_list)
        yield torch.LongTensor([bos_index] + tokens_list)  # Length of the token
        yield torch.LongTensor([eos_index] + tokens_list)  # Length of the token
    
    hparams["audio_pipeline"] = audio_pipeline
    hparams["text_pipeline"] = text_pipeline
    hparams["modules"] = {
    "encoder": TransformerEncoder(
        input_shape=[None, hparams["sample_rate"]],
        d_model=hparams["emb_size"],
        nhead=hparams["nhead"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        d_ffn=hparams["hidden_size"]
    ),
    "decoder": TransformerDecoder(
        d_model=hparams["emb_size"],
        nhead=hparams["nhead"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        d_ffn=hparams["hidden_size"]
    ),
    "ctc_lin": Linear(
        in_features=hparams["emb_size"],
        out_features=hparams["output_neurons"],
    ),
    "embedding": torch.nn.Embedding(num_embeddings=hparams["output_neurons"], embedding_dim=hparams["emb_size"]),
    "ctc_loss": torch.nn.CTCLoss(blank=tokenizer.sp.piece_to_id("<blank>"), zero_infinity=True),
    }


    hparams["train_data"] = load_data_csv(hparams["train_csv"])
    hparams["test_data"] = load_data_csv(hparams["valid_csv"]
    )
    # Convert the data to DynamicItemDataset
    train_dataset = DynamicItemDataset(hparams["train_data"], output_keys=["id", "wav", "tokens", "tokens_bos", "tokens_eos"])
    valid_dataset = DynamicItemDataset(hparams["test_data"], output_keys=["id", "wav", "tokens", "tokens_bos", "tokens_eos"])


    # Apply dynamic pipeline from hparams
    train_dataset.add_dynamic_item(hparams["audio_pipeline"], takes=["wav"], provides=["sig"])
    train_dataset.add_dynamic_item(hparams["text_pipeline"], takes=["transcript"], provides=["tokens", "tokens_bos", "tokens_eos"])

    train_dataset.set_output_keys(["id", "sig", "tokens", "tokens_bos", "tokens_eos"])

    valid_dataset.add_dynamic_item(hparams["audio_pipeline"], takes=["wav"], provides=["sig"])
    valid_dataset.add_dynamic_item(hparams["text_pipeline"], takes=["transcript"], provides=["tokens", "tokens_bos", "tokens_eos"])
    valid_dataset.set_output_keys(["id", "sig", "tokens", "tokens_bos", "tokens_eos"])

    # Store into hparams
    hparams["train_data"] = train_dataset
    hparams["test_data"] = valid_dataset

    hparams["train_loader"] = SaveableDataLoader(
        dataset=hparams["train_data"],
        batch_size=hparams["batch_size"],
        shuffle=True,
    )

    hparams["test_loader"] = SaveableDataLoader(
        dataset=hparams["test_data"],
        batch_size=hparams["batch_size"],
    )

    asr_brain = ASRBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=Checkpointer(
            hparams["output_root"],
        )
    )

    asr_brain.tokenizer = tokenizer
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        hparams["train_data"],
        hparams["test_data"]
    )

    logger.info(f"Training completed for {language_code}. Model saved to {hparams['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR model for specified language.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., zu, xh, ss, nr)")
    args = parser.parse_args()
    train_model(args.language)