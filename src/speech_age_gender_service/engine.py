import json
import os
import threading
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

from .audio import InvalidAudio, SAMPLE_RATE, decode_audio, window_bounds
from .model import OragenAudioModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("SPEECH_MODEL_DIR", os.environ.get("ORAGEN_MODEL_DIR", ROOT / "models")))
BACKBONE = "facebook/wav2vec2-large-robust"
REPO = "markitantov/ORAGEN"


def download_models(model_dir: Path = MODEL_DIR):
    """Resolve immutable revisions once and record them alongside local artifacts."""
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "oragen_revision": HfApi().model_info(REPO).sha,
        "backbone_revision": HfApi().model_info(BACKBONE).sha,
        "upstream_code_revision": "2f5429aa8ad6d0d7ec0a7acb9ed5fd7e337def8d",
    }
    hf_hub_download(REPO, "audio_model.pt", revision=manifest["oragen_revision"], local_dir=model_dir)
    for name in ("config.json", "preprocessor_config.json"):
        hf_hub_download(BACKBONE, name, revision=manifest["backbone_revision"], local_dir=model_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class Engine:
    def __init__(self, device: str = "cpu", model_dir: Path = MODEL_DIR):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.model_dir = model_dir
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        for name in ("audio_model.pt", "config.json", "preprocessor_config.json"):
            if not (self.model_dir / name).is_file():
                raise RuntimeError("Model assets missing. Run: speech-age-gender download")
        config = Wav2Vec2Config.from_pretrained(self.model_dir, local_files_only=True)
        model = OragenAudioModel(config)
        checkpoint = torch.load(self.model_dir / "audio_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
        from silero_vad import load_silero_vad, get_speech_timestamps
        self.vad = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps
        # Silero's import sets this globally to 1; apply our setting afterwards.
        torch.set_num_threads(int(os.environ.get("SPEECH_CPU_THREADS", os.environ.get("ORAGEN_CPU_THREADS", "4"))))
        self.model = model.to(self.device).eval()

    def predict(self, path: Path) -> dict:
        waveform = decode_audio(path)
        with self.lock, torch.inference_mode():
            self.load()
            speech = self.get_speech_timestamps(torch.from_numpy(waveform), self.vad, sampling_rate=SAMPLE_RATE)
            if not speech:
                raise InvalidAudio("No speech detected in the recording.")
            predictions = []
            for start, end in window_bounds(len(waveform)):
                if not any(s["start"] < end and s["end"] > start for s in speech):
                    continue
                values = self.processor(
                    waveform[start:end], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                    padding="max_length", max_length=4 * SAMPLE_RATE, return_attention_mask=False,
                ).input_values.to(self.device)
                output = self.model(values)
                probs = output["gen"].softmax(-1)[0].cpu().tolist()
                age = float(output["age"].sigmoid()[0].cpu()) * 100
                predictions.append({"start_seconds": start / SAMPLE_RATE,
                                    "end_seconds": end / SAMPLE_RATE,
                                    "age_years": age, "female": probs[0], "male": probs[1]})
            if not predictions:
                raise InvalidAudio("No usable speech windows detected.")
        age = float(np.mean([p["age_years"] for p in predictions]))
        probs = {label: float(np.mean([p[label] for p in predictions])) for label in ("female", "male")}
        # A display bucket, not a statistically calibrated uncertainty interval.
        lower = min(int(age // 10) * 10, 90)
        return {"model": REPO, "device": str(self.device),
                "age": {"years": round(age, 2), "range": [lower, lower + 9],
                        "range_type": "decade_bucket"},
                "gender": {"label": max(probs, key=probs.get), "probabilities": probs},
                "duration_seconds": round(len(waveform) / SAMPLE_RATE, 3),
                "num_windows": len(predictions), "windows": predictions}
