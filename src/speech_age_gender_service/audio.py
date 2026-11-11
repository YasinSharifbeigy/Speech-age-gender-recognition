import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

SAMPLE_RATE = 16000
MAX_SECONDS = 120


class InvalidAudio(ValueError):
    pass


def decode_audio(path: Path) -> np.ndarray:
    """Bound decoding to max duration + 1s; FFmpeg handles WAV, MP3 and FLAC."""
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
             "-i", str(path), "-t", str(MAX_SECONDS + 1), "-vn", "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
            capture_output=True, timeout=30, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise InvalidAudio("Cannot decode audio; provide a valid WAV, MP3 or FLAC file.") from exc
    wave = np.frombuffer(result.stdout, dtype="<f4").copy()
    if wave.size < SAMPLE_RATE:
        raise InvalidAudio("At least 1 second of audio is required.")
    if wave.size > MAX_SECONDS * SAMPLE_RATE:
        raise InvalidAudio(f"Audio must be at most {MAX_SECONDS} seconds.")
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) < 1e-5:
        raise InvalidAudio("Audio is silent or contains invalid samples.")
    return wave


def window_bounds(length: int):
    """Match upstream 4s windows / 2s shift / 1s minimum tail."""
    if length <= 4 * SAMPLE_RATE:
        yield 0, length
        return
    for start in range(0, length, 2 * SAMPLE_RATE):
        end = min(start + 4 * SAMPLE_RATE, length)
        if end - start >= SAMPLE_RATE:
            yield start, end
        if end == length:
            break
