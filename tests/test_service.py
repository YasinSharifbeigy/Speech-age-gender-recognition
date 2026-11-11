import io
import threading
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from speech_age_gender_service.api import create_app
from speech_age_gender_service.audio import InvalidAudio, decode_audio, window_bounds


def wav_bytes(seconds=2, stereo=False, rate=16000, silent=False):
    buffer = io.BytesIO()
    x = np.zeros(int(seconds * rate)) if silent else np.sin(np.arange(int(seconds * rate)) * 440 * 2 * np.pi / rate) * 0.2
    if stereo:
        x = np.column_stack((x, x))
    with wave.open(buffer, "wb") as out:
        out.setnchannels(2 if stereo else 1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((x * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_decode_resamples_and_rejects_silence(tmp_path):
    import pytest
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes(stereo=True, rate=44100))
    assert decode_audio(path).shape == (32000,)
    path.write_bytes(wav_bytes(silent=True))
    with pytest.raises(InvalidAudio, match="silent"):
        decode_audio(path)
    path.write_bytes(b"broken")
    with pytest.raises(InvalidAudio, match="decode"):
        decode_audio(path)


def test_window_tails():
    assert list(window_bounds(64000)) == [(0, 64000)]
    assert list(window_bounds(96000)) == [(0, 64000), (32000, 96000)]
    assert list(window_bounds(112000))[-1] == (64000, 112000)


class FakeEngine:
    model = None
    device = "cpu"

    def __init__(self, gate=None, fail=False):
        self.gate, self.fail, self.path = gate, fail, None

    def predict(self, path):
        self.path = path
        if self.gate:
            self.gate.wait(timeout=5)
        if self.fail:
            raise InvalidAudio("No speech detected")
        return {"age": {"years": 30}, "gender": {"label": "male"}}


def test_sync_and_validation():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict", files={"file": ("a.wav", b"audio")})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert not engine.path.exists()
        assert client.get(response.headers["location"]).json()["result"]["age"]["years"] == 30
        assert client.post("/predict", files={"file": ("a.exe", b"x")}).status_code == 415
        assert client.post("/predict", files={"file": ("a.wav", b"")}).status_code == 400
        assert client.get("/track/missing").status_code == 404


def test_timeout_does_not_cancel_worker():
    gate = threading.Event()
    engine = FakeEngine(gate)
    with TestClient(create_app(engine)) as client:
        response = client.post("/predict?wait_seconds=0", files={"file": ("a.wav", b"audio")})
        assert response.status_code == 202
        gate.set()
        # A second synchronous request runs behind the first in the same queue.
        assert client.post("/predict", files={"file": ("b.wav", b"audio")}).status_code == 200
        assert client.get(response.headers["location"]).json()["status"] == "completed"


def test_invalid_audio_is_reported():
    with TestClient(create_app(FakeEngine(fail=True))) as client:
        response = client.post("/predict", files={"file": ("a.wav", b"audio")})
        assert response.status_code == 422
        assert response.json()["status"] == "failed"
