"""Exercise actual model inference through HTTP, using supplied single-speaker media.

Run from project root: .venv/Scripts/python.exe scripts/smoke_test.py recording.wav
No network access or additional checkpoints needed after `speech-age-gender download`.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from fastapi.testclient import TestClient

from oragen_service.api import create_app
from oragen_service.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    engine = Engine()
    results = []
    with tempfile.TemporaryDirectory() as directory, TestClient(create_app(engine)) as client:
        for extension in ("wav", "mp3"):
            path = Path(directory) / f"sample.{extension}"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error",
                            "-i", str(args.audio), "-vn", "-ar", "16000", "-ac", "1", str(path)], check=True)
            start = time.monotonic()
            with path.open("rb") as stream:
                response = client.post("/predict", files={"file": (path.name, stream)})
            if response.status_code == 202:
                track_url = response.headers["location"]
                while True:
                    response = client.get(track_url)
                    if response.json()["status"] not in {"queued", "processing"}:
                        break
                    time.sleep(0.1)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "completed", body
            prediction = body["result"]
            assert 0 <= prediction["age"]["years"] <= 100
            assert abs(sum(prediction["gender"]["probabilities"].values()) - 1) < 1e-5
            assert client.get(f"/track/{body['track_id']}").json() == body
            results.append({"format": extension, "elapsed_seconds": round(time.monotonic() - start, 3),
                            "result": prediction})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
