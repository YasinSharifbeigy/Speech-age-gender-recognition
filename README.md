# Speech-age-gender-recognition

Upload a single-speaker recording and receive JSON with an estimated age, a
decade bucket, female/male scores, and predictions for each speech window.
The service processes audio directly; transcription is not required.

Clone the [GitHub repository](https://github.com/YasinSharifbeigy/Speech-age-gender-recognition)
into any directory and enter the project root before running the commands below:

```sh
git clone https://github.com/YasinSharifbeigy/Speech-age-gender-recognition.git
cd Speech-age-gender-recognition
```

If you already cloned it, enter your existing clone instead. All paths in this
README are relative to the project root unless stated otherwise.

## Model overview

This implements the **audio branch of ORAGEN**, using the public
[`markitantov/ORAGEN`](https://huggingface.co/markitantov/ORAGEN) `audio_model.pt`
checkpoint. Its backbone is the 24-layer **Wav2Vec2-Large-Robust** speech encoder,
followed by two attention blocks, statistical pooling, and a joint age/gender
head. It does not use the image branch or audio/video fusion model.

Audio is converted to 16 kHz mono. Silero VAD detects speech, and the service
analyzes 4-second windows with a 2-second shift (minimum tail: 1 second).
The backbone feature extractor normalizes/pads each window. Gender logits use
softmax in female/male order; age uses sigmoid multiplied by 100. Predictions
are averaged over windows containing speech.

The model card reports age MAE of **6.90 years on TIMIT** and **10.47 years on
Common Voice**, with gender UAR of **98.60%** and **92.59%**, respectively.
These are the authors' benchmark results, not guaranteed service accuracy.
Persian accuracy remains unvalidated. A local smoke test produced ages 26.18
and 35.42 for WAV and MP3 versions of the same recording; both were classified
female. Evaluate compression and recording conditions on your deployment data.

Architecture source: [markitantov/chimera_ml](https://github.com/markitantov/chimera_ml),
commit `2f5429aa8ad6d0d7ec0a7acb9ed5fd7e337def8d`. Adapted code is MIT licensed;
see [ORAGEN-LICENSE.txt](src/speech_age_gender_service/ORAGEN-LICENSE.txt). The checkpoint
card does not specify a license; the code license alone does not establish
commercial permission for the weights.

## Hardware and GPU VRAM

| Deployment | Resource guidance |
| --- | --- |
| CPU (default) | **No GPU required; 0 GB VRAM.** Start with 4 CPU cores and 8 GB system RAM. |
| NVIDIA GPU | Plan for **4-6 GB free VRAM** for one model and one job at a time; **6 GB free is the recommended initial budget**. This is an estimate, not a measured minimum. |
| Storage | Approximately 678 MB for the audio checkpoint, plus container image and package caches. CUDA images are substantially larger than CPU images. |

The GPU estimate includes roughly 1.3 GB for float32 model weights, plus
activations, CUDA libraries, and allocator overhead. Reduced-precision checkpoint
weights are loaded into float32. Actual peak depends on the GPU, PyTorch/CUDA
versions, and other processes. GPU VRAM has **not been measured** here: the
installed PyTorch build is CPU-only.

Only one window runs at a time. Longer recordings create more sequential
windows. Additional replicas load additional model copies. Use the included
`scripts/measure_vram.py` on the target CUDA deployment to measure PyTorch's peak
allocated/reserved memory; driver/context memory and other processes add overhead.

For reference, a 12.2-second sample took about 13.7 seconds including model load,
then 6.9 seconds for a subsequent request on the local CPU. These are smoke-test
timings, not a throughput benchmark.

## Docker: CPU deployment

Requires Docker Engine with Compose, or Docker Desktop using Linux containers.
From the project root:

```sh
docker compose build
docker compose run --rm download
docker compose up -d api
docker compose logs -f api
```

Setup needs internet access. It downloads weights/configuration into the project
directory `./models`, recording immutable Hugging Face revisions in
`models/manifest.json`. Later setup runs reuse these revisions. Compose mounts
that directory as `/models` inside the container. Weights are excluded from the
image/build context; runtime mounts the directory read-only and does not download
additional models. Run setup before the first prediction. The `models` folder
is excluded from Git except for `.gitkeep`, so a new clone needs setup too.

On Linux, before `docker compose run --rm download`, set the container download
user to the host user who owns the project folder:

```sh
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
```

This lets the non-root download process write to `./models`. On Docker Desktop
for Windows the default UID/GID usually works; ensure the project directory is
shared with Docker Desktop if a bind-mount permission error occurs.

- Interactive docs and upload form: http://localhost:8000/docs
- OpenAPI specification: http://localhost:8000/openapi.json
- Process health: http://localhost:8000/health

The host port binds to loopback by default. Services in the same Compose network
should use **`http://api:8000`**. For your main project, include the API service
and model bind mount in its Compose configuration, or attach both services to the
same Docker network. Change the host port binding explicitly for remote access.

`docker compose down` preserves model weights in `./models`, including when
`-v` is supplied, because this folder is a bind mount rather than a volume.

If Docker reports `403 Forbidden` while fetching a Docker Hub token, check
`docker pull python:3.12-slim` independently of the project. Sign in with
`docker login` if appropriate, and check the Docker Desktop proxy and any
organization registry-access policy with your administrator. The build also
needs network access to the package repositories, and model setup needs Hugging
Face access. Earlier versions of this Dockerfile requested the external
`docker/dockerfile:1` frontend; this version uses the bundled frontend, but it
still needs access to the Python base image unless that image is cached.

Earlier Compose versions stored model files in a Docker-managed `speech-models`
or `oragen-models` volume. Existing volumes are left untouched and are not
automatically copied to `./models`. Run the download step to populate the project
folder, or copy the existing volume's files into it before starting the service.

To reuse an existing volume on Windows instead of downloading again, run the
following from the project root, substituting the actual name from `docker volume
ls` for `YOUR_OLD_VOLUME`:

```powershell
$oldVolume = 'YOUR_OLD_VOLUME'
$modelsPath = (Resolve-Path .\models).Path
docker run --rm --mount "type=volume,source=$oldVolume,target=/old,readonly" --mount "type=bind,source=$modelsPath,target=/new" python:3.12-slim sh -c "cp -R /old/. /new/"
```

This copies the files and leaves the old volume intact. Then restart the API
with `docker compose up -d api` using the updated Compose configuration.

## Docker: NVIDIA GPU deployment

Requires an NVIDIA GPU, a driver compatible with CUDA 12.8, and NVIDIA Container
Toolkit configured for Docker. On Windows, use Docker Desktop's WSL2 GPU support.
The override installs CUDA 12.8 PyTorch wheels and reserves one GPU:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.gpu.yaml run --rm download
docker compose -f compose.yaml -f compose.gpu.yaml up -d api
```

Check `/health`: `device` should be `cuda`. `model_loaded` becomes true after
successful initialization on the first prediction. Explicit CUDA configuration
fails inference if CUDA is unavailable; it does not silently fall back to CPU.

Measure VRAM using a representative recording (Linux/macOS shell):

```sh
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps \
  --entrypoint python -v "$(pwd)/samples:/samples:ro" \
  api scripts/measure_vram.py /samples/recording.wav
```

Replace the samples path with your recording directory. In PowerShell put the
command on one line and use an absolute Windows path for the samples directory.

## Request format

**`POST /predict`**, content type **`multipart/form-data`**.

| Field | Location | Required | Meaning |
| --- | --- | --- | --- |
| `file` | Multipart form | Yes | Audio bytes with a WAV, MP3, FLAC, OGG, or M4A filename extension. |
| `wait_seconds` | Query string | No | Wait 0-60 seconds for completion; default **60**. Set 0 for immediate tracking. |

Limits: **25 MiB**, **1-120 seconds** decoded duration, **one speaker** per file.
Supported sample rates/channel counts are converted to 16 kHz mono. Silent files
and recordings without detected speech are rejected. Send file bytes, not a path
string, URL, JSON, or base64 value.

```sh
curl -F 'file=@recording.wav' http://localhost:8000/predict
curl -F 'file=@recording.mp3' 'http://localhost:8000/predict?wait_seconds=0'
```

On PowerShell use `curl.exe` with the same arguments.

Python client (install `requests` in the calling application):

```python
import time
import requests

base_url = 'http://localhost:8000'  # http://api:8000 inside the same Compose network
with open('recording.wav', 'rb') as audio:
    response = requests.post(
        f'{base_url}/predict',
        files={'file': ('recording.wav', audio, 'audio/wav')},
        params={'wait_seconds': 60},
        timeout=90,  # Longer than the server's 60-second wait.
    )
response.raise_for_status()
body = response.json()
while body['status'] in {'queued', 'processing'}:
    time.sleep(1)
    response = requests.get(f"{base_url}/track/{body['track_id']}", timeout=10)
    response.raise_for_status()
    body = response.json()
if body['status'] == 'failed':
    raise RuntimeError(body['error'])
print(body['result'])
```

## Response format

**HTTP 200** from `/predict` means inference completed. Illustrative response:

```json
{
  "track_id": "7e8c4abdbbcd40d3a8c2effd77928542",
  "status": "completed",
  "result": {
    "model": "markitantov/ORAGEN",
    "device": "cpu",
    "age": {"years": 29.02, "range": [20, 29], "range_type": "decade_bucket"},
    "gender": {
      "label": "male",
      "probabilities": {"female": 0.03, "male": 0.97}
    },
    "duration_seconds": 4.0,
    "num_windows": 1,
    "windows": [
      {"start_seconds": 0.0, "end_seconds": 4.0, "age_years": 29.02,
       "female": 0.03, "male": 0.97}
    ]
  }
}
```

| Output | Meaning |
| --- | --- |
| `track_id` | Request identifier for polling. |
| `age.years` | Mean estimated age across speech windows, rounded to two decimals. |
| `age.range` | Inclusive integer decade bucket; **not a confidence interval or error bound**. |
| `gender.label` | Highest mean score: `female` or `male`. |
| `gender.probabilities` | Mean softmax scores between 0 and 1; not calibrated certainty or proof of gender identity. |
| `duration_seconds` | Total decoded duration including silence. |
| `num_windows` | Number of speech windows used for inference. |
| `windows` | Per-window time boundaries in seconds, age estimate, and gender scores. |

**HTTP 202** means the job continues after the requested wait; it is not a
failure. The `Location` header gives a polling path, e.g. `/track/TRACK_ID`:

```json
{"track_id": "7e8c4abdbbcd40d3a8c2effd77928542", "status": "processing"}
```

Poll **`GET /track/{track_id}`**. Existing jobs return HTTP 200 and `status`
equal to `queued`, `processing`, `completed`, or `failed`. Completed responses
have the structure above. Failed jobs return, for example:

```json
{
  "track_id": "7e8c4abdbbcd40d3a8c2effd77928542",
  "status": "failed",
  "error": "No speech detected in the recording."
}
```

The tracking endpoint returns HTTP 200 even for a failed job; inspect `status`.
Inference continues when the synchronous HTTP wait expires.

## Errors and operational limits

| HTTP status | Meaning |
| --- | --- |
| 400 | Empty upload. |
| 404 | Unknown or expired tracking ID. |
| 413 | Upload exceeds 25 MiB. |
| 415 | Unsupported filename extension. |
| 422 | Invalid form/query parameters, undecodable audio, invalid duration, silence, or no speech. |
| 500 | Inference failure, such as missing weights or unavailable configured GPU; see server logs. |
| 503 | Job capacity reached; retry later. |

Errors before queuing use FastAPI's `{"detail": ...}` format. Errors within
submitted jobs use `track_id`, `status: "failed"`, and `error`. Query validation
errors use a list under `detail`. During the synchronous wait, invalid audio
returns 422 and other inference failures return 500. Later failures are visible
through tracking.

- One inference job runs at a time; other jobs queue.
- Maximum 32 retained jobs, **including completed jobs**. Completed jobs expire
  one hour after submission; cleanup occurs on later prediction/tracking calls.
- Jobs/results live in process memory. Restarting loses tracking IDs and pending
  work. Model files persist in `./models`.
- Run **one Uvicorn worker**. Multiple replicas need a shared durable job store
  or routing that sends tracking requests to the original instance.
- Temporary uploads are deleted after processing. Multiple speakers are not
  separated.
- `/health` reports process health, selected device, and `model_loaded`. It does
  not establish model readiness before the first prediction. Docker's health
  check uses it. Allow for first-request model loading time.
- This service has no authentication/TLS; integrate through your application's
  authenticated gateway for shared deployment.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `SPEECH_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto`. CUDA requires CUDA-enabled PyTorch wheels. |
| `SPEECH_MODEL_DIR` | Local `models/`; `/models` in Docker | Checkpoint/configuration/feature extractor and revision manifest directory. |
| `SPEECH_CPU_THREADS` | `4` | PyTorch CPU thread count. |

For compatibility, the former `ORAGEN_DEVICE`, `ORAGEN_MODEL_DIR`, and
`ORAGEN_CPU_THREADS` environment variables still work when their corresponding
`SPEECH_` variable is unset. The new names take precedence.

## Local development and verification

Requires Python 3.12 and `uv`. Run from the project root; `uv sync` creates the
virtual environment in the project's `.venv` directory. This is separate from
Docker's Linux Python environment, which is installed inside the Docker image.
A Docker build does not create a Windows `.venv` on the host.

```powershell
uv sync --extra test
.venv\Scripts\speech-age-gender.exe download
.venv\Scripts\speech-age-gender.exe serve
.venv\Scripts\speech-age-gender.exe predict recording.wav
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/smoke_test.py recording.wav
```

On Linux or macOS, use `.venv/bin/speech-age-gender` and `.venv/bin/python`
instead of the Windows `.venv\Scripts\` paths above.

With local Python, `download` also writes to
the project's `models` directory by default; it uses the same model files as the
Docker bind mount. The virtual environment and model weights are intentionally
excluded from Git and must be set up separately on each machine.

For local GPU inference use `speech-age-gender predict recording.wav --device cuda` in a
CUDA-enabled environment. Tests cover decoding/resampling, silent/invalid audio,
window boundaries, uploads, synchronous responses, tracking, and continuation
after timeout. The smoke helper exercises real WAV and MP3 HTTP inference.
