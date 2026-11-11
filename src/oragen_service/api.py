import asyncio
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .audio import InvalidAudio
from .engine import Engine

MAX_UPLOAD = 25 * 1024 * 1024
MAX_JOBS = 32
JOB_TTL = 3600
logger = logging.getLogger(__name__)


def create_app(engine=None):
    predictor = engine or Engine(os.environ.get("SPEECH_DEVICE", os.environ.get("ORAGEN_DEVICE", "cpu")))
    jobs: dict[str, tuple[float, Future]] = {}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-age-gender")

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="Speech Age and Gender Service", version="0.1.0", lifespan=lifespan)

    def run(path):
        try:
            return predictor.predict(path)
        finally:
            path.unlink(missing_ok=True)

    def payload(job_id):
        future = jobs[job_id][1]
        if not future.done():
            return {"track_id": job_id, "status": "processing" if future.running() else "queued"}
        error = future.exception()
        if error:
            return {"track_id": job_id, "status": "failed",
                    "error": str(error) if isinstance(error, InvalidAudio) else "Inference failed; check server logs."}
        return {"track_id": job_id, "status": "completed", "result": future.result()}

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": predictor.model is not None,
                "device": str(predictor.device)}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), wait_seconds: float = Query(60, ge=0, le=60)):
        now = time.monotonic()
        for key in list(jobs):
            created, future = jobs[key]
            if future.done() and now - created > JOB_TTL:
                del jobs[key]
        if len(jobs) >= MAX_JOBS:
            raise HTTPException(503, "Job capacity reached. Retry later.")
        suffix = Path(file.filename or "audio.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            raise HTTPException(415, "Supported formats: WAV, MP3, FLAC, OGG, M4A.")
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
                path = Path(stream.name)
                total = 0
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "Upload exceeds 25 MiB.")
                    stream.write(chunk)
            if not total:
                raise HTTPException(400, "Empty upload.")
            # Recheck after awaits: simultaneous uploads must not bypass the bound.
            if len(jobs) >= MAX_JOBS:
                raise HTTPException(503, "Job capacity reached. Retry later.")
            job_id = uuid.uuid4().hex
            future = executor.submit(run, path)
            jobs[job_id] = (time.monotonic(), future)
            path = None  # Worker now owns cleanup, even after HTTP timeout/disconnect.
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            await file.close()
        def log_error(done):
            error = done.exception()
            if error and not isinstance(error, InvalidAudio):
                logger.error("Job %s failed", job_id, exc_info=(type(error), error, error.__traceback__))
        future.add_done_callback(log_error)
        wrapped = asyncio.wrap_future(future)
        # asyncio.wait does not cancel inference when the response timeout expires.
        await asyncio.wait({wrapped}, timeout=wait_seconds)
        # Retrieve exceptions to avoid unhandled asyncio Future warnings.
        wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        body = payload(job_id)
        code = 202
        if body["status"] == "completed":
            code = 200
        elif body["status"] == "failed":
            code = 422 if isinstance(future.exception(), InvalidAudio) else 500
        return JSONResponse(body, status_code=code, headers={"Location": f"/track/{job_id}"})

    @app.get("/track/{track_id}")
    async def track(track_id: str):
        if track_id not in jobs:
            raise HTTPException(404, "Unknown or expired track ID.")
        created, future = jobs[track_id]
        if future.done() and time.monotonic() - created > JOB_TTL:
            del jobs[track_id]
            raise HTTPException(404, "Unknown or expired track ID.")
        return payload(track_id)

    return app


app = create_app()
