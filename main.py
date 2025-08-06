import os
import shutil
import tempfile
import socket
import uuid
import uvicorn
import json
from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your pipeline and API key
from pipeline1 import transcribe_and_summarize_large_audio, SONIOX_API_KEY

# --- Validate API Key ---
if not SONIOX_API_KEY:
    raise RuntimeError("SONIOX_API_KEY is not set. Please check your .env file.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Accessible on your LAN at: http://{local_ip}:8000")
    except socket.gaierror:
        logging.info("Could not determine local IP.")
    logging.info("Frontend: http://127.0.0.1:8000/static/index.html")
    yield

# --- App Setup ---
app = FastAPI(
    title="Soniox + Gemini Transcription & Summary",
    version="2.0",
    lifespan=lifespan
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Serve static files (transcript + summary) ---
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static/transcript", exist_ok=True)

JOB_DIR = "jobs"
os.makedirs(JOB_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/static/index.html">')

# --- POST endpoint using your imported pipeline ---
# @app.post("/transcribe-and-summarize")
# async def transcribe_and_summarize(file: UploadFile = File(...)):
#     temp_dir = tempfile.mkdtemp()
#     try:
#         temp_file_path = os.path.join(temp_dir, file.filename)
#         with open(temp_file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         logging.info(f"Starting Soniox + Gemini pipeline for {file.filename}...")

#         # Run your actual pipeline function from pipeline1
#         try:
#             transcript, summary = await run_in_threadpool(
#                 transcribe_and_summarize_large_audio,
#                 file_path=temp_file_path,
#                 language_hints=["en", "hi"]
#             )
#         except Exception as e:
#             logging.error(f"Pipeline error: {e}")
#             return JSONResponse(
#                 status_code=500,
#                 content={"error": "Pipeline failed", "details": str(e)}
#             )

#         logging.info("Pipeline completed.")

#         return JSONResponse({
#             "message": "Transcription and summarization complete.",
#             "transcript_text": transcript,
#             "summary_text": summary,
#             "download_links": {
#                 "transcript": "/static/transcript/transcript.txt",
#                 "summary": "/static/transcript/summary.txt"
#             }
#         })

#     finally:
#         await file.close()
#         shutil.rmtree(temp_dir)


def background_pipeline_runner(file_path: str, job_id: str, language_hints=None):
    job_status_path = os.path.join(JOB_DIR, f"job_{job_id}_status.json")
    try:
        # Set job as running
        with open(job_status_path, "w") as f:
            json.dump({"status": "processing"}, f)

        transcript, summary = transcribe_and_summarize_large_audio(
            file_path=file_path,
            language_hints=language_hints
        )

        # Save status and result
        with open(job_status_path, "w") as f:
            json.dump({
                "status": "complete",
                "result": {
                    "transcript_path": "/static/transcript/transcript.txt",
                    "summary_path": "/static/transcript/summary.txt"
                }
            }, f)

    except Exception as e:
        with open(job_status_path, "w") as f:
            json.dump({"status": "error", "error": str(e)}, f)


@app.post("/transcribe-and-summarize")
async def transcribe_and_summarize(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        job_id = str(uuid.uuid4())

        background_tasks.add_task(
            run_in_threadpool,
            background_pipeline_runner,
            temp_file_path,
            job_id,
            ["en", "hi"]
        )

        return {"status": "processing", "job_id": job_id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        await file.close()


@app.get("/transcription-status/{job_id}")
async def check_status(job_id: str):
    job_status_path = os.path.join(JOB_DIR, f"job_{job_id}_status.json")
    if not os.path.exists(job_status_path):
        return JSONResponse(status_code=404, content={"error": "Job ID not found"})

    with open(job_status_path, "r") as f:
        return json.load(f)


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

