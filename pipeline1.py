import os
import time
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import tempfile
import concurrent.futures
from pydub import AudioSegment
import logging

# --- Configuration ---
load_dotenv()
SONIOX_API_KEY = os.environ.get("SONIOX_API_KEY")
API_BASE = "https://api.soniox.com"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing Google API Key in .env file!")
if not SONIOX_API_KEY:
    raise ValueError("Missing SONIOX_API_KEY in .env file!")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (formatting, no API calls) ---

def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def _build_diarized_transcript_from_tokens(tokens: list) -> str:
    if not tokens:
        return "⚠️ No tokens found."

    transcript_lines = []
    current_speaker = None
    current_text = []
    block_start_time = None
    block_end_time = None

    speaker_map = {}
    speaker_counter = 1

    def get_speaker_name(spk_id):
        nonlocal speaker_counter
        if spk_id not in speaker_map:
            speaker_map[spk_id] = f"Speaker {speaker_counter}"
            speaker_counter += 1
        return speaker_map[spk_id]

    for token in tokens:
        spk_raw = token.get("speaker")
        spk_name = get_speaker_name(spk_raw) if spk_raw is not None else "Speaker Unknown"
        token_start = token.get("start_ms", 0) / 1000.0
        token_end = token.get("end_ms", 0) / 1000.0
        token_text = token.get("text", "")

        if spk_name != current_speaker:
            if current_text:
                transcript_lines.append(
                    f"[{_format_time(block_start_time)} - {_format_time(block_end_time)}] {current_speaker}: {''.join(current_text).strip()}"
                )
            current_speaker = spk_name
            block_start_time = token_start
            current_text = []

        block_end_time = token_end
        current_text.append(token_text)

    if current_text:
        transcript_lines.append(
            f"[{_format_time(block_start_time)} - {_format_time(block_end_time)}] {current_speaker}: {''.join(current_text).strip()}"
        )

    return "\n".join(transcript_lines)

def split_audio(file_path: str, chunk_length_min: int = 45) -> list[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logging.info(f"Loading audio file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = chunk_length_min * 60 * 1000
    
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        logging.info(f"Exporting chunk {len(chunks) + 1} to {temp_path}")
        chunk.export(temp_path, format="wav")
        chunks.append(temp_path)
        
    logging.info(f"Successfully split audio into {len(chunks)} chunks.")
    return chunks

def transcribe_audio_chunk(
    file_path: str,
    api_key: str,
    model: str = "stt-async-preview",
    enable_speaker_diarization: bool = True,
    language_hints: list = None,
) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"

    file_id = None
    transcription_id = None

    try:
        with open(file_path, "rb") as f:
            res = session.post(f"{API_BASE}/v1/files", files={"file": f})
        res.raise_for_status()
        file_id = res.json()["id"]
        
        payload = {
            "file_id": file_id,
            "model": model,
            "enable_speaker_diarization": enable_speaker_diarization,
        }
        if language_hints:
            payload["language_hints"] = language_hints
            
        res = session.post(f"{API_BASE}/v1/transcriptions", json=payload)
        res.raise_for_status()
        transcription_id = res.json()["id"]

        while True:
            res = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}")
            res.raise_for_status()
            data = res.json()
            if data["status"] == "completed":
                break
            elif data["status"] == "error":
                raise Exception(f"Transcription failed: {data.get('error_message', 'Unknown error')}")
            time.sleep(2)

        res_tokens = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}/transcript")
        res_tokens.raise_for_status()
        return res_tokens.json().get("tokens", [])

    finally:
        if transcription_id:
            session.delete(f"{API_BASE}/v1/transcriptions/{transcription_id}")
        if file_id:
            session.delete(f"{API_BASE}/v1/files/{file_id}")

def summarizer(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    summary_dir = os.path.dirname("static/transcript/summary.txt")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    prompt = f"""
    You will be given the transcription of a meeting. Your task is to provide a concise summary of the key discussion points, decisions made, and action items assigned.
    
    TRANSCRIPTION:
    {text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    summary_text = response.text

    with open("static/transcript/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text

def transcribe_and_summarize_large_audio(
    file_path: str,
    chunk_minutes: int = 15,
    max_workers: int = 2,
    language_hints: list = None
) -> tuple[str, str]:
    chunk_files = []
    try:
        chunk_files = split_audio(file_path, chunk_length_min=chunk_minutes)
        chunk_duration_ms = chunk_minutes * 60 * 1000
        
        all_tokens = []
        
        logging.info(f"Starting parallel transcription for {len(chunk_files)} chunks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk_index = {
                executor.submit(transcribe_audio_chunk, chunk_path, SONIOX_API_KEY,language_hints=language_hints): i
                for i, chunk_path in enumerate(chunk_files)
            }
            results = [None] * len(chunk_files)
            errors = []
            for future in concurrent.futures.as_completed(future_to_chunk_index):
                chunk_index = future_to_chunk_index[future]
                try:
                    chunk_tokens = future.result()
                    results[chunk_index] = chunk_tokens
                    logging.info(f"✅ Successfully transcribed chunk {chunk_index + 1}/{len(chunk_files)}")
                except Exception as exc:
                    errors.append((chunk_index, str(exc)))
                    logging.error(f"❌ Chunk {chunk_index + 1} failed: {exc}")

            if all(r is None for r in results):
                raise RuntimeError("All audio chunks failed to transcribe. Check Soniox API or audio format.")
        
        logging.info("Merging transcripts and adjusting timestamps...")
        for i, chunk_tokens in enumerate(results):
            if chunk_tokens:
                time_offset_ms = i * chunk_duration_ms
                for token in chunk_tokens:
                    token['start_ms'] += time_offset_ms
                    token['end_ms'] += time_offset_ms
                all_tokens.extend(chunk_tokens)
        
        final_transcript = _build_diarized_transcript_from_tokens(all_tokens)
        
        transcript_dir = os.path.dirname("static/transcript/transcript.txt")
        if not os.path.exists(transcript_dir):
            os.makedirs(transcript_dir)
            
        with open("static/transcript/transcript.txt", "w", encoding="utf-8") as f:
            f.write(final_transcript)
        logging.info("Final transcript saved to static/transcript/transcript.txt")
        
        logging.info("Generating summary...")
        final_summary = summarizer("static/transcript/transcript.txt")
        logging.info("Summary saved to static/transcript/summary.txt")

        return final_transcript, final_summary

    finally:
        logging.info("Cleaning up temporary audio chunks...")
        for path in chunk_files:
            try:
                os.remove(path)
                logging.info(f"Removed {path}")
            except OSError as e:
                logging.error(f"Error removing file {path}: {e}")
        logging.info("Cleanup complete.")