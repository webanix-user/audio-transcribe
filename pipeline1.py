import os
import time
import requests
from dotenv import load_dotenv
import google.generativeai as genai
import tempfile
import concurrent.futures
from pydub import AudioSegment # ### NEW ###

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


# --- Helper Functions (formatting, no API calls) ---

def _format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _build_diarized_transcript(segments: list) -> str:
    """Builds a formatted transcript string from diarized segments."""
    if not segments:
        return "⚠️ No diarized segments found."

    speaker_map = {}
    speaker_counter = 1

    def get_speaker_name(spk_id):
        nonlocal speaker_counter
        if spk_id not in speaker_map:
            speaker_map[spk_id] = f"Speaker {speaker_counter}"
            speaker_counter += 1
        return speaker_map[spk_id]

    lines = []
    for segment in segments:
        speaker = get_speaker_name(segment.get("speaker", "Unknown"))
        start_time = segment.get("start_time", 0.0)
        end_time = segment.get("end_time", 0.0)
        text = segment.get("text", "")
        lines.append(
            f"[{_format_time(start_time)} - {_format_time(end_time)}] {speaker}: {text}"
        )

    return "\n".join(lines)


def _build_diarized_transcript_from_tokens(tokens: list) -> str:
    """Builds a formatted transcript string from word tokens with speaker info."""
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

# ### NEW ### - Function to split audio into chunks
def split_audio(file_path: str, chunk_length_min: int = 45) -> list[str]:
    """
    Splits an audio file into chunks of a specified duration.

    Args:
        file_path (str): Path to the audio file.
        chunk_length_min (int): Desired length of each chunk in minutes.

    Returns:
        list[str]: A list of file paths to the temporary chunk files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    print(f"Loading audio file: {file_path}")
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = chunk_length_min * 60 * 1000
    
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        
        # Create a temporary file to store the chunk
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd) # Close file descriptor
        
        print(f"Exporting chunk {len(chunks) + 1} to {temp_path}")
        chunk.export(temp_path, format="wav")
        chunks.append(temp_path)
        
    print(f"Successfully split audio into {len(chunks)} chunks.")
    return chunks

# ### MODIFIED ### - Renamed and now returns raw data instead of formatted string
def transcribe_audio_chunk(
    file_path: str,
    api_key: str,
    model: str = "stt-async-preview",
    enable_speaker_diarization: bool = True,
    language_hints: list = None,
) -> list:
    """
    Transcribes a single audio chunk and returns the raw token data.
    This function handles upload, transcription, polling, and cleanup for one file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {api_key}"

    file_id = None
    transcription_id = None

    try:
        # 1. Upload File
        with open(file_path, "rb") as f:
            res = session.post(f"{API_BASE}/v1/files", files={"file": f})
        res.raise_for_status()
        file_id = res.json()["id"]
        
        # 2. Start Transcription
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

        # 3. Poll until complete
        while True:
            res = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}")
            res.raise_for_status()
            data = res.json()
            if data["status"] == "completed":
                break
            elif data["status"] == "error":
                raise Exception(f"Transcription failed: {data.get('error_message', 'Unknown error')}")
            time.sleep(2)

        # 4. Fetch Transcript Tokens (we need raw tokens for timestamp adjustment)
        res_tokens = session.get(f"{API_BASE}/v1/transcriptions/{transcription_id}/transcript")
        res_tokens.raise_for_status()
        return res_tokens.json().get("tokens", [])

    finally:
        # 5. Cleanup Resources
        if transcription_id:
            session.delete(f"{API_BASE}/v1/transcriptions/{transcription_id}")
        if file_id:
            session.delete(f"{API_BASE}/v1/files/{file_id}")


# --- Summarizer Function (Unchanged) ---
def summarizer(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Create directories if they don't exist
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


# ### NEW ### - Main orchestration pipeline for large files
def transcribe_and_summarize_large_audio(
    file_path: str,
    chunk_minutes: int = 45,
    max_workers: int = 4,
    language_hints: list = None
) -> tuple[str, str]:
    """
    Orchestrates the entire process for a large audio file:
    1. Splits the audio into chunks.
    2. Transcribes chunks in parallel.
    3. Merges transcripts, adjusting timestamps.
    4. Saves the final transcript.
    5. Generates and saves a summary.
    """
    chunk_files = []
    try:
        # 1. Split audio file into manageable chunks
        chunk_files = split_audio(file_path, chunk_length_min=chunk_minutes)
        chunk_duration_ms = chunk_minutes * 60 * 1000
        
        all_tokens = []
        
        # 2. Transcribe chunks in parallel
        print(f"\nStarting parallel transcription for {len(chunk_files)} chunks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each chunk transcription
            print("Chunk Given to executor")
            future_to_chunk_index = {
                executor.submit(transcribe_audio_chunk, chunk_path, SONIOX_API_KEY,language_hints=language_hints): i
                for i, chunk_path in enumerate(chunk_files)
            }
            print("chunk data received")
            # Process results as they complete
            results = [None] * len(chunk_files)
            for future in concurrent.futures.as_completed(future_to_chunk_index):
                chunk_index = future_to_chunk_index[future]
                try:
                    chunk_tokens = future.result()
                    results[chunk_index] = chunk_tokens
                    print(f"✅ Successfully transcribed chunk {chunk_index + 1}/{len(chunk_files)}")
                except Exception as exc:
                    print(f"❌ Chunk {chunk_index + 1} generated an exception: {exc}")
        
        # 3. Combine results and adjust timestamps
        print("\nMerging transcripts and adjusting timestamps...")
        for i, chunk_tokens in enumerate(results):
            if chunk_tokens:
                time_offset_ms = i * chunk_duration_ms
                for token in chunk_tokens:
                    token['start_ms'] += time_offset_ms
                    token['end_ms'] += time_offset_ms
                all_tokens.extend(chunk_tokens)
        
        # 4. Build final transcript from all tokens
        final_transcript = _build_diarized_transcript_from_tokens(all_tokens)
        
        # Ensure the transcript directory exists
        transcript_dir = os.path.dirname("static/transcript/transcript.txt")
        if not os.path.exists(transcript_dir):
            os.makedirs(transcript_dir)
            
        with open("static/transcript/transcript.txt", "w", encoding="utf-8") as f:
            f.write(final_transcript)
        print("\nFinal transcript saved to static/transcript/transcript.txt")
        
        # 5. Summarize the full transcript
        print("Generating summary...")
        final_summary = summarizer("static/transcript/transcript.txt")
        print("Summary saved to static/transcript/summary.txt")

        return final_transcript, final_summary

    finally:
        # 6. Clean up temporary chunk files
        print("\nCleaning up temporary audio chunks...")
        for path in chunk_files:
            try:
                os.remove(path)
                print(f"Removed {path}")
            except OSError as e:
                print(f"Error removing file {path}: {e}")
        print("Cleanup complete.")