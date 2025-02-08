from fastapi import FastAPI, File, UploadFile
import shutil
import os
from faster_whisper import WhisperModel

app = FastAPI()

# Ensure necessary directories exist
UPLOAD_DIR = "uploads"
LOG_FILE = "transcriptions.log"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)  # Ensure model cache directory exists

# Load Whisper model at startup
print("Loading Whisper model...")
model = WhisperModel("medium", device="cuda", compute_type="float16", download_root="./models")
print("Model loaded successfully.")

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    """Handles file uploads and saves them to the server."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename}

@app.post("/transcribe/")
async def transcribe_audio(filename: str):
    """Transcribes the uploaded audio file and logs the output."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}

    try:
        segments, _ = model.transcribe(file_path, vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
        transcription = " ".join([segment.text for segment in segments])

        # Log transcription
        with open(LOG_FILE, "a") as log:
            log.write(f"{filename}: {transcription}\n")

        # Remove the audio file after transcription (optional)
        os.remove(file_path)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}
