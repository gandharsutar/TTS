from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import pyttsx3
import uuid
import os

app = FastAPI()
OUTPUT_DIR = "audio_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/tts")
def text_to_speech(text: str = Form(...)):
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Initialize TTS engine
    engine = pyttsx3.init()
    
    # Save to file
    engine.save_to_file(text, filepath)
    engine.runAndWait()

    return FileResponse(path=filepath, filename=filename, media_type='audio/mpeg')
