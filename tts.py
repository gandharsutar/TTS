# Updated tts.py
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
import pyttsx3
import uuid
import os
from pathlib import Path

app = FastAPI()
OUTPUT_DIR = "tts_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "TTS Service is running"}

@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=filepath, filename=filename, media_type='audio/mpeg')

@app.get("/api/list-audio-files")
async def list_audio_files():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp3')]
    return {"audio_files": files, "count": len(files)}

# POST method to generate audio
@app.post("/api/generate")
async def text_to_speech(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        
        if not os.path.exists(filepath):
            raise HTTPException(status_code=500, detail="Audio generation failed")
            
        return JSONResponse({
            "status": "success",
            "audio_url": f"/api/audio/{filename}",
            "filename": filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.119", port=8001)