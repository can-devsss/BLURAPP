from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import json

# Render ve lokal yapı uyumu için mutlak import (Absolute Import)
try:
    from backend.processor import BlurProcessor
except ImportError:
    from processor import BlurProcessor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html bulunamadı!</h1>"

@app.post("/api/process-media")
async def process_media(
    file: UploadFile = File(...),
    intensity: int = Form(...),
    media_type: str = Form(...),
    box: str = Form(None)  # Bölgeli blur için JSON koordinatlar
):
    file_extension = os.path.splitext(file.filename)[1] or (".mp4" if media_type == "video" else ".jpg")
    unique_name = f"{uuid.uuid4()}{file_extension}"
    
    input_path = os.path.join(UPLOAD_DIR, f"orig_{unique_name}")
    output_filename = f"blured_{unique_name}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    box_data = None
    if box:
        try:
            box_data = json.loads(box)
        except:
            pass

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if media_type == "video":
            BlurProcessor.apply_video_blur(input_path, output_path, intensity, box_data)
        else:
            BlurProcessor.apply_progressive_blur(input_path, output_path, intensity, box_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "download_url": f"/api/download/{output_filename}"}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type='application/octet-stream')
    raise HTTPException(status_code=404, detail="Dosya bulunamadı!")
