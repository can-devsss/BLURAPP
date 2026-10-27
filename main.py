from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
import json
import mimetypes

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
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "index.html"),
        os.path.join(os.path.dirname(__file__), "../frontend/index.html"),
        os.path.join(os.path.dirname(__file__), "frontend/index.html"),
        os.path.join(os.getcwd(), "index.html"),
        os.path.join(os.getcwd(), "frontend/index.html"),
        "index.html"
    ]
    
    for html_path in possible_paths:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
                
    return "<h1>index.html bulunamadı!</h1>"

@app.post("/api/process-media")
async def process_media(
    file: UploadFile = File(...),
    intensity: float = Form(...),
    media_type: str = Form(...),
    box: str = Form(None)
):
    raw_ext = os.path.splitext(file.filename or "")[1].lower()
    if media_type == "video":
        file_extension = ".mp4"
        input_ext = raw_ext if raw_ext else ".mp4"
    else:
        file_extension = raw_ext if raw_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
        input_ext = file_extension

    unique_id = uuid.uuid4()
    input_path = os.path.join(UPLOAD_DIR, f"orig_{unique_id}{input_ext}")
    output_filename = f"blured_{unique_id}{file_extension}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    box_data = None
    if box and box.strip() not in ["", "null", "undefined"]:
        try:
            box_data = json.loads(box)
        except Exception:
            box_data = None

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        blur_val = int(round(intensity))

        if media_type == "video":
            BlurProcessor.apply_video_blur(input_path, output_path, blur_val, box_data)
        else:
            BlurProcessor.apply_progressive_blur(input_path, output_path, blur_val, box_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

    return {"success": True, "download_url": f"/api/download/{output_filename}"}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        return FileResponse(
            file_path,
            filename=filename,
            media_type=mime_type or "application/octet-stream"
        )
    raise HTTPException(status_code=404, detail="Dosya bulunamadı!")
