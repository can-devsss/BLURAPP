from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import uuid
import json
import time
import threading

try:
    from backend.processor import BlurProcessor
except ImportError:
    from processor import BlurProcessor

app = FastAPI(title="CAN STUDIO - Blur Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 20+ kullanıcının aynı anda siteyi çökertmeden işlem yapmasını sağlayan Thread Havuzu ve Görev Yöneticisi
MAX_WORKERS = min(8, (os.cpu_count() or 2) + 2)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
JOBS = {}
JOBS_LOCK = threading.Lock()

def cleanup_old_files():
    """Disk dolup sunucunun çökmesini önlemek için 30 dakikadan eski geçici dosyaları temizler."""
    now = time.time()
    try:
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 1800:
                try:
                    os.remove(fpath)
                except Exception:
                    pass
    except Exception:
        pass

    with JOBS_LOCK:
        expired = [jid for jid, j in JOBS.items() if (now - j.get("created_at", now)) > 1800]
        for jid in expired:
            JOBS.pop(jid, None)

def _run_blur_job(job_id: str, input_path: str, output_path: str, output_filename: str, download_name: str, media_type: str, intensity: float, box_data: dict, preview_w: float):
    def update_progress(pct: int):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = pct

    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["progress"] = 5

        if media_type == "video":
            BlurProcessor.apply_video_blur(input_path, output_path, intensity, box_data, preview_w, update_progress)
        else:
            BlurProcessor.apply_progressive_blur(input_path, output_path, intensity, box_data, preview_w, update_progress)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Çıktı dosyası oluşturulamadı!")

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["download_url"] = f"/api/download/{output_filename}?name={download_name}"
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass

@app.get("/", response_class=HTMLResponse)
def get_index():
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
def process_media(
    file: UploadFile = File(...),
    intensity: float = Form(...),
    media_type: str = Form(...),
    box: str = Form(None),
    preview_w: float = Form(360.0)
):
    cleanup_old_files()

    orig_name = file.filename or ("video.mp4" if media_type == "video" else "image.jpg")
    raw_ext = os.path.splitext(orig_name)[1].lower()

    if media_type == "video":
        file_extension = ".mp4"
        input_ext = raw_ext if raw_ext else ".mp4"
    else:
        file_extension = raw_ext if raw_ext in [".jpg", ".jpeg", ".png", ".webp"] else ".jpg"
        input_ext = file_extension

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"orig_{job_id}{input_ext}")
    output_filename = f"blured_{job_id}{file_extension}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    safe_download_name = f"can_studio_blur{file_extension}"

    box_data = None
    if box and box.strip() not in ["", "null", "undefined"]:
        try:
            box_data = json.loads(box)
        except Exception:
            box_data = None

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya yüklenemedi: {e}")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 2,
            "created_at": time.time(),
            "download_url": None,
            "error": None
        }

    # İşlemi arka plan havuzuna gönder (Sunucu anında yanıt döner, asla kilitlenmez)
    executor.submit(
        _run_blur_job,
        job_id, input_path, output_path, output_filename,
        safe_download_name, media_type, float(intensity), box_data, float(preview_w)
    )

    return {"success": True, "job_id": job_id}

@app.get("/api/job-status/{job_id}")
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="İşlem bulunamadı veya süresi doldu.")
        return dict(job)

@app.get("/api/download/{filename}")
def download_file(filename: str, name: str = None):
    # Path traversal güvenlik kontrolü
    safe_fname = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_fname)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı!")

    dl_name = name or safe_fname
    return FileResponse(
        path=file_path,
        filename=dl_name,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{dl_name}"',
            "Cache-Control": "no-store"
        }
    )
