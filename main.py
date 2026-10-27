from __future__ import annotations
import os, time, uuid, json, threading, shutil, mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from processor import BlurProcessor

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp_storage"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_BLUR = 15.0
MAX_PREVIEW_WIDTH = 5000.0
JOB_TTL = 1800
MAX_WORKERS = max(1, min(4, os.cpu_count() or 2))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="blur")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.RLock()
ACTIVE_UPLOADS = 0
MAX_QUEUED_JOBS = 12
SEMAPHORE = threading.BoundedSemaphore(MAX_QUEUED_JOBS)

app = FastAPI(title="CAN STUDIO - Blur Engine")
# Aynı origin'de çalışan frontend için CORS gerekmez. Geliştirme için izin verilen origin'ler env ile eklenebilir.
allowed_origins = [x.strip() for x in os.getenv("BLUR_ALLOWED_ORIGINS", "").split(",") if x.strip()]
if allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

def cleanup_old_files():
    now = time.time()
    with JOBS_LOCK:
        active_paths = set()
        expired = []
        for jid, job in list(JOBS.items()):
            if job.get("status") in {"queued", "processing", "cancelling"}:
                active_paths.update(job.get("paths", []))
            elif now - job.get("created_at", now) > JOB_TTL:
                expired.append(jid)
        for jid in expired:
            JOBS.pop(jid, None)
    for p in UPLOAD_DIR.iterdir():
        try:
            if p.is_file() and str(p) not in active_paths and now - p.stat().st_mtime > JOB_TTL:
                p.unlink(missing_ok=True)
        except OSError:
            pass

def set_progress(job_id: str, pct: int):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job and not job["cancel_event"].is_set():
            job["progress"] = max(0, min(100, int(pct)))

def _run_job(job_id: str, input_path: str, output_path: str, output_filename: str,
             media_type: str, intensity: float, box_data: dict | None, preview_w: float):
    def progress(p): set_progress(job_id, p)
    try:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job: return
            if job["cancel_event"].is_set():
                job["status"] = "cancelled"; return
            job["status"] = "processing"; job["progress"] = 3
            cancel_event = job["cancel_event"]
        if media_type == "video":
            BlurProcessor.apply_video_blur(input_path, output_path, intensity, box_data, preview_w, progress, cancel_event)
        else:
            BlurProcessor.apply_progressive_blur(input_path, output_path, intensity, box_data, preview_w, progress, cancel_event)
        if cancel_event.is_set():
            raise InterruptedError("İşlem iptal edildi.")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Geçerli çıktı dosyası oluşturulamadı.")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status="completed", progress=100, download_url=f"/api/download/{output_filename}",
                           download_name=f"can_studio_blur{Path(output_filename).suffix}")
    except InterruptedError:
        with JOBS_LOCK:
            if job_id in JOBS: JOBS[job_id]["status"] = "cancelled"
        Path(output_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"[JOB_FAILED] job_id={job_id} media_type={media_type} error={type(exc).__name__}: {exc}")
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(status="failed", error=str(exc)[:500])
        Path(output_path).unlink(missing_ok=True)
    finally:
        Path(input_path).unlink(missing_ok=True)
        for suffix in ("_temp_silent.mp4", "_temp_audio.mp4"):
            Path(output_path + suffix).unlink(missing_ok=True)
        with JOBS_LOCK:
            if job_id in JOBS: JOBS[job_id]["paths"] = []
        SEMAPHORE.release()

@app.get("/", response_class=HTMLResponse)
def get_index():
    p = BASE_DIR / "index.html"
    if not p.is_file(): raise HTTPException(status_code=500, detail="index.html bulunamadı.")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {"ok": True, "workers": MAX_WORKERS}

@app.post("/api/process-media")
async def process_media(file: UploadFile = File(...), intensity: float = Form(...),
                        media_type: str = Form(...), box: str | None = Form(None),
                        preview_w: float = Form(360.0)):
    cleanup_old_files()
    if media_type not in {"image", "video"}: raise HTTPException(422, "Geçersiz medya türü.")
    if not (0 <= intensity <= MAX_BLUR): raise HTTPException(422, f"Blur değeri 0–{MAX_BLUR:g} aralığında olmalı.")
    if not (100 <= preview_w <= MAX_PREVIEW_WIDTH): raise HTTPException(422, "Önizleme genişliği geçersiz.")
    if not SEMAPHORE.acquire(blocking=False): raise HTTPException(429, "Sunucu meşgul. Biraz sonra tekrar deneyin.")
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "").suffix.lower()
    allowed_image = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    allowed_video = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
    if media_type == "image" and suffix not in allowed_image: suffix = ".jpg"
    if media_type == "video" and suffix not in allowed_video:
        await file.close()
        raise HTTPException(415, "Desteklenmeyen video uzantısı.")
    if media_type == "image" and suffix not in allowed_image:
        await file.close()
        raise HTTPException(415, "Desteklenmeyen görsel uzantısı.")
    try:
        box_data = None
        if box:
            try: box_data = json.loads(box)
            except (json.JSONDecodeError, TypeError): raise HTTPException(422, "Bölge seçimi geçersiz.")
            if not isinstance(box_data, dict) or not all(k in box_data for k in ("rx","ry","rw","rh")):
                raise HTTPException(422, "Bölge koordinatları eksik.")
            try: vals = [float(box_data[k]) for k in ("rx","ry","rw","rh")]
            except (TypeError, ValueError): raise HTTPException(422, "Bölge koordinatları sayısal olmalı.")
            rx, ry, rw, rh = vals
            if not all(0 <= v <= 1 for v in vals) or rw <= 0 or rh <= 0 or rx+rw > 1.00001 or ry+rh > 1.00001:
                raise HTTPException(422, "Bölge koordinatları sınır dışında.")
            box_data = dict(zip(("rx","ry","rw","rh"), vals))
        max_bytes = MAX_VIDEO_BYTES if media_type == "video" else MAX_IMAGE_BYTES
        input_path = UPLOAD_DIR / f"orig_{job_id}{suffix}"
        output_suffix = ".jpg" if media_type == "image" and suffix not in {".png",".webp",".bmp",".tif",".tiff"} else (suffix if media_type == "image" else ".mp4")
        output_filename = f"blurred_{job_id}{output_suffix}"
        output_path = UPLOAD_DIR / output_filename
        total = 0
        try:
            with input_path.open("wb") as dst:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk: break
                    total += len(chunk)
                    if total > max_bytes:
                        dst.close(); input_path.unlink(missing_ok=True)
                        raise HTTPException(413, f"Dosya boyutu sınırı aşıldı ({max_bytes // (1024*1024)} MB).")
                    dst.write(chunk)
        except Exception:
            input_path.unlink(missing_ok=True)
            raise
        if total == 0:
            input_path.unlink(missing_ok=True)
            raise HTTPException(400, "Boş dosya gönderildi.")
        job = {"status":"queued","progress":1,"created_at":time.time(),"download_url":None,
               "download_name":None,"error":None,"cancel_event":threading.Event(),
               "paths":[str(input_path),str(output_path)]}
        with JOBS_LOCK: JOBS[job_id] = job
        try:
            executor.submit(_run_job, job_id, str(input_path), str(output_path), output_filename,
                            media_type, float(intensity), box_data, float(preview_w))
        except Exception:
            with JOBS_LOCK: JOBS.pop(job_id, None)
            input_path.unlink(missing_ok=True)
            raise
        return {"success":True,"job_id":job_id}
    except HTTPException:
        SEMAPHORE.release(); raise
    except Exception as exc:
        SEMAPHORE.release()
        raise HTTPException(500, f"Dosya yüklenemedi: {str(exc)[:200]}")
    finally:
        await file.close()

@app.get("/api/job-status/{job_id}")
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            # Jobs live in process memory; a 404 here often means the service restarted
            # while a long video was processing. Log this so Render runtime logs can confirm.
            print(f"[JOB_STATUS_MISSING] job_id={job_id}; process restarted, instance changed, or job expired")
            raise HTTPException(404, "İşlem kaydı sunucuda bulunamadı. Sunucu yeniden başlamış olabilir; Render loglarını kontrol edin ve videoyu tekrar deneyin.")
        return {k: job.get(k) for k in ("status","progress","download_url","download_name","error")}

@app.post("/api/cancel-job/{job_id}")
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: raise HTTPException(404, "İşlem bulunamadı.")
        if job["status"] in {"completed","failed","cancelled"}: return {"success":True,"status":job["status"]}
        job["cancel_event"].set()
        job["status"] = "cancelling"
    return {"success":True,"status":"cancelling"}

@app.get("/api/download/{filename}")
def download_file(filename: str):
    # Only allow generated UUID-based output filenames.
    safe = Path(filename).name
    if safe != filename or not safe.startswith("blurred_") or any(x in safe for x in ("/","\\","..")):
        raise HTTPException(400, "Geçersiz dosya adı.")
    path = UPLOAD_DIR / safe
    if not path.is_file(): raise HTTPException(404, "Dosya bulunamadı veya süresi doldu.")
    return FileResponse(path=str(path), filename=safe, media_type="application/octet-stream",
                        headers={"Cache-Control":"no-store", "X-Content-Type-Options":"nosniff"})

@app.on_event("shutdown")
def shutdown_executor():
    executor.shutdown(wait=False, cancel_futures=True)

