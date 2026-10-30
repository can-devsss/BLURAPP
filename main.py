from __future__ import annotations
import os, time, uuid, json, threading
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
MAX_QUEUED_JOBS = 12
SEMAPHORE = threading.BoundedSemaphore(MAX_QUEUED_JOBS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_old_files()
    yield
    executor.shutdown(wait=False, cancel_futures=True)

app = FastAPI(title="CAN STUDIO - Pro Blur Engine", lifespan=lifespan)

@app.api_route("/healthz", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def health_check():
    return {"status": "ok", "workers": MAX_WORKERS}

allowed_origins = [x.strip() for x in os.getenv("BLUR_ALLOWED_ORIGINS", "").split(",") if x.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

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

def _run_job(
    job_id: str,
    input_path: str,
    output_path: str,
    output_filename: str,
    media_type: str,
    intensity: float,
    boxes_data: list[dict] | None,
    blur_mode: str,
    effect_type: str,
    preview_w: float,
):
    def progress(p):
        set_progress(job_id, p)

    try:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            if job["cancel_event"].is_set():
                job["status"] = "cancelled"
                return
            job["status"] = "processing"
            job["progress"] = 5
            cancel_event = job["cancel_event"]

        if media_type == "video":
            BlurProcessor.apply_video_blur(
                input_path, output_path, intensity, boxes_data,
                blur_mode, effect_type, preview_w, progress, cancel_event
            )
        else:
            BlurProcessor.apply_progressive_blur(
                input_path, output_path, intensity, boxes_data,
                blur_mode, effect_type, preview_w, progress, cancel_event
            )

        if cancel_event.is_set():
            raise InterruptedError("İşlem iptal edildi.")
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Geçerli çıktı dosyası oluşturulamadı.")

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(
                    status="completed",
                    progress=100,
                    download_url=f"/api/download/{output_filename}",
                    download_name=f"can_studio_blur{Path(output_filename).suffix}",
                )
    except InterruptedError:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "cancelled"
        Path(output_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"[JOB_FAILED] job_id={job_id} media_type={media_type} error={type(exc).__name__}: {exc}")
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id].update(status="failed", error=str(exc)[:500])
        Path(output_path).unlink(missing_ok=True)
    finally:
        Path(input_path).unlink(missing_ok=True)
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["paths"] = []
        SEMAPHORE.release()

def _parse_boxes(boxes_raw: str | None, box_legacy: str | None) -> list[dict] | None:
    raw = boxes_raw or box_legacy
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(422, "Bölge seçimi JSON formatı geçersiz.")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or len(parsed) == 0 or len(parsed) > 25:
        raise HTTPException(422, "En fazla 25 bölge seçilebilir.")
    validated = []
    for item in parsed:
        if not isinstance(item, dict) or not all(k in item for k in ("rx", "ry", "rw", "rh")):
            raise HTTPException(422, "Bölge koordinatları eksik.")
        try:
            rx, ry, rw, rh = (float(item[k]) for k in ("rx", "ry", "rw", "rh"))
        except (TypeError, ValueError):
            raise HTTPException(422, "Bölge koordinatları sayısal olmalı.")
        if not all(0.0 <= v <= 1.0 for v in (rx, ry, rw, rh)) or rw <= 0 or rh <= 0:
            raise HTTPException(422, "Bölge koordinatları sınır dışında.")
        rw = min(rw, 1.0 - rx)
        rh = min(rh, 1.0 - ry)
        validated.append({"rx": rx, "ry": ry, "rw": rw, "rh": rh})
    return validated

@app.get("/", response_class=HTMLResponse)
def get_index():
    p = BASE_DIR / "index.html"
    if not p.is_file():
        raise HTTPException(status_code=500, detail="index.html bulunamadı.")
    return HTMLResponse(p.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {"ok": True, "workers": MAX_WORKERS}

@app.post("/api/process-media")
async def process_media(
    file: UploadFile = File(...),
    intensity: float = Form(...),
    media_type: str = Form(...),
    blur_mode: str = Form("full"),
    effect_type: str = Form("gaussian"),
    boxes: str | None = Form(None),
    box: str | None = Form(None),
    preview_w: float = Form(360.0),
):
    cleanup_old_files()
    if media_type not in {"image", "video"}:
        raise HTTPException(422, "Geçersiz medya türü.")
    if blur_mode not in {"full", "partial", "invert"}:
        raise HTTPException(422, "Geçersiz bulanıklaştırma modu.")
    if effect_type not in {"gaussian", "pixelate"}:
        raise HTTPException(422, "Geçersiz efekt türü.")
    if not (0 <= intensity <= MAX_BLUR):
        raise HTTPException(422, f"Blur değeri 0–{MAX_BLUR:g} aralığında olmalı.")
    if not (100 <= preview_w <= MAX_PREVIEW_WIDTH):
        raise HTTPException(422, "Önizleme genişliği geçersiz.")
    if not SEMAPHORE.acquire(blocking=False):
        raise HTTPException(429, "Sunucu meşgul. Biraz sonra tekrar deneyin.")

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "").suffix.lower()
    allowed_image = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    allowed_video = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}

    try:
        if media_type == "video" and suffix not in allowed_video:
            raise HTTPException(415, "Desteklenmeyen video uzantısı.")
        if media_type == "image" and suffix not in allowed_image:
            suffix = ".jpg"

        boxes_data = _parse_boxes(boxes, box) if blur_mode != "full" else None
        if blur_mode in {"partial", "invert"} and not boxes_data:
            raise HTTPException(422, "Bölgesel mod için en az bir alan seçilmelidir.")

        max_bytes = MAX_VIDEO_BYTES if media_type == "video" else MAX_IMAGE_BYTES
        input_path = UPLOAD_DIR / f"orig_{job_id}{suffix}"
        output_suffix = (
            ".jpg"
            if media_type == "image" and suffix not in {".png", ".webp", ".bmp", ".tif", ".tiff"}
            else (suffix if media_type == "image" else ".mp4")
        )
        output_filename = f"blurred_{job_id}{output_suffix}"
        output_path = UPLOAD_DIR / output_filename

        total = 0
        try:
            with input_path.open("wb") as dst:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        dst.close()
                        input_path.unlink(missing_ok=True)
                        raise HTTPException(413, f"Dosya boyutu sınırı aşıldı ({max_bytes // (1024*1024)} MB).")
                    dst.write(chunk)
        except Exception:
            input_path.unlink(missing_ok=True)
            raise

        if total == 0:
            input_path.unlink(missing_ok=True)
            raise HTTPException(400, "Boş dosya gönderildi.")

        job = {
            "status": "queued",
            "progress": 1,
            "created_at": time.time(),
            "download_url": None,
            "download_name": None,
            "error": None,
            "cancel_event": threading.Event(),
            "paths": [str(input_path), str(output_path)],
        }
        with JOBS_LOCK:
            JOBS[job_id] = job

        executor.submit(
            _run_job, job_id, str(input_path), str(output_path), output_filename,
            media_type, float(intensity), boxes_data, blur_mode, effect_type, float(preview_w)
        )
        return {"success": True, "job_id": job_id}
    except HTTPException:
        SEMAPHORE.release()
        raise
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
            raise HTTPException(404, "İşlem kaydı sunucuda bulunamadı.")
        return {k: job.
