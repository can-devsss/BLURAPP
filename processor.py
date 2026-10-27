from __future__ import annotations
import os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from PIL import Image, ImageOps

Progress = Callable[[int], None]

class BlurProcessor:
    @staticmethod
    def _check_cancel(cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("İşlem kullanıcı tarafından iptal edildi.")

    @staticmethod
    def _read_image_safe(path: str):
        """Read image and apply EXIF orientation before any blur coordinates are used."""
        try:
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                # Keep alpha when present; normalize palette and uncommon modes.
                if oriented.mode not in ("RGB", "RGBA", "L"):
                    oriented = oriented.convert("RGBA" if "transparency" in oriented.info else "RGB")
                arr = np.asarray(oriented).copy()
                if arr.ndim == 2:  # grayscale
                    return arr
                if arr.shape[2] == 4:  # RGBA -> BGRA for OpenCV
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            # Fallback for formats Pillow cannot decode; IMREAD_COLOR applies EXIF orientation
            # in OpenCV builds that support it (unlike IMREAD_UNCHANGED).
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image_safe(path: str, img):
        ext = Path(path).suffix.lower()
        if ext not in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}: ext = ".jpg"
        # OpenCV may not encode some formats in a given build; fail explicitly instead of silent success.
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in {".jpg",".jpeg"} else []
        ok, encoded = cv2.imencode(ext, img, params)
        if not ok: raise RuntimeError(f"Görsel {ext} formatında kodlanamadı.")
        try: encoded.tofile(path)
        except OSError as exc: raise RuntimeError(f"Çıktı dosyası yazılamadı: {exc}") from exc

    @staticmethod
    def _compute_sigma(blur_px: float, media_width: int, preview_w: float = 360.0) -> float:
        if blur_px <= 0: return 0.0
        if not np.isfinite(blur_px) or blur_px > 15: raise ValueError("Blur değeri 0–15 aralığında olmalı.")
        ref_w = max(100.0, min(5000.0, float(preview_w or 360.0)))
        return max(0.0, float(blur_px) * max(1, media_width) / ref_w)

    @staticmethod
    def _blur_image(img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.05: return img.copy()
        # OpenCV GaussianBlur avoids browser-dependent CSS/canvas filter differences in export.
        # Limit huge kernels by downsampling, but keep scale bounded and predictable.
        h, w = img.shape[:2]
        if sigma > 40 and min(h,w) > 128:
            scale = min(1.0, max(0.025, 12.0 / sigma))
            sw, sh = max(2, round(w*scale)), max(2, round(h*scale))
            small = cv2.resize(img, (sw,sh), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0,0), sigmaX=max(.1,sigma*scale), sigmaY=max(.1,sigma*scale))
            return cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(img, (0,0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def _apply_region(img: np.ndarray, sigma: float, box: dict | None) -> np.ndarray:
        if sigma <= 0.05: return img
        h,w=img.shape[:2]
        if not box: return BlurProcessor._blur_image(img, sigma)
        x=max(0,min(int(round(box["rx"]*w)),w-1)); y=max(0,min(int(round(box["ry"]*h)),h-1))
        x2=max(x+1,min(w,int(round((box["rx"]+box["rw"])*w))))
        y2=max(y+1,min(h,int(round((box["ry"]+box["rh"])*h))))
        # Blur the full frame to avoid ROI-edge artifacts, then copy only the selected area.
        # A binary mask avoids multiple full-frame float32 allocations, which can OOM on
        # 4K phone videos on small cloud instances.
        blurred=BlurProcessor._blur_image(img,sigma)
        mask=np.zeros((h,w),dtype=np.uint8)
        mask[y:y2,x:x2]=255
        result=img.copy()
        cv2.copyTo(blurred, mask, result)
        return result

    @staticmethod
    def apply_progressive_blur(image_path: str, output_path: str, blur_strength: float,
                               box_data: dict | None=None, preview_w: float=360.0,
                               progress_cb: Progress | None=None, cancel_event=None):
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb: progress_cb(10)
        img=BlurProcessor._read_image_safe(image_path)
        if img is None: raise ValueError("Görsel okunamadı veya bozuk.")
        h,w=img.shape[:2]
        if w<1 or h<1: raise ValueError("Görsel boyutları geçersiz.")
        sigma=BlurProcessor._compute_sigma(float(blur_strength),w,preview_w)
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb: progress_cb(45)
        result=BlurProcessor._apply_region(img,sigma,box_data)
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb: progress_cb(85)
        BlurProcessor._write_image_safe(output_path,result)
        if progress_cb: progress_cb(100)
        return output_path

    @staticmethod
    def _find_ffmpeg():
        exe=shutil.which("ffmpeg")
        if exe: return exe
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @staticmethod
    def apply_video_blur(
        video_path: str,
        output_path: str,
        blur_strength: float,
        box_data: dict | None = None,
        preview_w: float = 360.0,
        progress_cb: Progress | None = None,
        cancel_event=None,
    ):
        """Process video directly with FFmpeg, avoiding Python per-frame loops."""
        import math

        BlurProcessor._check_cancel(cancel_event)
        ffmpeg = BlurProcessor._find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg bulunamadı. requirements.txt içindeki imageio-ffmpeg "
                "paketinin kurulduğunu kontrol edin."
            )

        # Read metadata only; FFmpeg does the actual video filtering/encoding.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı veya desteklenmeyen codec kullanılıyor.")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()

        if width < 2 or height < 2:
            raise ValueError("Video boyutları okunamadı.")
        if not math.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 30.0

        sigma = BlurProcessor._compute_sigma(float(blur_strength), width, preview_w)
        # FFmpeg boxblur is efficient on CPU. Cap radius to avoid extreme costs.
        radius = max(1, min(100, int(round(sigma))))

        if box_data:
            rx = max(0.0, min(1.0, float(box_data["rx"])))
            ry = max(0.0, min(1.0, float(box_data["ry"])))
            rw = max(0.0, min(1.0 - rx, float(box_data["rw"])))
            rh = max(0.0, min(1.0 - ry, float(box_data["rh"])))

            # Align crop coordinates/sizes to even pixels for yuv420p output.
            x = min(width - 2, int(rx * width) // 2 * 2)
            y = min(height - 2, int(ry * height) // 2 * 2)
            cw = min(width - x, max(2, int(rw * width) // 2 * 2))
            ch = min(height - y, max(2, int(rh * height) // 2 * 2))
            cw -= cw % 2
            ch -= ch % 2
            if cw < 2 or ch < 2:
                raise ValueError("Seçilen bulanıklaştırma alanı çok küçük.")

            filter_graph = (
                f"[0:v:0]split=2[base][src];"
                f"[src]crop={cw}:{ch}:{x}:{y},boxblur={radius}:2[blur];"
                f"[base][blur]overlay={x}:{y}:shortest=1,"
                f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
            )
        else:
            filter_graph = (
                f"[0:v:0]boxblur={radius}:2,"
                f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
            )

        # -progress reports machine-readable progress on stdout.
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-stats_period", "0.5", "-progress", "pipe:1", "-i", video_path,
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-threads", "0", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", output_path,
        ]

        process = None
        try:
            BlurProcessor._check_cancel(cancel_event)
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            last_progress = 8
            if progress_cb:
                progress_cb(last_progress)

            assert process.stdout is not None
            for line in process.stdout:
                BlurProcessor._check_cancel(cancel_event)
                line = line.strip()
                if line.startswith("frame=") and total_frames > 0:
                    try:
                        frame = int(line.split("=", 1)[1])
                        percent = max(8, min(95, int(8 + 87 * frame / total_frames)))
                        if percent > last_progress:
                            last_progress = percent
                            if progress_cb:
                                progress_cb(percent)
                    except ValueError:
                        pass

            # FFmpeg normally writes only errors to stderr due to -loglevel error.
            return_code = process.wait(timeout=1800)
            stderr_text = process.stderr.read() if process.stderr else ""
            if return_code != 0:
                raise RuntimeError(
                    "FFmpeg video işleme hatası: "
                    + (stderr_text[-1200:] or "Bilinmeyen hata")
                )
            if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("FFmpeg geçerli çıktı oluşturamadı.")
            if progress_cb:
                progress_cb(100)
            return output_path

        except InterruptedError:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            Path(output_path).unlink(missing_ok=True)
            raise
        except Exception:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            Path(output_path).unlink(missing_ok=True)
            raise
        finally:
            if process:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            if cancel_event is not None and cancel_event.is_set():
                Path(output_path).unlink(missing_ok=True)
