from __future__ import annotations
import math
import os
import shutil
import subprocess
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
        """Görseli okur ve koordinatlar uygulanmadan önce EXIF yönlendirmesini düzeltir."""
        try:
            with Image.open(path) as source:
                oriented = ImageOps.exif_transpose(source)
                if oriented.mode not in ("RGB", "RGBA", "L"):
                    oriented = oriented.convert("RGBA" if "transparency" in oriented.info else "RGB")
                arr = np.asarray(oriented).copy()
                if arr.ndim == 2:
                    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                if arr.shape[2] == 4:
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image_safe(path: str, img):
        ext = Path(path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 96] if ext in {".jpg", ".jpeg"} else (
            [cv2.IMWRITE_WEBP_QUALITY, 95] if ext == ".webp" else []
        )
        ok, encoded = cv2.imencode(ext, img, params)
        if not ok:
            raise RuntimeError(f"Görsel {ext} formatında kodlanamadı.")
        try:
            encoded.tofile(path)
        except OSError as exc:
            raise RuntimeError(f"Çıktı dosyası yazılamadı: {exc}") from exc

    @staticmethod
    def _compute_sigma(blur_px: float, media_width: int, preview_w: float = 360.0) -> float:
        if blur_px <= 0:
            return 0.0
        if not np.isfinite(blur_px) or blur_px > 15:
            raise ValueError("Blur değeri 0–15 aralığında olmalı.")
        ref_w = max(100.0, min(5000.0, float(preview_w or 360.0)))
        return max(0.0, float(blur_px) * max(1, media_width) / ref_w)

    @staticmethod
    def _effect_image(img: np.ndarray, sigma: float, blur_style: str = "gaussian") -> np.ndarray:
        if sigma <= 0.05:
            return img.copy()
        h, w = img.shape[:2]

        if blur_style == "pixelate":
            # Mozaik / Sansür efekti
            block_size = max(4, int(round(sigma * 1.35)))
            sw = max(2, w // block_size)
            sh = max(2, h // block_size)
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        if blur_style == "box":
            k = max(3, int(round(sigma * 1.6)) | 1)
            if k > 99 and min(h, w) > 128:
                scale = 99.0 / k
                sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
                small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                small = cv2.blur(small, (99, 99))
                return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            return cv2.blur(img, (k, k))

        # Varsayılan: Kademeli Gaussian Blur
        if sigma > 40 and min(h, w) > 128:
            scale = min(1.0, max(0.025, 12.0 / sigma))
            sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
            small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(
                small, (0, 0), sigmaX=max(0.1, sigma * scale), sigmaY=max(0.1, sigma * scale)
            )
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def _apply_region(
        img: np.ndarray,
        sigma: float,
        box: dict | None,
        blur_style: str = "gaussian",
        invert: bool = False,
        feather: int = 0,
    ) -> np.ndarray:
        if sigma <= 0.05:
            return img.copy()
        h, w = img.shape[:2]
        blurred = BlurProcessor._effect_image(img, sigma, blur_style)
        if not box:
            return blurred

        x = max(0, min(int(round(box["rx"] * w)), w - 1))
        y = max(0, min(int(round(box["ry"] * h)), h - 1))
        x2 = max(x + 1, min(w, int(round((box["rx"] + box["rw"]) * w))))
        y2 = max(y + 1, min(h, int(round((box["ry"] + box["rh"]) * h))))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y2, x:x2] = 255
        if invert:
            mask = cv2.bitwise_not(mask)

        # Kenar yumuşatma (Feather) istenmişse ve mozaik değilse yumuşak geçiş uygula
        if feather > 0 and blur_style != "pixelate":
            fk = max(3, int(round((feather / 100.0) * min(w, h) * 0.12)) | 1)
            fk = min(fk, 151)
            mask_blur = cv2.GaussianBlur(mask, (fk, fk), 0)
            alpha = (mask_blur.astype(np.float32) / 255.0)[..., np.newaxis]
            blended = blurred.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha)
            return np.clip(blended, 0, 255).astype(np.uint8)

        result = img.copy()
        cv2.copyTo(blurred, mask, result)
        return result

    @staticmethod
    def apply_progressive_blur(
        image_path: str,
        output_path: str,
        blur_strength: float,
        box_data: dict | None = None,
        preview_w: float = 360.0,
        progress_cb: Progress | None = None,
        cancel_event=None,
        blur_style: str = "gaussian",
        invert: bool = False,
        feather: int = 0,
    ):
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb:
            progress_cb(10)
        img = BlurProcessor._read_image_safe(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı veya bozuk.")
        h, w = img.shape[:2]
        if w < 1 or h < 1:
            raise ValueError("Görsel boyutları geçersiz.")
        sigma = BlurProcessor._compute_sigma(float(blur_strength), w, preview_w)
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb:
            progress_cb(45)
        result = BlurProcessor._apply_region(img, sigma, box_data, blur_style, invert, feather)
        BlurProcessor._check_cancel(cancel_event)
        if progress_cb:
            progress_cb(85)
        BlurProcessor._write_image_safe(output_path, result)
        if progress_cb:
            progress_cb(100)
        return output_path

    @staticmethod
    def _find_ffmpeg():
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
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
        blur_style: str = "gaussian",
        invert: bool = False,
        feather: int = 0,
    ):
        """FFmpeg ile dikey/yatay rotasyon duyarlı ve chroma-safe hızlı video işleme."""
        BlurProcessor._check_cancel(cancel_event)
        ffmpeg = BlurProcessor._find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg bulunamadı. requirements.txt içindeki imageio-ffmpeg paketinin kurulu olduğundan emin olun."
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı veya desteklenmeyen codec kullanılıyor.")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            # Dikey telefon videolarında (rotate=90/270) genişlik-yükseklik takasını yakala
            orientation = 0
            if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
                orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
            if orientation in (90, 270, -90, -270):
                width, height = height, width
        finally:
            cap.release()

        if width < 2 or height < 2:
            raise ValueError("Video boyutları okunamadı.")
        if not math.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 30.0

        sigma = BlurProcessor._compute_sigma(float(blur_strength), width, preview_w)

        def _build_blur_filter(target_w: int, target_h: int) -> str:
            if sigma <= 0.05:
                return "null"
            if blur_style == "pixelate":
                factor = max(2, min(40, int(round(sigma * 1.2))))
                sw = max(2, (target_w // factor) // 2 * 2)
                sh = max(2, (target_h // factor) // 2 * 2)
                return f"scale={sw}:{sh}:flags=bilinear,scale={target_w}:{target_h}:flags=neighbor"
            # yuv420p chroma düzlemi çökmesini önlemek için maksimum güvenli yarıçap hesabı
            max_luma = max(1, min(target_w, target_h) // 2 - 1)
            max_chroma = max(1, min(target_w, target_h) // 4 - 1)
            luma_r = max(1, min(100, int(round(sigma)), max_luma))
            chroma_r = max(1, min(luma_r, max_chroma))
            power = 2 if blur_style == "gaussian" else 1
            return f"boxblur={luma_r}:{power}:{chroma_r}:{power}"

        if sigma <= 0.05:
            filter_graph = "[0:v:0]scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
        elif box_data:
            rx = max(0.0, min(1.0, float(box_data["rx"])))
            ry = max(0.0, min(1.0, float(box_data["ry"])))
            rw = max(0.0, min(1.0 - rx, float(box_data["rw"])))
            rh = max(0.0, min(1.0 - ry, float(box_data["rh"])))

            x = min(width - 4, int(rx * width) // 2 * 2)
            y = min(height - 4, int(ry * height) // 2 * 2)
            cw = min(width - x, max(4, int(rw * width) // 2 * 2))
            ch = min(height - y, max(4, int(rh * height) // 2 * 2))
            cw -= cw % 2
            ch -= ch % 2
            if cw < 4 or ch < 4:
                raise ValueError("Seçilen bulanıklaştırma alanı çok küçük.")

            if invert:
                # Portre modu: Arka planı blurla, seçili kutuyu net bırak
                b_flt = _build_blur_filter(width, height)
                filter_graph = (
                    f"[0:v:0]split=2[orig][bgsrc];"
                    f"[bgsrc]{b_flt}[bgblur];"
                    f"[orig]crop={cw}:{ch}:{x}:{y}[sharp];"
                    f"[bgblur][sharp]overlay={x}:{y}:shortest=1,"
                    f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
                )
            else:
                # Normal bölgesel mod: Sadece seçili kutuyu blurla
                b_flt = _build_blur_filter(cw, ch)
                filter_graph = (
                    f"[0:v:0]split=2[base][src];"
                    f"[src]crop={cw}:{ch}:{x}:{y},{b_flt}[blur];"
                    f"[base][blur]overlay={x}:{y}:shortest=1,"
                    f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
                )
        else:
            b_flt = _build_blur_filter(width, height)
            filter_graph = (
                f"[0:v:0]{b_flt},"
                f"scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p[v]"
            )

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-stats_period", "0.4", "-progress", "pipe:1", "-i", video_path,
            "-filter_complex", filter_graph,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
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

            return_code = process.wait(timeout=1800)
            stderr_text = process.stderr.read() if process.stderr else ""
            if return_code != 0:
                raise RuntimeError(
                    "FFmpeg video işleme hatası: " + (stderr_text[-1200:] or "Bilinmeyen hata")
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
