import cv2
import numpy as np
import subprocess
import os
import shutil
from typing import Callable, Optional

class BlurProcessor:
    @staticmethod
    def _read_image_safe(image_path: str):
        with open(image_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image_safe(output_path: str, img):
        ext = os.path.splitext(output_path)[1].lower() or ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in [".jpg", ".jpeg"] else []
        success, encoded_img = cv2.imencode(ext, img, params)
        if success:
            with open(output_path, "wb") as f:
                encoded_img.tofile(f)

    @staticmethod
    def _compute_exact_sigma(blur_px: float, media_width: int, media_height: int, preview_w: float = 360.0) -> float:
        if blur_px <= 0:
            return 0.0
        ref_w = max(100.0, float(preview_w or 360.0))
        scale_factor = float(media_width) / ref_w
        sigma = float(blur_px) * scale_factor
        return max(0.1, sigma)

    @staticmethod
    def _apply_exact_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.05:
            return img
        h, w = img.shape[:2]
        if sigma > 12.0 and min(h, w) > 64:
            scale = max(0.15, min(1.0, 8.0 / sigma))
            small_w = max(2, int(w * scale))
            small_h = max(2, int(h * scale))
            small_sigma = sigma * scale
            small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            blurred_small = cv2.GaussianBlur(small_img, (0, 0), sigmaX=small_sigma, sigmaY=small_sigma)
            return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def apply_progressive_blur(
        image_path: str,
        output_path: str,
        blur_strength: float,
        box_data: dict = None,
        preview_w: float = 360.0,
        progress_cb: Optional[Callable[[int], None]] = None
    ):
        if progress_cb: progress_cb(15)
        img = BlurProcessor._read_image_safe(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı veya bozuk!")

        h, w = img.shape[:2]
        blur_val = float(blur_strength)

        if blur_val <= 0:
            BlurProcessor._write_image_safe(output_path, img)
            if progress_cb: progress_cb(100)
            return output_path

        sigma = BlurProcessor._compute_exact_sigma(blur_val, w, h, preview_w)
        if progress_cb: progress_cb(50)

        if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
            x = max(0, min(int(round(float(box_data['rx']) * w)), w - 1))
            y = max(0, min(int(round(float(box_data['ry']) * h)), h - 1))
            bw = max(2, min(int(round(float(box_data['rw']) * w)), w - x))
            bh = max(2, min(int(round(float(box_data['rh']) * h)), h - y))

            roi = img[y:y+bh, x:x+bw]
            if roi.size > 0:
                img[y:y+bh, x:x+bw] = BlurProcessor._apply_exact_gaussian(roi, sigma)
        else:
            img = BlurProcessor._apply_exact_gaussian(img, sigma)

        if progress_cb: progress_cb(85)
        BlurProcessor._write_image_safe(output_path, img)
        if progress_cb: progress_cb(100)
        return output_path

    @staticmethod
    def apply_video_blur(
        video_path: str,
        output_path: str,
        blur_strength: float,
        box_data: dict = None,
        preview_w: float = 360.0,
        progress_cb: Optional[Callable[[int], None]] = None
    ):
        if progress_cb: progress_cb(10)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video dosyası açılamadı!")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps) or fps > 240:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        blur_val = float(blur_strength)
        sigma = BlurProcessor._compute_exact_sigma(blur_val, width, height, preview_w)

        temp_silent_output = output_path + "_temp_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise RuntimeError("Video yazıcı (VideoWriter) başlatılamadı!")

        has_box = bool(box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']))
        if has_box:
            x = max(0, min(int(round(float(box_data['rx']) * width)), width - 1))
            y = max(0, min(int(round(float(box_data['ry']) * height)), height - 1))
            bw = max(2, min(int(round(float(box_data['rw']) * width)), width - x))
            bh = max(2, min(int(round(float(box_data['rh']) * height)), height - y))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if blur_val > 0:
                if has_box:
                    roi = frame[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        frame[y:y+bh, x:x+bw] = BlurProcessor._apply_exact_gaussian(roi, sigma)
                else:
                    frame = BlurProcessor._apply_exact_gaussian(frame, sigma)

            out.write(frame)
            frame_idx += 1

            if progress_cb and total_frames > 0 and frame_idx % 5 == 0:
                pct = min(80, int(10 + (frame_idx / total_frames) * 70))
                progress_cb(pct)

        cap.release()
        out.release()
        if progress_cb: progress_cb(85)

        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None

        if ffmpeg_exe and os.path.exists(temp_silent_output):
            try:
                ffmpeg_cmd = [
                    ffmpeg_exe, "-y",
                    "-i", temp_silent_output,
                    "-i", video_path,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "128k",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", output_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
            except Exception as e:
                print(f"FFmpeg uyarı: {e}")
                if os.path.exists(temp_silent_output):
                    os.replace(temp_silent_output, output_path)
            finally:
                if os.path.exists(temp_silent_output):
                    try: os.remove(temp_silent_output)
                    except: pass
        else:
            if os.path.exists(temp_silent_output):
                os.replace(temp_silent_output, output_path)

        if progress_cb: progress_cb(100)
        return output_path
