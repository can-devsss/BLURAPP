import cv2
import numpy as np
import subprocess
import os
import shutil

class BlurProcessor:
    @staticmethod
    def _read_image_safe(image_path: str):
        """Windows Türkçe karakterli dosya yollarında cv2.imread çökmesini önler."""
        with open(image_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image_safe(output_path: str, img):
        """Windows Türkçe karakterli dosya yollarında cv2.imwrite çökmesini önler."""
        ext = os.path.splitext(output_path)[1] or ".jpg"
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(output_path, "wb") as f:
                encoded_img.tofile(f)

    @staticmethod
    def _calculate_kernel(blur_strength: float, width: int, height: int) -> int:
        """Görsel/Video çözünürlüğüne göre gerçekçi ve güçlü bir tek sayı kernel hesaplar."""
        scale = max(1.0, min(width, height) / 500.0)
        ksize = int(float(blur_strength) * 2.5 * scale)
        if ksize % 2 == 0:
            ksize += 1
        return max(3, ksize)

    @staticmethod
    def apply_progressive_blur(image_path: str, output_path: str, blur_strength: int, box_data: dict = None):
        img = BlurProcessor._read_image_safe(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı!")

        if float(blur_strength) <= 0:
            BlurProcessor._write_image_safe(output_path, img)
            return output_path

        h, w = img.shape[:2]
        ksize = BlurProcessor._calculate_kernel(blur_strength, w, h)

        if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
            x = int(float(box_data['rx']) * w)
            y = int(float(box_data['ry']) * h)
            bw = int(float(box_data['rw']) * w)
            bh = int(float(box_data['rh']) * h)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            roi = img[y:y+bh, x:x+bw]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                img[y:y+bh, x:x+bw] = blurred_roi
        else:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        BlurProcessor._write_image_safe(output_path, img)
        return output_path

    @staticmethod
    def apply_video_blur(video_path: str, output_path: str, blur_strength: int, box_data: dict = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı!")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0
        
        temp_silent_output = output_path + "_temp.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))

        ksize = BlurProcessor._calculate_kernel(blur_strength, width, height)

        has_box = box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh'])
        if has_box:
            x = max(0, min(int(float(box_data['rx']) * width), width - 1))
            y = max(0, min(int(float(box_data['ry']) * height), height - 1))
            bw = max(2, min(int(float(box_data['rw']) * width), width - x))
            bh = max(2, min(int(float(box_data['rh']) * height), height - y))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if float(blur_strength) > 0:
                if has_box:
                    roi = frame[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                        frame[y:y+bh, x:x+bw] = blurred_roi
                else:
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            out.write(frame)

        cap.release()
        out.release()

        # FFmpeg ile videoyu her cihazda/tarayıcıda açılan H.264 formatına çevir ve orijinal sesi ekle
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = None

        if ffmpeg_exe and os.path.exists(temp_silent_output):
            try:
                ffmpeg_cmd = [
                    ffmpeg_exe, "-y",
                    "-i", temp_silent_output,
                    "-i", video_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", output_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"FFmpeg dönüştürme uyarısı: {e}")
                if os.path.exists(temp_silent_output):
                    os.replace(temp_silent_output, output_path)
            finally:
                if os.path.exists(temp_silent_output):
                    os.remove(temp_silent_output)
        else:
            if os.path.exists(temp_silent_output):
                os.replace(temp_silent_output, output_path)

        return output_path
