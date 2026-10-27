import cv2
import numpy as np
import subprocess
import os

class BlurProcessor:
    @staticmethod
    def apply_progressive_blur(image_path: str, output_path: str, blur_strength: int, box_data: dict = None):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı!")

        if blur_strength <= 0:
            cv2.imwrite(output_path, img)
            return output_path

        ksize = int(blur_strength) * 3 + 1
        if ksize % 2 == 0:
            ksize += 1

        h, w = img.shape[:2]

        if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
            x = int(box_data['rx'] * w)
            y = int(box_data['ry'] * h)
            bw = int(box_data['rw'] * w)
            bh = int(box_data['rh'] * h)

            x = max(0, min(x, w))
            y = max(0, min(y, h))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            roi = img[y:y+bh, x:x+bw]
            if roi.size > 0:
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                img[y:y+bh, x:x+bw] = blurred_roi
        else:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        cv2.imwrite(output_path, img)
        return output_path

    @staticmethod
    def apply_video_blur(video_path: str, output_path: str, blur_strength: int, box_data: dict = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı!")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        temp_silent_output = output_path + "_temp.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_silent_output, fourcc, fps, (width, height))

        ksize = int(blur_strength) * 3 + 1
        if ksize % 2 == 0:
            ksize += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if blur_strength > 0:
                if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
                    x = int(float(box_data['rx']) * width)
                    y = int(float(box_data['ry']) * height)
                    bw = int(float(box_data['rw']) * width)
                    bh = int(float(box_data['rh']) * height)

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    bw = max(2, min(bw, width - x))
                    bh = max(2, min(bh, height - y))

                    roi = frame[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                        frame[y:y+bh, x:x+bw] = blurred_roi
                else:
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            out.write(frame)

        cap.release()
        out.release()

        try:
            probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            has_audio = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

            if has_audio:
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", temp_silent_output, "-i", video_path,
                    "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", output_path
                ]
            else:
                ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_silent_output, "-c", "copy", output_path]

            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"FFmpeg ses birleştirme hatası: {e}")
            if os.path.exists(temp_silent_output):
                os.replace(temp_silent_output, output_path)
        finally:
            if os.path.exists(temp_silent_output):
                os.remove(temp_silent_output)

        return output_path
