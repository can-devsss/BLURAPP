import cv2
import numpy as np
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

        ksize = int(blur_strength) * 2 + 1
        # Tek sayı garantisi
        if ksize % 2 == 0:
            ksize += 1

        # Eğer bölgesel (kısımlı) blur istenmişse
        if box_data and all(k in box_data for k in ['x', 'y', 'width', 'height']):
            h, w = img.shape[:2]
            # Oransal veya piksel koordinatları güvenli sınırlara çekelim
            x = max(0, int(box_data['x']))
            y = max(0, int(box_data['y']))
            bw = min(w - x, int(box_data['width']))
            bh = min(h - y, int(box_data['height']))

            if bw > 0 and bh > 0:
                roi = img[y:y+bh, x:x+bw]
                blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                img[y:y+bh, x:x+bw] = blurred_roi
        else:
            # Tam ekran blur
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
        
        # Tarayıcı uyumluluğu için h264 / mp4v codec denemesi
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ksize = int(blur_strength) * 2 + 1
        if ksize % 2 == 0:
            ksize += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if blur_strength > 0:
                if box_data and all(k in box_data for k in ['x', 'y', 'width', 'height']):
                    h, w = frame.shape[:2]
                    x = max(0, int(box_data['x']))
                    y = max(0, int(box_data['y']))
                    bw = min(w - x, int(box_data['width']))
                    bh = min(h - y, int(box_data['height']))
                    if bw > 0 and bh > 0:
                        roi = frame[y:y+bh, x:x+bw]
                        frame[y:y+bh, x:x+bw] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                else:
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            out.write(frame)

        cap.release()
        out.release()
        return output_path
