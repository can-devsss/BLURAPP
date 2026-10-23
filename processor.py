import cv2
import numpy as np

class BlurProcessor:
    @staticmethod
    def apply_progressive_blur(image_path: str, output_path: str, blur_strength: int):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı!")

        if blur_strength <= 0:
            cv2.imwrite(output_path, img)
            return output_path

        # Önizleme ve indirme motorunu matematiksel olarak yüzde yüz eşitleyen kernel hesabı
        ksize = int(blur_strength) * 2 + 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        cv2.imwrite(output_path, blurred)
        return output_path

    @staticmethod
    def apply_video_blur(video_path: str, output_path: str, blur_strength: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Video açılamadı!")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if blur_strength > 0:
                ksize = int(blur_strength) * 2 + 1
                frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            out.write(frame)

        cap.release()
        out.release()
        return output_path