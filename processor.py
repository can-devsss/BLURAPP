import cv2
import numpy as np

class BlurProcessor:
    @staticmethod
    def apply_progressive_blur(image_path: str, output_path: str, blur_strength: int, box_data: dict = None):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Görsel okunamadı!")

        if blur_strength <= 0:
            cv2.imwrite(output_path, img)
            return output_path

        # Önizleme ile birebir eşleşen güçlü kernel katsayısı
        ksize = int(blur_strength) * 3 + 1
        if ksize % 2 == 0:
            ksize += 1

        h, w = img.shape[:2]

        if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
            # Yüzdesel oranları gerçek piksele çeviriyoruz (Asla şaşmaz)
            x = int(box_data['rx'] * w)
            y = int(box_data['ry'] * h)
            bw = int(box_data['rw'] * w)
            bh = int(box_data['rh'] * h)

            x = max(0, min(x, w))
            y = max(0, min(y, h))
            bw = max(1, min(bw, w - x))
            bh = max(1, min(bh, h - y))

            roi = img[y:y+bh, x:x+bw]
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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ksize = int(blur_strength) * 3 + 1
        if ksize % 2 == 0:
            ksize += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if blur_strength > 0:
                if box_data and all(k in box_data for k in ['rx', 'ry', 'rw', 'rh']):
                    x = int(box_data['rx'] * width)
                    y = int(box_data['ry'] * height)
                    bw = int(box_data['rw'] * width)
                    bh = int(box_data['rh'] * height)

                    x = max(0, min(x, width))
                    y = max(0, min(y, height))
                    bw = max(1, min(bw, width - x))
                    bh = max(1, min(bh, height - y))

                    roi = frame[y:y+bh, x:x+bw]
                    blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
                    frame[y:y+bh, x:x+bw] = blurred_roi
                else:
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

            out.write(frame)

        cap.release()
        out.release()
        return output_path
