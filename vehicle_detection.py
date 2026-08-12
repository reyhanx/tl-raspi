"""
vehicle_detection.py
Deteksi kendaraan menggunakan background subtraction (MOG2).
Dipilih karena ringan dan cukup akurat untuk Raspberry Pi 3
dengan kamera statis (fixed position).
"""

import cv2
import time
import os

from logger_setup import get_logger

logger = get_logger("vehicle_detection")


class VehicleDetector:
    def __init__(self, min_area=800, resize_width=320, learning_rate=-1):
        self.min_area = min_area
        self.resize_width = resize_width
        self.learning_rate = learning_rate
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w == self.resize_width:
            return frame
        ratio = self.resize_width / float(w)
        new_h = max(1, int(h * ratio))
        return cv2.resize(frame, (self.resize_width, new_h))

    def detect(self, frame):
        empty_result = {
            "count": 0, "density": 0.0, "bbox": None, "bboxes": [],
            "frame": None, "timestamp": time.time(), "ok": False,
        }

        if frame is None or frame.size == 0:
            logger.warning("Frame kosong/invalid diterima, dilewati.")
            return empty_result

        try:
            frame = self._resize(frame)
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            fg_mask = self.back_sub.apply(blurred, learningRate=self.learning_rate)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            frame_area = frame.shape[0] * frame.shape[1]
            occupied_area = 0
            largest_area = 0
            largest_bbox = None
            all_bboxes = []

            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                occupied_area += area
                x, y, w, h = cv2.boundingRect(c)
                all_bboxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if area > largest_area:
                    largest_area = area
                    largest_bbox = (x, y, w, h)

            density = min(occupied_area / frame_area, 1.0) if frame_area else 0

            return {
                "count": len(all_bboxes),
                "density": round(density, 3),
                "bbox": largest_bbox,
                "bboxes": all_bboxes,
                "frame": frame,
                "timestamp": time.time(),
                "ok": True,
            }

        except cv2.error:
            logger.exception("OpenCV error saat memproses frame, dilewati.")
            return empty_result
        except Exception:
            logger.exception("Error tak terduga saat deteksi, dilewati.")
            return empty_result


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    detector = VehicleDetector()

    if not cap.isOpened():
        logger.error("Kamera tidak terdeteksi. Cek koneksi kamera.")
        exit(1)

    logger.info("Tekan 'q' di window video untuk keluar.")

    prev_time = time.time()
    fps = 0.0
    DISPLAY_SCALE = 2.5

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = detector.detect(frame)
        if not result["ok"]:
            continue

        now = time.time()
        instant_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        fps = fps * 0.9 + instant_fps * 0.1
        prev_time = now

        small_frame = result["frame"]
        display_frame = cv2.resize(
            small_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST,
        )
        h_disp, w_disp = display_frame.shape[:2]
        BAR_HEIGHT = 40

        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_disp, BAR_HEIGHT), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, h_disp - BAR_HEIGHT), (w_disp, h_disp), (15, 15, 15), -1)
        display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)

        cv2.putText(display_frame, f"Kendaraan: {result['count']}", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (14, h_disp - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Deteksi Kendaraan (q untuk keluar)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
PYEOF
echo "Selesai ditulis ulang"