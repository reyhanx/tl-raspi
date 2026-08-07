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
        """
        learning_rate: seberapa cepat MOG2 "melupakan" objek diam dan
        menelannya ke background. -1 = otomatis (default OpenCV, cepat
        melupakan). Nilai kecil positif (misal 0.001) bikin objek diam
        butuh JAUH lebih lama sebelum mulai luntur ke background --
        mitigasi untuk masalah "kendaraan berhenti lama hilang dari
        deteksi" yang didiskusikan (lihat HANDOFF.md bagian 10).
        """
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
        """
        Input  : 1 frame BGR dari kamera
        Output : dict berisi jumlah kendaraan terdeteksi, kepadatan (0-1),
                 bbox objek TERBESAR (x, y, w, h) atau None kalau gak ada,
                 daftar SEMUA bbox yang lolos MIN_AREA (dipakai object_classifier.py
                 untuk filter pejalan kaki), dan frame yang sudah digambar
                 kotak deteksi.
        Kalau frame rusak/kosong atau proses OpenCV gagal, tidak melempar
        exception ke pemanggil — kembalikan hasil kosong yang aman supaya
        loop utama tetap berjalan.
        """
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

            cv2.putText(
                frame, f"Kendaraan: {len(all_bboxes)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

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
    # CATATAN: mode headless (TANPA cv2.imshow). Kalau kamu jalanin ini
    # lewat Raspberry Pi Connect / remote screen sharing, cv2.imshow()
    # bikin video di-render 2x (kamera -> window lokal -> di-stream
    # ulang lewat remote desktop ke layar kamu) -- itu yang bikin
    # patah-patah parah, BUKAN performa deteksinya. main.py (produksi)
    # juga headless, jadi masalah ini gak akan muncul di sistem asli.
    #
    # Tes ini cuma print FPS + jumlah kendaraan ke terminal, dan nyimpen
    # 1 frame contoh ke file tiap beberapa detik biar bisa dicek visual
    # (download filenya lewat SFTP/file manager, bukan lewat GUI).

    cap = cv2.VideoCapture(0)

    # Set resolusi capture dari AWAL (bukan resize belakangan) --
    # capture native resolution tinggi lalu di-resize itu buang-buang
    # CPU buat decode+convert frame besar yang gak kepake.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    detector = VehicleDetector()

    if not cap.isOpened():
        logger.error("Kamera tidak terdeteksi. Cek koneksi kamera.")
        exit(1)

    logger.info("Tes headless berjalan 20 detik. Tekan Ctrl+C untuk berhenti lebih awal.")

    frame_count = 0
    start_time = time.time()
    last_save = 0
    test_duration = 20  # detik

    try:
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            result = detector.detect(frame)
            if not result["ok"]:
                continue

            frame_count += 1
            now = time.time()

            # Simpan 1 contoh frame tiap 3 detik buat dicek visual nanti
            if now - last_save >= 3:
                preview_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_preview.jpg")
                os.makedirs(os.path.dirname(preview_path), exist_ok=True)
                cv2.imwrite(preview_path, result["frame"])
                last_save = now
                logger.info(
                    f"Kendaraan: {result['count']} | Kepadatan: {result['density']} "
                    f"| (contoh frame disimpan ke data/test_preview.jpg)"
                )
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"Selesai. {frame_count} frame diproses dalam {elapsed:.1f}s -> {fps:.1f} FPS rata-rata.")
        cap.release()

