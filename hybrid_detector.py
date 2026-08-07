"""
hybrid_detector.py
Orkestrator deteksi lengkap: MOG2 -> filter pejalan kaki (object_classifier)
-> grace-period untuk kendaraan diam (appearance_verifier).

Lihat HANDOFF.md untuk diskusi lengkap kenapa arsitekturnya begini.

ALUR:
1. MOG2 (vehicle_detection.py) nemuin semua blob yang bergerak di frame.
2. object_classifier.py nyaring blob itu -- buang yang keklasifikasi
   sebagai "bukan kendaraan" (pejalan kaki, PKL, dll). Kalau classifier
   gak di-set (None), langkah ini di-skip, semua blob MOG2 dipercaya
   apa adanya (perilaku identik ke versi sebelum fitur ini ada).
3. Dari hasil yang udah disaring, kalau kosong di frame ini tapi ada
   "posisi terakhir diketahui" dari sebelumnya -> appearance_verifier
   dipanggil buat mastiin itu beneran udah pergi atau cuma lagi diem
   (masalah "kendaraan berhenti lama hilang dari deteksi").

Kalau appearance_verifier ATAU object_classifier tidak aktif (model
belum didownload / library belum terinstall), masing-masing fitur
otomatis nonaktif dan sistem tetap jalan normal -- ini semua lapisan
TAMBAHAN, bukan pengganti MOG2.
"""

import time

from logger_setup import get_logger

logger = get_logger("hybrid_detector")


def _crop_with_padding(frame, bbox, padding):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(w, x + bw + padding)
    y1 = min(h, y + bh + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


class HybridVehicleDetector:
    def __init__(self, mog2_detector, appearance_verifier, object_classifier=None,
                 verify_interval=2.0, crop_padding=25, fallback_density=0.02):
        self.mog2 = mog2_detector
        self.verifier = appearance_verifier
        self.classifier = object_classifier
        self.verify_interval = verify_interval
        self.crop_padding = crop_padding
        self.fallback_density = fallback_density

        self.last_bbox = None
        self.last_verified_time = 0.0

    def detect(self, frame):
        result = self.mog2.detect(frame)
        if not result["ok"]:
            return result

        # PENTING: dari titik ini, "frame" yang dipakai untuk crop HARUS
        # result["frame"] (sudah di-resize oleh VehicleDetector), BUKAN
        # parameter frame mentah di atas -- soalnya semua koordinat bbox
        # dihitung dalam ruang koordinat frame yang sudah di-resize itu.
        resized_frame = result["frame"]

        # ---- Lapis 0+1+2: saring pejalan kaki/non-kendaraan ----
        if self.classifier is not None and result["bboxes"]:
            vehicle_bboxes = self.classifier.classify_frame(resized_frame, result["bboxes"])
            result = self._recompute_from_bboxes(result, vehicle_bboxes)

        # ---- Grace-period untuk kendaraan yang berhenti lama ----
        if result["count"] > 0:
            self.last_bbox = result["bbox"]
            self.last_verified_time = 0.0
            return result

        if self.verifier is None or not self.verifier.enabled or self.last_bbox is None:
            return result

        now = time.time()
        if now - self.last_verified_time < self.verify_interval:
            return self._override_present(result)

        crop = _crop_with_padding(resized_frame, self.last_bbox, self.crop_padding)
        present = self.verifier.verify(crop) if crop is not None else False
        self.last_verified_time = now

        if present:
            return self._override_present(result)

        logger.info("Appearance verification: kendaraan di posisi terakhir sudah tidak ada.")
        self.last_bbox = None
        return result

    @staticmethod
    def _recompute_from_bboxes(result, vehicle_bboxes):
        """Hitung ulang count/density/bbox setelah bbox non-kendaraan
        dibuang oleh object_classifier."""
        new_result = dict(result)
        new_result["bboxes"] = vehicle_bboxes
        new_result["count"] = len(vehicle_bboxes)

        if not vehicle_bboxes:
            new_result["density"] = 0.0
            new_result["bbox"] = None
            return new_result

        frame = result["frame"]
        frame_area = frame.shape[0] * frame.shape[1]
        occupied = sum(w * h for (_, _, w, h) in vehicle_bboxes)
        new_result["density"] = round(min(occupied / frame_area, 1.0), 3) if frame_area else 0.0
        new_result["bbox"] = max(vehicle_bboxes, key=lambda b: b[2] * b[3])
        return new_result

    def _override_present(self, mog2_result):
        """MOG2 (setelah difilter) bilang kosong, tapi appearance
        verification (atau masih dalam grace period) bilang masih ada --
        override count/density supaya controller.update() tidak keliru
        menganggap cabang sudah sepi."""
        result = dict(mog2_result)
        result["count"] = max(result["count"], 1)
        result["density"] = max(result["density"], self.fallback_density)
        return result
