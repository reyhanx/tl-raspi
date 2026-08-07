"""
object_classifier.py
Filter 3-lapis untuk membedakan KENDARAAN dari pejalan kaki/PKL/objek
non-kendaraan lain yang ikut kedeteksi MOG2 -- lihat diskusi lengkap
soal ini di HANDOFF.md.

Dirancang berlapis dari yang PALING MURAH ke yang PALING MAHAL, supaya
model TFLite (appearance_verifier) yang berat itu cuma dipanggil untuk
kasus yang beneran perlu:

  Lapis 0 (gratis)   -> filter geometris: rasio lebar/tinggi bbox.
                        Orang itu tinggi & sempit, kendaraan cenderung
                        lebih lebar. Kasus yang JELAS langsung diputus
                        di sini tanpa nyentuh model sama sekali.
  Lapis 1 (mahal,     -> appearance_verifier (TFLite), CUMA dipanggil
    rate-limited)        untuk objek yang: (a) baru pertama kali muncul,
                        (b) hasil Lapis 0 ambigu, (c) belum melebihi
                        rate limit (APPEARANCE_CROP_PADDING dkk di
                        config.py).
  Lapis 2 (memori)   -> begitu 1 objek sudah pernah diklasifikasi,
                        hasilnya DIINGAT selama objek itu masih di
                        posisi yang kurang lebih sama (dicek pakai IoU
                        antar bbox) -- gak perlu diklasifikasi ulang
                        tiap frame.

PENTING: threshold rasio di Lapis 0 (PERSON_MAX_ASPECT_RATIO,
VEHICLE_MIN_ASPECT_RATIO di config.py) itu NILAI AWAL/TEBAKAN, bukan
hasil pengukuran nyata -- wajib dikalibrasi ulang begitu kamera
terpasang fisik di posisi & sudut yang sebenarnya. Ini sama seperti
DETECTION_MIN_AREA yang juga butuh kalibrasi lapangan.
"""

import time

from logger_setup import get_logger

logger = get_logger("object_classifier")

VEHICLE = "vehicle"
NOT_VEHICLE = "not_vehicle"
AMBIGUOUS = "ambiguous"


def _iou(a, b):
    """Intersection-over-Union antara 2 bbox (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0

    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


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


class ObjectClassifier:
    def __init__(self, appearance_verifier,
                 person_max_aspect_ratio=0.6, vehicle_min_aspect_ratio=1.2,
                 model_min_interval=1.0, memory_timeout=5.0,
                 iou_match_threshold=0.3, crop_padding=20):
        self.verifier = appearance_verifier
        self.person_max_aspect_ratio = person_max_aspect_ratio
        self.vehicle_min_aspect_ratio = vehicle_min_aspect_ratio
        self.model_min_interval = model_min_interval
        self.memory_timeout = memory_timeout
        self.iou_match_threshold = iou_match_threshold
        self.crop_padding = crop_padding

        self._memory = []  # list of {"bbox", "classification", "last_seen"}
        self._last_model_call = 0.0

    def classify_frame(self, frame, bboxes):
        """
        frame  : frame yang SAMA persis dipakai buat hasilin bboxes ini
                 (harus koordinat yang sama -- pakai result["frame"] dari
                 vehicle_detection.py, BUKAN frame mentah sebelum resize).
        bboxes : list of (x, y, w, h) dari VehicleDetector.detect().
        Return : list bbox yang diklasifikasi sebagai KENDARAAN saja
                 (pejalan kaki/objek lain sudah disaring keluar).
        """
        now = time.time()
        vehicle_bboxes = []

        for bbox in bboxes:
            remembered = self._match_memory(bbox, now)
            if remembered is not None:
                if remembered == VEHICLE:
                    vehicle_bboxes.append(bbox)
                continue

            geom = self._geometric_filter(bbox)

            if geom in (VEHICLE, NOT_VEHICLE):
                self._remember(bbox, geom, now)
                if geom == VEHICLE:
                    vehicle_bboxes.append(bbox)
                continue

            # Ambigu -> eskalasi ke model, tapi rate-limited
            if self.verifier is not None and self.verifier.enabled and self._can_call_model(now):
                crop = _crop_with_padding(frame, bbox, self.crop_padding)
                is_vehicle = self.verifier.verify(crop) if crop is not None else False
                self._last_model_call = now
                classification = VEHICLE if is_vehicle else NOT_VEHICLE
                self._remember(bbox, classification, now)
                if is_vehicle:
                    vehicle_bboxes.append(bbox)
                logger.info(
                    f"Objek ambigu diverifikasi model: bbox={bbox} -> {classification}"
                )
            else:
                # Rate-limited atau verifier gak aktif -- BELUM bisa
                # dipastikan. Default konservatif: JANGAN dihitung dulu
                # di frame ini (lebih aman terlewat 1 kendaraan sesaat
                # daripada keliru ngitung pejalan kaki sebagai demand).
                # Kalau ini beneran kendaraan, kemungkinan besar akan
                # segera lolos Lapis 0 juga begitu posisinya makin
                # jelas/besar di frame berikutnya.
                continue

        self._cleanup_memory(now)
        return vehicle_bboxes

    def _geometric_filter(self, bbox):
        _, _, w, h = bbox
        if h == 0:
            return AMBIGUOUS
        aspect_ratio = w / h

        if aspect_ratio <= self.person_max_aspect_ratio:
            return NOT_VEHICLE
        if aspect_ratio >= self.vehicle_min_aspect_ratio:
            return VEHICLE
        return AMBIGUOUS

    def _match_memory(self, bbox, now):
        for entry in self._memory:
            if _iou(bbox, entry["bbox"]) >= self.iou_match_threshold:
                entry["bbox"] = bbox
                entry["last_seen"] = now
                return entry["classification"]
        return None

    def _remember(self, bbox, classification, now):
        self._memory.append({"bbox": bbox, "classification": classification, "last_seen": now})

    def _cleanup_memory(self, now):
        self._memory = [e for e in self._memory if now - e["last_seen"] < self.memory_timeout]

    def _can_call_model(self, now):
        return (now - self._last_model_call) >= self.model_min_interval
