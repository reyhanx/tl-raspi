"""
test_hybrid.py
Script tes VISUAL khusus buat fitur "kenalin objek & kunci kendaraan
diam" (Tingkat 2 + filter pejalan kaki) -- lihat HANDOFF.md bagian
10, 13, 14.

Beda dari vehicle_detection.py (yang cuma tes MOG2 polos), script ini
nunjukin efek SETELAH object_classifier (buang pejalan kaki) dan
appearance_verifier (kunci kendaraan diam) diterapkan.

CARA BACA WARNA KOTAK & TEKS DI LAYAR:
- Kotak HIJAU + teks "LIVE"    -> MOG2 lagi lihat langsung, lolos filter
- Kotak KUNING + teks "TERKUNCI" -> MOG2 udah gak lihat gerakan lagi,
                                     tapi appearance verification masih
                                     yakin kendaraannya ada di situ
- Gak ada kotak sama sekali     -> dianggap kosong beneran (baik dari
                                     awal, atau setelah verifikasi bilang
                                     "udah gak ada")
- Angka "MOG2 mentah: X | Setelah filter: Y" di pojok atas -> kalau
  X > Y, artinya ada blob yang dibuang classifier (kemungkinan besar
  itu pejalan kaki/noise, bukan kendaraan)

KALAU ai-edge-litert / model BELUM diinstall:
Script ini TETAP JALAN, tapi cuma Lapis 0 (filter geometris) yang aktif
-- gak akan pernah muncul kotak kuning "TERKUNCI" karena appearance
verification-nya nonaktif. Itu normal, bukan error.
"""

import cv2
import time

from vehicle_detection import VehicleDetector
from appearance_verifier import AppearanceVerifier
from object_classifier import ObjectClassifier
from hybrid_detector import HybridVehicleDetector
import config
from logger_setup import get_logger

logger = get_logger("test_hybrid")

SOURCE_COLOR = {
    "live": (0, 255, 0),           # hijau
    "locked_grace": (0, 220, 255),  # kuning
    "locked_verified": (0, 220, 255),
}
SOURCE_LABEL = {
    "live": "LIVE",
    "locked_grace": "TERKUNCI (grace)",
    "locked_verified": "TERKUNCI (verified)",
}


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        logger.error("Kamera tidak terdeteksi.")
        return

    mog2 = VehicleDetector(
        min_area=config.DETECTION_MIN_AREA,
        resize_width=config.DETECTION_RESIZE_WIDTH,
        learning_rate=config.MOG2_LEARNING_RATE,
    )

    verifier = AppearanceVerifier(
        model_path=config.APPEARANCE_MODEL_PATH,
        labelmap_path=config.APPEARANCE_LABELMAP_PATH,
        confidence_threshold=config.APPEARANCE_CONFIDENCE_THRESHOLD,
    )
    if verifier.enabled:
        logger.info("Appearance model AKTIF -- fitur kunci & filter model penuh jalan.")
    else:
        logger.warning(
            "Appearance model NONAKTIF (belum diinstall/didownload) -- "
            "cuma filter geometris (Lapis 0) yang jalan. Kotak kuning "
            "'TERKUNCI' gak akan pernah muncul. Ini normal kalau kamu "
            "belum install ai-edge-litert + download model."
        )

    classifier = ObjectClassifier(
        appearance_verifier=verifier,
        person_max_aspect_ratio=config.PERSON_MAX_ASPECT_RATIO,
        vehicle_min_aspect_ratio=config.VEHICLE_MIN_ASPECT_RATIO,
        model_min_interval=config.CLASSIFIER_MODEL_MIN_INTERVAL,
        memory_timeout=config.CLASSIFIER_MEMORY_TIMEOUT,
        iou_match_threshold=config.CLASSIFIER_IOU_MATCH_THRESHOLD,
        crop_padding=config.APPEARANCE_CROP_PADDING,
    )

    hybrid = HybridVehicleDetector(
        mog2_detector=mog2,
        appearance_verifier=verifier,
        object_classifier=classifier,
        verify_interval=config.APPEARANCE_VERIFY_INTERVAL,
        crop_padding=config.APPEARANCE_CROP_PADDING,
    )

    logger.info("Tekan 'q' di window video untuk keluar.")
    prev_time = time.time()
    fps = 0.0
    DISPLAY_SCALE = 2.5  # perbesar cuma buat TAMPILAN, deteksi tetap di resolusi kecil (ringan)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        raw_mog2 = mog2.detect(frame)
        if not raw_mog2["ok"]:
            continue
        raw_count = raw_mog2["count"]

        result = hybrid.detect(frame)
        if not result.get("ok", False):
            continue

        now = time.time()
        instant_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        fps = fps * 0.9 + instant_fps * 0.1
        prev_time = now

        small_frame = result["frame"] if result.get("frame") is not None else frame
        # Perbesar frame DULU (biar teks yang digambar belakangan tetap
        # tajam, bukan teks kecil yang di-blur gara-gara di-scale up).
        display = cv2.resize(
            small_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST,
        )

        source = result.get("source", "empty")
        if result["count"] > 0 and result.get("bbox"):
            x, y, w, h = result["bbox"]
            # Skalakan koordinat bbox ikut upscale, biar kotaknya tetap
            # pas nempel ke objeknya di frame yang udah diperbesar.
            x, y, w, h = (int(v * DISPLAY_SCALE) for v in (x, y, w, h))
            color = SOURCE_COLOR.get(source, (0, 255, 0))
            label = SOURCE_LABEL.get(source, source)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Baris 1 (paling atas): perbandingan sebelum/sesudah filter
        cv2.putText(display, f"MOG2 mentah: {raw_count}  |  Setelah filter: {result['count']}",
                    (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Baris paling bawah: FPS
        cv2.putText(display, f"FPS: {fps:.1f}", (14, display.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Tes Hybrid Detector (q untuk keluar)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
