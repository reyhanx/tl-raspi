"""
main.py
Orkestrator utama untuk PERTIGAAN dengan logika actuated:
- Kamera terus memantau jalur CABANG (Jakenan)
- Jalur UTAMA (Pantura) default hijau, hanya diinterupsi kalau ada
  demand terkonfirmasi dari cabang
- Tiap pembacaan kamera baru langsung di-feed ke controller.update(),
  yang menentukan sendiri kapan harus ganti fase

Dirancang untuk jalan lama tanpa pengawasan (unattended):
- Kamera yang putus koneksi akan dicoba dibuka ulang otomatis
- Frame gagal dibaca tidak menghentikan program, hanya dilewati
- Frame image ditulis secara atomic (tulis ke .tmp lalu rename) agar
  dashboard tidak pernah membaca file yang setengah tertulis
- Status disimpan ke database secara periodik (bukan tiap frame) untuk
  dashboard, dan baris log lama otomatis dipangkas
- Error di satu iterasi loop tidak menjatuhkan seluruh program
- PROCESS_EVERY_N_FRAMES melewati sebagian frame supaya CPU Pi 3 tidak
  terus-terusan penuh terpakai oleh background subtraction
"""

import cv2
import sqlite3
import time
import os
from datetime import datetime

import config
from logger_setup import get_logger
from vehicle_detection import VehicleDetector
from appearance_verifier import AppearanceVerifier
from object_classifier import ObjectClassifier
from hybrid_detector import HybridVehicleDetector
from traffic_controller import ActuatedIntersectionController

logger = get_logger("main")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "traffic_log.db")
FRAME_PATH = os.path.join(DATA_DIR, "latest_frame.jpg")
FRAME_TMP_PATH = os.path.join(DATA_DIR, "latest_frame_tmp.jpg")


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            vehicle_count INTEGER,
            density REAL,
            state TEXT,
            phase_elapsed INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_log(vehicle_count, density, state, phase_elapsed):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.execute(
            "INSERT INTO logs (timestamp, vehicle_count, density, state, phase_elapsed) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), vehicle_count, density, state, phase_elapsed),
        )
        conn.commit()
    except sqlite3.Error:
        logger.exception("Gagal menyimpan log ke database, dilewati.")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def prune_old_logs():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        if count > config.MAX_LOG_ROWS:
            excess = count - config.MAX_LOG_ROWS
            conn.execute(
                "DELETE FROM logs WHERE id IN (SELECT id FROM logs ORDER BY id ASC LIMIT ?)",
                (excess,),
            )
            conn.commit()
            logger.info(f"Memangkas {excess} baris log lama.")
    except sqlite3.Error:
        logger.exception("Gagal memangkas log lama, dilewati.")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def save_frame_atomic(frame):
    try:
        ok = cv2.imwrite(FRAME_TMP_PATH, frame)
        if ok:
            os.replace(FRAME_TMP_PATH, FRAME_PATH)
    except Exception:
        logger.exception("Gagal menyimpan preview frame, dilewati.")


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    # Set resolusi capture dari AWAL, bukan resize belakangan -- capture
    # native resolution tinggi lalu di-resize itu buang-buang CPU buat
    # decode+convert frame besar yang gak kepake (penting di Pi 3).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.DETECTION_RESIZE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.DETECTION_RESIZE_WIDTH * 0.75))
    if cap.isOpened():
        logger.info("Kamera berhasil dibuka.")
    return cap


def main():
    init_db()

    cap = open_camera()
    mog2_detector = VehicleDetector(
        min_area=config.DETECTION_MIN_AREA,
        resize_width=config.DETECTION_RESIZE_WIDTH,
        learning_rate=config.MOG2_LEARNING_RATE,
    )

    # Lapisan ke-2 (opsional): kalau model/library belum tersedia, ini
    # otomatis nonaktif dan detector di bawah berperilaku sama persis
    # kayak MOG2 polos -- lihat appearance_verifier.py.
    models_dir = os.path.dirname(__file__)
    verifier = AppearanceVerifier(
        model_path=os.path.join(models_dir, config.APPEARANCE_MODEL_PATH),
        labelmap_path=os.path.join(models_dir, config.APPEARANCE_LABELMAP_PATH),
        confidence_threshold=config.APPEARANCE_CONFIDENCE_THRESHOLD,
    )
    detector = HybridVehicleDetector(
        mog2_detector=mog2_detector,
        appearance_verifier=verifier,
        object_classifier=ObjectClassifier(
            appearance_verifier=verifier,
            person_max_aspect_ratio=config.PERSON_MAX_ASPECT_RATIO,
            vehicle_min_aspect_ratio=config.VEHICLE_MIN_ASPECT_RATIO,
            model_min_interval=config.CLASSIFIER_MODEL_MIN_INTERVAL,
            memory_timeout=config.CLASSIFIER_MEMORY_TIMEOUT,
            iou_match_threshold=config.CLASSIFIER_IOU_MATCH_THRESHOLD,
            crop_padding=config.APPEARANCE_CROP_PADDING,
        ) if config.CLASSIFIER_ENABLED else None,
        verify_interval=config.APPEARANCE_VERIFY_INTERVAL,
        crop_padding=config.APPEARANCE_CROP_PADDING,
    )

    controller = ActuatedIntersectionController(
        main_pins={"red": config.PIN_MAIN_RED, "yellow": config.PIN_MAIN_YELLOW, "green": config.PIN_MAIN_GREEN},
        branch_pins={"red": config.PIN_BRANCH_RED, "yellow": config.PIN_BRANCH_YELLOW, "green": config.PIN_BRANCH_GREEN},
        arrow_pins={"red": config.PIN_ARROW_RED, "yellow": config.PIN_ARROW_YELLOW, "green": config.PIN_ARROW_GREEN},
        min_green_main=config.MIN_GREEN_MAIN,
        max_green_main=config.MAX_GREEN_MAIN,
        branch_call_min_vehicles=config.BRANCH_CALL_MIN_VEHICLES,
        branch_call_confirm_seconds=config.BRANCH_CALL_CONFIRM_SECONDS,
        min_green_branch=config.MIN_GREEN_BRANCH,
        max_green_branch=config.MAX_GREEN_BRANCH,
        gap_out_seconds=config.BRANCH_GAP_OUT_SECONDS,
        yellow_time=config.YELLOW_TIME,
        all_red_clearance=config.ALL_RED_CLEARANCE,
    )

    logger.info("Sistem berjalan (mode actuated). Tekan Ctrl+C untuk berhenti.")

    frame_counter = 0
    consecutive_fails = 0
    last_status_log = 0.0
    prune_counter = 0

    try:
        while True:
            try:
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= config.CAMERA_MAX_CONSECUTIVE_FAILS:
                        logger.error("Kamera dianggap putus. Mencoba membuka ulang...")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(config.CAMERA_RETRY_DELAY)
                        cap = open_camera()
                        consecutive_fails = 0
                    time.sleep(0.1)
                    continue
                consecutive_fails = 0

                # Skip sebagian frame biar CPU Pi 3 tidak selalu penuh --
                # untuk keperluan hitung kepadatan, tidak perlu proses
                # SETIAP frame yang masuk dari kamera.
                frame_counter += 1
                if frame_counter % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                result = detector.detect(frame)
                if not result["ok"]:
                    continue

                save_frame_atomic(result["frame"])

                # Ini jantung sistem actuated: controller yang menentukan
                # sendiri kapan harus ganti fase berdasarkan data ini.
                controller.update(result["density"], result["count"])

                now = time.time()
                if now - last_status_log >= config.STATUS_LOG_INTERVAL:
                    save_log(
                        result["count"], result["density"],
                        controller.state, controller.elapsed_in_state(),
                    )
                    last_status_log = now

                    prune_counter += 1
                    if prune_counter >= 50:
                        prune_old_logs()
                        prune_counter = 0

            except Exception:
                logger.exception("Error tak terduga di loop utama, lanjut.")
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Sistem dihentikan oleh user.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        controller.cleanup()
        logger.info("Sistem berhenti dengan bersih.")


if __name__ == "__main__":
    main()
