"""
appearance_verifier.py
Verifikasi keberadaan kendaraan berdasarkan BENTUK (appearance), bukan
gerakan -- pakai model SSD MobileNetV1 (TFLite, quantized, pretrained
dataset COCO) lewat library `ai-edge-litert`.

KENAPA MODUL INI ADA (lihat diskusi lengkap di HANDOFF.md bagian 10):
Background subtraction (MOG2) di vehicle_detection.py itu berbasis
GERAKAN -- begitu kendaraan berhenti lama (misal ngantri nunggu lampu),
lama-lama "ditelan" ke background dan kotak deteksinya hilang padahal
kendaraannya masih ada secara fisik. Modul ini jadi lapisan kedua yang
BENERAN ngenalin bentuk kendaraan, jadi imun terhadap masalah itu.

DIPANGGIL SESEKALI, BUKAN TIAP FRAME (lihat config.APPEARANCE_VERIFY_INTERVAL):
Model ini fixed input 300x300 (bawaan arsitekturnya, gak bisa diperkecil
lagi tanpa re-training/convert ulang), jadi biaya komputasi per
panggilan itu TETAP walau crop yang dimasukkan kecil -- penghematannya
BUKAN dari ukuran crop, tapi dari FREKUENSI panggilan yang rendah +
model yang sudah di-quantize INT8. Di Pi 3, 1x inferensi model ini
kira-kira ~1 detik -- karena cuma dipanggil tiap beberapa detik (bukan
tiap frame), ini masih dalam anggaran CPU yang wajar, TAPI itu artinya
tiap kali dipanggil, loop utama akan "nge-pause" sekitar 1 detik.
Untuk versi pertama ini sengaja dibuat SYNCHRONOUS (blocking) dulu demi
kesederhanaan & gampang di-debug -- kalau nanti kerasa terlalu
mengganggu responsivitas sistem, jalanin di thread terpisah adalah
optimasi lanjutan yang bisa ditambahkan (belum diimplementasi di sini).

CARA DAPETIN FILE MODEL (jalankan di Raspberry Pi, BUKAN dari sini):
    cd ~/traffic-light-system
    mkdir -p models
    cd models
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    mv detect.tflite detect.tflite
    mv labelmap.txt labelmap.txt
    # hasil akhir: models/detect.tflite dan models/labelmap.txt

INSTALL LIBRARY (di Raspberry Pi):
    pip3 install ai-edge-litert --break-system-packages
"""

import numpy as np
import cv2

from logger_setup import get_logger

logger = get_logger("appearance_verifier")

try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False
    logger.warning(
        "ai_edge_litert tidak ditemukan -- appearance verification akan "
        "otomatis nonaktif (sistem tetap jalan normal pakai MOG2 saja)."
    )

# Label COCO yang dianggap "kendaraan". Nama-nama ini harus PERSIS sama
# dengan yang ada di labelmap.txt bawaan model coco_ssd_mobilenet_v1.
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


class AppearanceVerifier:
    def __init__(self, model_path, labelmap_path, confidence_threshold=0.5):
        self.enabled = False
        self.interpreter = None
        self.confidence_threshold = confidence_threshold
        self.input_w = None
        self.input_h = None

        if not LITERT_AVAILABLE:
            return

        try:
            self.labels = self._load_labels(labelmap_path)
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_h = self.input_details[0]["shape"][1]
            self.input_w = self.input_details[0]["shape"][2]
            self.enabled = True
            logger.info(f"Model appearance verification siap ({self.input_w}x{self.input_h}).")
        except FileNotFoundError:
            logger.warning(
                f"File model/labelmap tidak ditemukan ({model_path} / {labelmap_path}). "
                "Appearance verification nonaktif -- jalankan langkah download di "
                "docstring modul ini. Sistem tetap jalan normal pakai MOG2 saja."
            )
            self.enabled = False
        except Exception:
            logger.exception("Gagal load model appearance verification, fitur ini dinonaktifkan.")
            self.enabled = False

    @staticmethod
    def _load_labels(path):
        with open(path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        # Kuirk resmi model coco_ssd_mobilenet_v1: baris pertama labelmap.txt
        # itu placeholder "???" yang harus dibuang supaya index label pas
        # sama index kelas yang dikeluarkan model. Ini praktik standar dari
        # implementasi referensi resmi (EdjeElectronics TFLite_detection_*),
        # bukan asumsi saya sendiri.
        if labels and labels[0] == "???":
            del labels[0]
        return labels

    def verify(self, crop_bgr):
        """
        crop_bgr: potongan frame (numpy array BGR dari OpenCV) di sekitar
        posisi kendaraan yang terakhir diketahui.
        Return True kalau appearance model mengonfirmasi ada kendaraan di
        crop itu, False kalau tidak ATAU kalau verifier sedang nonaktif
        (caller harus anggap False sebagai "tidak bisa dipastikan", bukan
        bukti kendaraan benar-benar tidak ada).
        """
        if not self.enabled:
            return False

        if crop_bgr is None or crop_bgr.size == 0:
            return False

        try:
            resized = cv2.resize(crop_bgr, (self.input_w, self.input_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            # Model coco_ssd_mobilenet_v1 sudah termasuk TFLite_Detection_PostProcess
            # (NMS dsb sudah dilakukan di dalam graph), output index standarnya:
            # [0]=boxes, [1]=classes, [2]=scores, [3]=num_detections
            classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

            for cls_id, score in zip(classes, scores):
                if score < self.confidence_threshold:
                    continue
                idx = int(cls_id)
                if 0 <= idx < len(self.labels) and self.labels[idx] in VEHICLE_LABELS:
                    logger.info(f"Appearance verification: '{self.labels[idx]}' terkonfirmasi (skor {score:.2f}).")
                    return True
            return False

        except Exception:
            logger.exception("Error saat inferensi appearance verification.")
            return False
