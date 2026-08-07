"""
config.py
Konfigurasi terpusat. Ubah parameter di sini saja — jangan ubah
langsung di dalam main.py / vehicle_detection.py / traffic_controller.py,
supaya tidak ada nilai yang tercecer dan tidak sinkron.

DESAIN SIMPANG: PERTIGAAN dengan logika ACTUATED
Studi kasus: Pertigaan Sampang, Pati (Jalan Pantura Pati-Rembang +
cabang ke arah Jakenan).

- Jalur UTAMA (Pantura, 2 kaki berlawanan arah, lurus/tidak
  berpotongan) -> DEFAULT HIJAU terus-menerus. Ini jalan nasional
  dengan volume tinggi & konstan, tidak boleh diinterupsi tanpa alasan.
- Jalur CABANG (Jakenan, 1 kaki) -> kamera dipasang di sini. Cabang
  HANYA dapat giliran hijau kalau ada kendaraan terdeteksi menunggu
  ("actuated" / demand-based), bukan giliran tetap seperti round-robin.
- Kalau cabang sepi terus -> tetap MERAH, Utama tidak pernah diganggu.
- Utara & Selatan (Utama) dikontrol 1 set pin GPIO yang sama (selalu
  bareng). Cabang cukup 1 pole/1 set pin sendiri (cuma 1 kaki fisik).
"""

# ---- GPIO (BCM numbering) ----
# Jalur UTAMA (Pantura) -- default hijau, 2 tiang lampu paralel
PIN_MAIN_RED = 17
PIN_MAIN_YELLOW = 27
PIN_MAIN_GREEN = 22

# Jalur CABANG (Jakenan) -- kaki dengan kamera, actuated (demand-based)
PIN_BRANCH_RED = 23
PIN_BRANCH_YELLOW = 24
PIN_BRANCH_GREEN = 25

# ---- Kamera (dipasang menghadap jalur CABANG) ----
CAMERA_INDEX = 0
CAMERA_RETRY_DELAY = 3          # detik, jeda sebelum mencoba buka ulang kamera
CAMERA_MAX_CONSECUTIVE_FAILS = 30  # kalau read() gagal seberapa kali berturut-turut
                                     # sebelum kamera dianggap putus & dibuka ulang
PROCESS_EVERY_N_FRAMES = 5      # skip frame supaya CPU Pi 3 tidak terus dipakai
                                  # penuh -- cukup proses 1 dari N frame yang masuk

# ---- Deteksi kendaraan ----
DETECTION_MIN_AREA = 800
DETECTION_RESIZE_WIDTH = 320
MOG2_LEARNING_RATE = 0.0008      # kecil = MOG2 lebih lambat "melupakan"
                                   # objek diam ke background. -1 = default
                                   # OpenCV (cepat lupa, rawan masalah
                                   # kendaraan diam lama hilang dari deteksi)

# ---- Appearance verification (lapisan ke-2, opsional) ----
# Mengatasi masalah "kendaraan berhenti lama hilang dari deteksi MOG2"
# dengan verifikasi berbasis BENTUK, bukan gerakan. Lihat
# appearance_verifier.py dan HANDOFF.md bagian 10 untuk detail lengkap.
# Kalau file model belum ada / ai-edge-litert belum terinstall, fitur
# ini otomatis nonaktif dan sistem tetap jalan normal pakai MOG2 saja.
APPEARANCE_MODEL_PATH = "models/detect.tflite"
APPEARANCE_LABELMAP_PATH = "models/labelmap.txt"
APPEARANCE_CONFIDENCE_THRESHOLD = 0.5
APPEARANCE_VERIFY_INTERVAL = 2.0   # detik, seberapa sering verifikasi ulang
APPEARANCE_CROP_PADDING = 25       # px, padding di sekitar posisi terakhir
                                     # diketahui supaya kendaraan gak kepotong
                                     # saat di-crop untuk verifikasi

# ---- Object classifier: filter pejalan kaki/non-kendaraan (opsional) ----
# 3 lapis (geometris -> model TFLite -> memori), lihat object_classifier.py.
# Pakai appearance_verifier yang sama (model & library sama, cuma dipanggil
# lebih sering/dini di pipeline). Kalau mau nonaktifin fitur ini tanpa
# ubah kode, set CLASSIFIER_ENABLED = False.
CLASSIFIER_ENABLED = True
# Rasio lebar/tinggi bbox. NILAI AWAL/TEBAKAN -- WAJIB dikalibrasi ulang
# setelah kamera terpasang fisik di posisi & sudut sebenarnya (sama
# seperti DETECTION_MIN_AREA).
PERSON_MAX_ASPECT_RATIO = 0.6      # di bawah ini -> dianggap "orang" (tinggi & sempit)
VEHICLE_MIN_ASPECT_RATIO = 1.2     # di atas ini -> dianggap "kendaraan" (lebar)
                                     # di antara keduanya -> ambigu, eskalasi ke model
CLASSIFIER_MODEL_MIN_INTERVAL = 1.0   # detik, jarak minimum antar panggilan
                                        # model untuk objek ambigu (rate limit)
CLASSIFIER_MEMORY_TIMEOUT = 5.0       # detik, berapa lama hasil klasifikasi
                                        # "diingat" untuk 1 objek sebelum
                                        # dilupakan (kalau objeknya udah gak
                                        # kelihatan lagi sama sekali)
CLASSIFIER_IOU_MATCH_THRESHOLD = 0.3  # seberapa besar overlap bbox supaya
                                        # dianggap "objek yang sama" dengan
                                        # yang sudah pernah diklasifikasi

# ---- Logika actuated: kapan cabang dianggap "ada yang menunggu" ----
BRANCH_CALL_MIN_VEHICLES = 1     # minimal berapa kendaraan kedetect di cabang
                                   # supaya dianggap ada demand
BRANCH_CALL_CONFIRM_SECONDS = 3  # demand harus konsisten selama sekian detik
                                   # dulu sebelum sistem ganti fase (anti false
                                   # trigger dari noise/1 frame doang)

# ---- Durasi & pengaman fase ----
MIN_GREEN_MAIN = 15              # Utama wajib hijau minimal segini dulu sebelum
                                   # boleh "diganggu" permintaan dari cabang
MIN_GREEN_BRANCH = 8             # Cabang begitu dapat giliran, hijau minimal segini
MAX_GREEN_BRANCH = 30            # ...dan maksimal segini, supaya Utama tidak
                                   # "disandera" kelamaan kalau cabang rame terus
BRANCH_GAP_OUT_SECONDS = 4       # kalau selama fase cabang tidak ada kendaraan
                                   # baru terdeteksi selama sekian detik (dan
                                   # MIN_GREEN_BRANCH sudah lewat), fase cabang
                                   # diakhiri lebih awal -> balik ke Utama cepat

YELLOW_TIME = 3
ALL_RED_CLEARANCE = 2            # jeda semua-merah antar fase, WAJIB ada
                                   # untuk keselamatan (kendaraan terakhir
                                   # sempat clear dari persimpangan)

# ---- Database & logging ----
STATUS_LOG_INTERVAL = 5          # detik, seberapa sering status disimpan ke DB
                                   # untuk keperluan dashboard (bukan tiap frame,
                                   # supaya tidak membebani I/O SD card)
MAX_LOG_ROWS = 5000              # log lama otomatis dipangkas di atas jumlah ini
LOG_FILE_MAX_BYTES = 2 * 1024 * 1024   # 2MB per file log
LOG_FILE_BACKUP_COUNT = 3

# ---- Dashboard ----
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = False          # WAJIB False saat dipakai jangka panjang / demo,
                                   # debug mode Flask boros memori & auto-reload
                                   # bisa bikin proses dobel
