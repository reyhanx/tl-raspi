"""
traffic_controller.py
Kontrol lampu untuk PERTIGAAN dengan logika ACTUATED (demand-based):

- Jalur UTAMA (Pati<->Juwana, arah Timur-Barat, 2 kaki berlawanan arah)
  -> DEFAULT HIJAU terus. Tidak pernah diinterupsi kecuali ada
  permintaan dari cabang, ATAU sudah kelamaan hijau (lihat
  max_green_main -- jaring pengaman supaya kendaraan yang mau belok
  lewat panah gak nunggu tanpa batas kalau Jakenan kebetulan kosong).
- Jalur CABANG (Jakenan, arah Selatan, 1 kaki, dengan kamera) -> hanya
  dapat giliran hijau kalau ada kendaraan terdeteksi menunggu,
  dikonfirmasi beberapa detik dulu (anti false-trigger), dan dibatasi
  durasi min/max.
- Kalau cabang kosong terus -> tetap MERAH, Utama tidak terganggu.
- Antar fase WAJIB ada jeda ALL-RED (semua lampu merah bareng) supaya
  kendaraan terakhir sempat clear sebelum arah lain dapat hijau.

PANAH BELOK (arrow_pins) -- kendaraan arah Timur (dari Pati, menuju
Juwana) yang mau belok ke Jakenan itu MOTONG jalur kendaraan arah Barat
(dari Juwana menuju Pati) yang berlawanan, jadi butuh diatur. Panah ini
SELALU MIRROR PERSIS ke state Cabang -- aman karena pas Cabang hijau,
arus Barat (lawannya) otomatis berhenti. Gak butuh sensor/logika
terpisah sama sekali, tinggal "nebeng" fase Cabang yang sudah ada.
Kendaraan arah Barat (dari Juwana) yang mau ke Jakenan TIDAK butuh
lampu ini -- itu belok kiri yang gak motong siapa2, cukup rambu fisik
"BELOK KIRI LANGSUNG" (di luar scope sistem ini).

CATATAN LIBRARY GPIO:
Pakai `gpiozero` (bukan RPi.GPIO), karena RPi.GPIO sudah tidak
direkomendasikan resmi oleh Raspberry Pi Foundation dan bermasalah
di Raspberry Pi OS Bookworm/Trixie. gpiozero pakai backend `lgpio`.
Install: sudo apt install python3-gpiozero python3-lgpio

PRINSIP FAIL-SAFE:
- Start selalu dari ALL-RED dulu (clearance), baru masuk default
  Utama-hijau.
- try/finally memastikan setiap transisi fase selalu berakhir aman
  (all-red) walau terjadi error/interrupt di tengah jalan.
- atexit & signal handler memastikan GPIO dibersihkan walau proses
  dikill paksa (Ctrl+C, systemd stop, SIGTERM, mati listrik lalu restart).
"""

import atexit
import signal
import time

from logger_setup import get_logger

logger = get_logger("traffic_controller")

try:
    from gpiozero import LED
    ON_PI = True
except ImportError:
    ON_PI = False
    logger.warning("gpiozero tidak ditemukan, jalan dalam mode simulasi.")


def _make_pins(pin_dict):
    if not ON_PI:
        return {k: None for k in pin_dict}
    return {name: LED(pin) for name, pin in pin_dict.items()}


# Nama state untuk state machine
STATE_MAIN_GREEN = "MAIN_GREEN"      # Utama hijau (default, boleh berlangsung lama)
STATE_BRANCH_GREEN = "BRANCH_GREEN"  # Cabang lagi dapat giliran hijau
STATE_TRANSITIONING = "TRANSITIONING"  # Lagi proses ganti fase (kuning/all-red)


class ActuatedIntersectionController:
    def __init__(self, main_pins, branch_pins, arrow_pins=None,
                 min_green_main=15, max_green_main=90,
                 branch_call_min_vehicles=1, branch_call_confirm_seconds=3,
                 min_green_branch=8, max_green_branch=30, gap_out_seconds=4,
                 yellow_time=3, all_red_clearance=2, on_light_change=None):
        """
        arrow_pins: dict {"red","yellow","green"} untuk panah belok
        Pati->Jakenan (arah Timur). Opsional -- kalau None, fitur ini
        dilewati sepenuhnya (sistem tetap jalan normal 2 fase seperti
        sebelumnya, cuma tanpa panah belok).

        max_green_main: JARING PENGAMAN. Kamera cuma mengintai Jakenan,
        gak bisa tahu ada kendaraan dari Pati yang nunggu mau belok
        (butuh panah hijau, yang cuma nyala bareng fase Cabang). Kalau
        Jakenan kebetulan kosong terus, tanpa ini Utama bisa hijau
        SELAMANYA dan si pembelok nunggu tanpa kepastian. Parameter ini
        memaksa Cabang+panah dapat giliran singkat secara berkala,
        walau Jakenan kelihatan kosong -- kalau beneran kosong, langsung
        disudahi cepat lewat gap-out yang sudah ada.

        on_light_change: callback opsional dipanggil SETIAP KALI warna
        lampu beneran berubah (termasuk kuning & all-red yang cuma
        berlangsung sebentar saat transisi). Dipakai main.py buat catat
        status ke dashboard secara instan -- tanpa ini, kuning/all-red
        gak akan pernah "kelihatan" di dashboard karena periodic log
        biasa gak sempat nangkep momen yang cuma 2-3 detik itu (seluruh
        transisi terjadi di 1 pemanggilan update() yang blocking).
        Signature: on_light_change(detail_state: str)
        """
        self._main_pin_numbers = main_pins
        self._branch_pin_numbers = branch_pins
        self._arrow_pin_numbers = arrow_pins
        self.main_devices = _make_pins(main_pins)
        self.branch_devices = _make_pins(branch_pins)
        self.arrow_devices = _make_pins(arrow_pins) if arrow_pins else None
        self.on_light_change = on_light_change

        self.min_green_main = min_green_main
        self.max_green_main = max_green_main
        self.branch_call_min_vehicles = branch_call_min_vehicles
        self.branch_call_confirm_seconds = branch_call_confirm_seconds
        self.min_green_branch = min_green_branch
        self.max_green_branch = max_green_branch
        self.gap_out_seconds = gap_out_seconds
        self.yellow_time = yellow_time
        self.all_red_clearance = all_red_clearance

        self._cleaned_up = False

        # Timer/state internal
        self.state = None        # state KASAR: MAIN_GREEN/BRANCH_GREEN/TRANSITIONING
                                   # -- ini yang dipakai logika update() di atas,
                                   # JANGAN diubah maknanya, banyak logic bergantung ini.
        self.detail_state = None  # state HALUS buat pelaporan/tampilan: termasuk
                                   # MAIN_YELLOW, ALL_RED, BRANCH_YELLOW yang gak
                                   # kebedain di `self.state` di atas.
        self.main_green_since = None
        self.branch_call_since = None      # kapan demand cabang mulai terdeteksi
                                             # terus-menerus (buat confirm window)
        self.branch_green_since = None
        self.branch_last_vehicle_seen = None

        atexit.register(self.cleanup)
        if ON_PI:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

        self.start()

    def _handle_signal(self, signum, frame):
        logger.info(f"Menerima sinyal {signum}, mematikan sistem dengan aman...")
        self.cleanup()
        raise SystemExit(0)

    # ---------- Kontrol GPIO level rendah ----------

    def _apply(self, devices, red, yellow, green):
        if devices is None:
            return
        try:
            if ON_PI:
                self._set_led(devices["red"], red)
                self._set_led(devices["yellow"], yellow)
                self._set_led(devices["green"], green)
        except Exception:
            logger.exception("Gagal set GPIO, memaksa failsafe ke merah")
            self._failsafe()

    def _apply_branch_group(self, red, yellow, green):
        """Branch dan panah belok SELALU disetel BARENG dengan nilai
        yang SAMA PERSIS di titik manapun -- ini satu-satunya tempat
        yang boleh menyentuh branch_devices/arrow_devices, supaya gak
        ada resiko lupa sinkronisasi kalau nanti kode ini direvisi lagi."""
        self._apply(self.branch_devices, red, yellow, green)
        self._apply(self.arrow_devices, red, yellow, green)

    @staticmethod
    def _set_led(led, on):
        if on:
            led.on()
        else:
            led.off()

    def _set_detail_state(self, detail_state):
        """Catat state halus + lapor instan ke callback (kalau ada).
        INI yang bikin dashboard bisa nangkep momen kuning/all-red yang
        cuma berlangsung sebentar -- dipanggil PERSIS di titik lampu
        beneran berubah, bukan nunggu periodic log."""
        self.detail_state = detail_state
        if self.on_light_change is not None:
            try:
                self.on_light_change(detail_state)
            except Exception:
                logger.exception("on_light_change callback error (non-fatal, lampu tetap jalan)")

    def _all_red(self):
        self._apply(self.main_devices, red=True, yellow=False, green=False)
        self._apply_branch_group(red=True, yellow=False, green=False)
        self._set_detail_state("ALL_RED")
        if not ON_PI:
            logger.info("[SIMULASI] -> ALL RED (clearance)")

    def _failsafe(self):
        """Simpang mati total (semua merah) lebih aman daripada 2 arah
        berpotongan sama-sama hijau akibat error tak terduga."""
        try:
            if ON_PI:
                all_groups = [self.main_devices, self.branch_devices]
                if self.arrow_devices:
                    all_groups.append(self.arrow_devices)
                for devices in all_groups:
                    for led in devices.values():
                        led.off()
                    devices["red"].on()
            self.state = STATE_TRANSITIONING
            self._set_detail_state("ALL_RED")
        except Exception:
            logger.exception("Failsafe pun gagal — cek wiring/hardware GPIO")

    # ---------- State machine ----------

    def start(self):
        """Startup selalu dari ALL-RED dulu (clearance), baru masuk ke
        default Utama-hijau. Ini paling aman kalau sistem baru nyala
        (misal habis listrik mati) -- jangan langsung hijau tanpa jeda."""
        self.state = STATE_TRANSITIONING
        self._all_red()
        time.sleep(self.all_red_clearance)
        self._go_main_green()

    def _go_main_green(self):
        self._apply(self.main_devices, red=False, yellow=False, green=True)
        self._apply_branch_group(red=True, yellow=False, green=False)
        self.state = STATE_MAIN_GREEN
        self._set_detail_state("MAIN_GREEN")
        self.main_green_since = time.time()
        self.branch_call_since = None
        logger.info("Fase UTAMA (Pati-Juwana) hijau -- state default. Panah belok ikut merah.")

    def elapsed_in_state(self):
        now = time.time()
        if self.state == STATE_MAIN_GREEN and self.main_green_since:
            return int(now - self.main_green_since)
        if self.state == STATE_BRANCH_GREEN and self.branch_green_since:
            return int(now - self.branch_green_since)
        return 0

    def update(self, branch_density, branch_vehicle_count):
        """Dipanggil berulang oleh main.py tiap ada pembacaan kamera baru.
        Berisi seluruh logika actuated: kapan cabang boleh diservis,
        kapan harus balik ke Utama. Method ini TIDAK blocking lama kecuali
        pas transisi fase (yang memang butuh beberapa detik kuning+all-red,
        itu bagian dari keselamatan, bukan bug)."""
        try:
            now = time.time()

            if self.state == STATE_MAIN_GREEN:
                main_elapsed = now - self.main_green_since
                has_demand = branch_vehicle_count >= self.branch_call_min_vehicles

                # Jaring pengaman: walau gak ada demand dari kamera Jakenan
                # sama sekali, paksa servis Cabang+panah kalau Utama udah
                # kelamaan hijau -- supaya kendaraan dari Pati yang mau
                # belok (butuh panah hijau) gak nunggu tanpa batas.
                if main_elapsed >= self.max_green_main:
                    logger.info(
                        f"Utama sudah hijau {main_elapsed:.0f}s (>= batas maksimum "
                        f"{self.max_green_main}s). Servis fase CABANG+panah walau "
                        f"Jakenan kelihatan kosong -- beri kesempatan kendaraan Pati "
                        f"yang mau belok."
                    )
                    self._transition_main_to_branch()

                elif has_demand:
                    if self.branch_call_since is None:
                        self.branch_call_since = now
                        logger.info(
                            f"Kendaraan terdeteksi di jalur cabang "
                            f"({branch_vehicle_count} kendaraan), mulai konfirmasi..."
                        )
                    confirm_elapsed = now - self.branch_call_since

                    if (confirm_elapsed >= self.branch_call_confirm_seconds
                            and main_elapsed >= self.min_green_main):
                        logger.info(
                            f"Demand cabang terkonfirmasi ({confirm_elapsed:.1f}s) dan "
                            f"Utama sudah hijau {main_elapsed:.0f}s (>= minimum "
                            f"{self.min_green_main}s). Servis fase CABANG."
                        )
                        self._transition_main_to_branch()
                else:
                    if self.branch_call_since is not None:
                        logger.info("Demand cabang hilang sebelum terkonfirmasi, reset.")
                    self.branch_call_since = None

            elif self.state == STATE_BRANCH_GREEN:
                if branch_vehicle_count >= self.branch_call_min_vehicles:
                    self.branch_last_vehicle_seen = now

                branch_elapsed = now - self.branch_green_since
                no_vehicle_for = now - (self.branch_last_vehicle_seen or self.branch_green_since)

                if branch_elapsed >= self.max_green_branch:
                    logger.info(
                        f"Fase CABANG mencapai batas maksimum ({self.max_green_branch}s), "
                        f"kembali ke UTAMA."
                    )
                    self._transition_branch_to_main()
                elif (branch_elapsed >= self.min_green_branch
                        and no_vehicle_for >= self.gap_out_seconds):
                    logger.info(
                        f"Jalur cabang kosong selama {no_vehicle_for:.0f}s (gap-out) "
                        f"setelah hijau {branch_elapsed:.0f}s, kembali ke UTAMA lebih awal."
                    )
                    self._transition_branch_to_main()

            # STATE_TRANSITIONING: sedang di tengah pergantian fase (blocking
            # call sedang berjalan di thread yang sama), tidak perlu aksi apa2
            # di sini karena update() dipanggil sekuensial dari loop utama.

        except Exception:
            logger.exception("Error tak terduga di update(), failsafe ke merah semua.")
            self._failsafe()

    def _transition_main_to_branch(self):
        self.state = STATE_TRANSITIONING
        try:
            self._apply(self.main_devices, red=False, yellow=True, green=False)
            self._set_detail_state("MAIN_YELLOW")
            time.sleep(self.yellow_time)
        finally:
            self._all_red()
            time.sleep(self.all_red_clearance)

        # Branch DAN panah belok hijau BARENG -- panah ini aman nyala
        # karena arus lawannya (arah Barat) udah pasti berhenti di titik ini.
        self._apply_branch_group(red=False, yellow=False, green=True)
        self.state = STATE_BRANCH_GREEN
        self._set_detail_state("BRANCH_GREEN")
        self.branch_green_since = time.time()
        self.branch_last_vehicle_seen = time.time()
        logger.info("Fase CABANG (Jakenan) hijau. Panah belok Pati->Jakenan ikut hijau.")

    def _transition_branch_to_main(self):
        self.state = STATE_TRANSITIONING
        try:
            self._apply_branch_group(red=False, yellow=True, green=False)
            self._set_detail_state("BRANCH_YELLOW")
            time.sleep(self.yellow_time)
        finally:
            self._all_red()
            time.sleep(self.all_red_clearance)

        self._go_main_green()

    # ---------- Cleanup ----------

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        if ON_PI:
            try:
                all_groups = [self.main_devices, self.branch_devices]
                if self.arrow_devices:
                    all_groups.append(self.arrow_devices)
                for devices in all_groups:
                    for led in devices.values():
                        led.off()
                        led.close()
                logger.info("GPIO cleanup selesai.")
            except Exception:
                logger.exception("Gagal cleanup GPIO")


if __name__ == "__main__":
    # Testing mandiri: simulasikan cabang sepi lalu tiba-tiba ada kendaraan.
    # Perhatikan log "Panah belok" -- harus SELALU muncul bareng log Cabang.
    controller = ActuatedIntersectionController(
        main_pins={"red": 17, "yellow": 27, "green": 22},
        branch_pins={"red": 23, "yellow": 24, "green": 25},
        arrow_pins={"red": 5, "yellow": 6, "green": 13},
        min_green_main=5,           # dipercepat khusus buat testing
        branch_call_confirm_seconds=2,
        min_green_branch=4,
        max_green_branch=10,
        gap_out_seconds=3,
    )
    try:
        logger.info("Simulasi: cabang sepi 6 detik...")
        for _ in range(6):
            controller.update(branch_density=0.0, branch_vehicle_count=0)
            time.sleep(1)

        logger.info("Simulasi: ada kendaraan muncul di cabang...")
        for _ in range(15):
            controller.update(branch_density=0.1, branch_vehicle_count=2)
            time.sleep(1)
            logger.info(f"State: {controller.state}, elapsed: {controller.elapsed_in_state()}s")
    finally:
        controller.cleanup()
