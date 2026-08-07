"""
dashboard/app.py
Web dashboard sederhana pakai Flask untuk monitoring.

Catatan produksi/jangka-panjang:
- debug=False (dari config.py) — mode debug Flask boros memori dan
  auto-reloader-nya bisa membuat proses dobel, tidak cocok untuk
  perangkat yang jalan lama tanpa pengawasan seperti Pi.
- threaded=True — supaya request /api/frame yang sering (polling tiap
  2 detik) tidak saling menunggu/blocking.
- Semua akses DB dibungkus try/except supaya dashboard tetap bisa
  menampilkan halaman walau database sedang locked/belum ada data.
"""

import os
import sys
import sqlite3
from flask import Flask, jsonify, render_template, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from logger_setup import get_logger

logger = get_logger("dashboard")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "traffic_log.db")
FRAME_PATH = os.path.join(BASE_DIR, "data", "latest_frame.jpg")


def query_logs(limit=50):
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        logger.exception("Gagal membaca database, mengembalikan data kosong.")
        return []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/logs")
def api_logs():
    return jsonify(query_logs(50))


@app.route("/api/latest")
def api_latest():
    logs = query_logs(1)
    return jsonify(logs[0] if logs else {})


@app.route("/api/frame")
def api_frame():
    if os.path.exists(FRAME_PATH):
        try:
            return send_file(FRAME_PATH, mimetype="image/jpeg")
        except Exception:
            logger.exception("Gagal mengirim frame preview.")
            return "", 404
    return "", 404


@app.route("/api/config")
def api_config():
    return jsonify({
        "min_green_main": config.MIN_GREEN_MAIN,
        "min_green_branch": config.MIN_GREEN_BRANCH,
        "max_green_branch": config.MAX_GREEN_BRANCH,
        "branch_call_confirm_seconds": config.BRANCH_CALL_CONFIRM_SECONDS,
        "gap_out_seconds": config.BRANCH_GAP_OUT_SECONDS,
        "yellow_time": config.YELLOW_TIME,
        "all_red_clearance": config.ALL_RED_CLEARANCE,
    })


@app.route("/api/health")
def api_health():
    """Endpoint sederhana untuk cek sistem masih hidup — berguna kalau
    nanti mau ditambah monitoring eksternal (uptime checker, dsb)."""
    return jsonify({
        "status": "ok",
        "db_exists": os.path.exists(DB_PATH),
        "frame_exists": os.path.exists(FRAME_PATH),
    })


if __name__ == "__main__":
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
        threaded=True,
    )
