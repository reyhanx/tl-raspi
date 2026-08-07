"""
logger_setup.py
Logging terpusat. Ganti semua print() debug dengan logger supaya:
- Ada timestamp & level (INFO/WARNING/ERROR) yang jelas
- Log tersimpan ke file (bisa dicek setelah sistem jalan berhari-hari
  tanpa harus terus buka terminal)
- File log otomatis di-rotate, tidak membesar tak terbatas dan
  menghabiskan storage SD card
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        # Sudah pernah di-setup sebelumnya (hindari duplikat handler)
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
