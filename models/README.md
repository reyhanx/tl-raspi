# Model untuk Appearance Verification

Folder ini harus berisi 2 file (belum di-include di zip karena file
binary model-nya lumayan besar dan harus didownload langsung dari Pi,
bukan lewat sini):

- `detect.tflite` — model SSD MobileNetV1 (quantized, pretrained COCO)
- `labelmap.txt` — daftar 90 label kelas COCO

## Cara download (jalankan di Raspberry Pi)

```bash
cd ~/traffic-light-system/models
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
rm coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
ls
# harus muncul: detect.tflite  labelmap.txt
```

Model ini dari Google resmi, ukuran ~5MB, sudah dipakai luas di banyak
tutorial TFLite Raspberry Pi selama bertahun-tahun jadi cukup stabil
linknya.

## Kalau fitur ini belum mau dipakai dulu

Gak apa-apa — sistem (`main.py`) tetap jalan normal tanpa file ini.
`appearance_verifier.py` akan otomatis mendeteksi file tidak ada, catat
warning ke log, dan sistem berjalan pakai deteksi MOG2 biasa saja
(persis seperti sebelum fitur ini ditambahkan).
