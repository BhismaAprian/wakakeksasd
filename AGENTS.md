# AGENTS.md — Kontrak Kerja Kompetisi Klasifikasi Bencana

## 1. Misi
Membangun model klasifikasi 6 kelas fenomena (`Earthquake`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, `Explosion`) dari pasangan citra `pre.jpg` dan `post.jpg` untuk setiap `Id`.

## 2. Batasan Teknis yang Tidak Boleh Dilanggar
- **Metrik Tunggal**: **Macro F1-Score**. Jangan pernah melaporkan akurasi sebagai metrik utama.
- **Protokol Validasi**: Gunakan **Stratified 5-Fold Cross-Validation** pada label `Phenomenon`. Simpan kolom `fold` ke dalam data latih agar semua eksperimen menggunakan permukaan perbandingan yang identik.
- **Arsitektur Inti**: **Siamese Network** dengan backbone **EfficientNet-B4** (atau varian EfficientNet/ConvNeXt lain). Jangan menggunakan arsitektur yang memproses `post.jpg` saja tanpa membandingkannya dengan `pre.jpg`.
- **Mekanisme Fusion**: Wajib mengeksplorasi setidaknya dua pendekatan: (a) *Absolute Difference* fitur (`|F_pre - F_post|`), dan (b) *Concatenation* fitur.
- **Augmentasi**: Terapkan augmentasi geometrik dan fotometrik yang **identik** pada pasangan `pre` dan `post` (mis. `RandomHorizontalFlip`, `ColorJitter`). Untuk mensimulasikan variasi iluminasi, `ColorJitter` boleh diterapkan secara independen pada masing-masing citra.

## 3. Alur Kerja Eksperimen Wajib
Setiap kali diminta menjalankan eksperimen, ikuti urutan ini:
1.  **EDA Otomatis**: Laporkan distribusi kelas, deteksi `Id` duplikat, dan hitung statistik sederhana perubahan piksel (mis. rata-rata selisih absolut) per kelas.
2.  **Baseline**: Latih Siamese Network dengan konfigurasi default (EfficientNet-B4, *absolute difference fusion*) menggunakan 5-fold CV. Laporkan **OOF Macro F1 per kelas**.
3.  **Iterasi**: Ubah **satu variabel pada satu waktu** (mis. ganti fusion ke *concatenation*, atau ganti backbone ke ConvNeXt-Small). Bandingkan OOF Macro F1 dengan baseline.
4.  **Dokumentasi**: Catat setiap eksperimen (parameter, skor CV) ke file `EXP_SUMMARY.md` di root proyek.
5.  **Gate Kualitas**: Hanya usulkan model untuk di-*ensemble* jika skor CV-nya kompetitif atau memberikan keragaman prediksi yang jelas.

## 4. Larangan
- Jangan melakukan *submission* ke Kaggle tanpa persetujuan eksplisit dari manusia (Anda).
- Jangan menggunakan `train_test_split` acak sederhana untuk evaluasi.
- Jangan mengubah protokol validasi (jumlah fold, stratifikasi) di tengah siklus eksperimen.

## 5. Skill yang Tersedia (Workspace-Local)
Skill berikut terpasang di `.agents/skills/` dan WAJIB digunakan sesuai konteks:

- `computer-vision-expert` → kerangka berpikir CV SOTA (pemilihan pendekatan change detection, VLM, segmentasi).
- `computer-vision-deep` → implementasi detection/segmentation/depth; keputusan klasik vs deep learning.
- `computer-vision-opencv` → pipeline OpenCV + PyTorch untuk I/O, filtering, transformasi geometrik.
- `space-data-processing` → domain satellite imagery & change detection (pemicu utama: perbandingan pre vs post).
- `geospatial-analysis` → operasi raster, temporal image stacks, environmental change detection.
- `rioxarray-remote-sensing` → membaca citra multispektral, cloud masking, NDVI/EVI.
- `kaggle-grandmaster` → mode berpikir strategis; dimuat di AWAL setiap sesi kompetisi.
- `agentic-kaggle-skill` → orkestrasi eksperimen (EDA → baseline → iterasi → OOF), mendukung task image.

ATURAN PEMAKAIAN:
- Untuk EDA dan preprocessing citra, gabungkan `computer-vision-opencv` + `space-data-processing` + `rioxarray-remote-sensing`.
- Untuk memilih arsitektur change detection, gunakan `computer-vision-expert` + `computer-vision-deep`.
- Untuk menjaga protokol Macro F1 dan Stratified 5-Fold, patuhi `kaggle-grandmaster` + `agentic-kaggle-skill` + Bagian 2 AGENTS.md.
- Jangan mengubah protokol validasi atau arsitektur inti (Siamese Network + EfficientNet-B4) tanpa persetujuan manusia.

## 6. Keputusan Arsitektur Terkunci (Pasca-Audit Forensik)
Berdasarkan Audit Forensik Matematis pada `3.180` pasangan citra (`reports/FORENSIC_AUDIT_REPORT.md` / `artifacts_synced/FORENSIC_AUDIT_REPORT.md`), disahkan oleh manusia:
- **Symmetric Temporal-Swap Invariant Fusion:** Gabungkan `[0.5*(f_pre + f_post), max(f_pre, f_post), |f_post - f_pre|]` (dengan *Channel Attention* pada `|f_post - f_pre|`) agar model kebal terhadap inversi kronologis `pre ↔ post` (`33–45%` sampel).
- **Random Temporal Swap (`p=0.5`)** saat training dan **Bidirectional Swap-TTA** (`0.5 * (P(pre, post) + P(post, pre))`) saat validasi & inferensi.
- **Kunci Validasi:** Gunakan `artifacts_synced/folds_skf5.npy` (`StratifiedKFold(5, random_state=42)` pada `pseudo_labels_v2.parquet`) secara konsisten di seluruh eksperimen.
