# EXP_SUMMARY — EDA & Preprocessing Kompetisi Fenomena Bencana

## 1. Status Data & Verifikasi Label
- **Total Id:** 3,180 pasangan (`pre.jpg` + `post.jpg`), seluruhnya valid (`0` missing, `0` corrupt, `0` duplikat MD5).
- **Resolusi & Mode Warna:** Seragam `1024x1024` RGB (`3.180/3.180`), tanpa kanal multispektral NIR (`NDVI = None`).
- **Status Label:** **TIDAK ADA LABEL TRAINING** (`HAS_LABELS = False`). Satu-satunya CSV yang tersedia adalah `/marimo/sample_submission.csv` (`3.180` baris, seluruhnya placeholder `'Earthquake'`).

## 2. Temuan Kunci EDA
1. **Ketiadaan Label Ground-Truth:** Seluruh 3.180 sampel berstatus uji sehingga supervised training langsung tidak dapat dilakukan tanpa pseudo-labeling.
2. **Sinyal Perubahan Global (SSIM & MAE):** Rata-rata `SSIM = 0.1694` (median `0.1456`, rentang `0.0064–0.8007`), dengan rata-rata MAE RGB (`R=55.61`, `G=52.40`, `B=51.32` pada skala 0–255).
3. **Distribusi SSIM Unimodal:** Uji KDE menunjukkan 1 puncak dominan di `0.1023` (Sarle BC `0.5042 < 0.555`) — **tidak bimodal**.
4. **Misalignment Spasial Tinggi:** Dari 300 sampel uji `ORB+RANSAC`, **229/300 (76.3%) berstatus `unregistered`**, dan pasangan `registered` memiliki median translasi **874.2 px** (min `152.8 px`). Artinya pasangan citra bukan tile satelit ter-registrasi piksel-per-piksel.
5. **Pergeseran Warna/Pencahayaan Kuat:** Rata-rata *KL divergence* 32-bin antara `pre` dan `post` mencapai `R=1.88`, `G=2.01`, `B=2.06`, mengindikasikan variasi iluminasi, asap, dan awan yang signifikan.

## 3. Rekomendasi Strategi & Estimasi Compute
- **Hardware Terdeteksi:** `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`101.98 GB VRAM`).
- **Strategi Utama:** Zero-shot CLIP/SigLIP (membandingkan embedding `post` dan `post - pre` terhadap prompt 6 kelas) + pseudo-labeling terkalibrasi menggunakan VLM (mis. Qwen2-VL/Qwen2.5-VL pada citra gabungan `[pre | post]`), diikuti *self-training* Siamese Network (`EfficientNet-B4` / `ConvNeXt`, fusion *concat* + *abs-diff*) menggunakan Stratified 5-Fold pada pseudo-label berkeyakinan tinggi.
- **Kandidat Drop:** `0` dari audit integritas (`df_integrity_issues` kosong), namun `~76.3%` pasangan `unregistered` pada ORB menandakan fusion *pixel-difference* mentah harus dilengkapi fitur semantik/global.
