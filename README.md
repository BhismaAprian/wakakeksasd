# Klasifikasi Fenomena Bencana (Datavidia — Pre/Post Disaster Pairs)

Pipeline *end-to-end* untuk klasifikasi 6 kelas fenomena bencana (**Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion**) dari `3.180` pasang citra `pre.jpg` dan `post.jpg` dengan metrik evaluasi **Macro F1-Score**.

---

## Struktur Repositori

```text
.
├── AGENTS.md                           # Kontrak kerja kompetisi, aturan validasi & protokol eksperimen
├── PIPELINE_HANDOFF.md                 # Status detail sesi terakhir & panduan lanjutan (anti-timeout)
├── README.md                           # Ringkasan arsitektur, temuan EDA, dan progres pipeline
├── notebooks/
│   ├── fenomena_eda_modeling.ipynb     # Notebook Jupyter (41 cell: 18 Markdown + 23 Kode, M1–K18a)
│   └── molab_notebook.py               # Snapshot source code marimo asli dari Molab
└── reports/
    ├── EXP_SUMMARY.md                  # Laporan temuan EDA & preprocessing (M1–K12)
    ├── model_hunting.md                # Hasil benchmark backbone satelit (RemoteCLIP, SkyCLIP, DINOv2, SigLIP)
    ├── training_log.md                 # Log training Siamese Dual Fusion
    ├── _meta.json                      # Metadata statistik dataset & konfigurasi artefak
    ├── molab_artifacts_manifest.json   # Daftar 24 file artefak yang tersimpan di server Molab
    └── cells_backup.json               # Backup JSON seluruh 41 cell notebook
```

---

## Ringkasan Progres Pipeline

| Tahap | Cakupan Cell | Status | Ringkasan |
| :--- | :--- | :--- | :--- |
| **1. EDA & Preprocessing** | `M1` – `K12` | Selesai (`100%`) | `3.180` pasang valid (`1024x1024` RGB, `0` corrupt, `0` duplikat MD5). Terkonfirmasi **`HAS_LABELS = False`** (seluruh sampel berstatus uji). |
| **2. Instalasi Library** | `M13` – `K13` | Selesai | `open_clip 4.2.2`, `timm 1.0.31`, `transformers 5.20.1`, `torch 2.11.0+cu130` di GPU `NVIDIA RTX PRO 6000 Blackwell` (`101.98 GB VRAM`). |
| **3. Model Hunting** | `M14` – `K14b` | Selesai | **TOP-2:** `RemoteCLIP` (`sil=0.02311`, `ent=1.7533`) & `SkyCLIP` (`sil=0.02151`, `ent=1.7675`). Embedding `3.180 x 768` tersimpan di Molab. |
| **4. VLM Pseudo-Labeling** | `M15` – `K15b` | Selesai | `Qwen/Qwen2.5-VL-7B-Instruct` (`235.3s`, `parse_ok=100%`, `conf>=0.7: 3180/3180`). **RED FLAG:** `Fire` = `61.57%` → ditangani dengan *class weights*. |
| **5. Model & Dataset** | `M16` – `K17` | Selesai | `SiameseDualFusion` (`efficientnet_b4`, `34.4M` params) & `SiameseConvNeXt` (`convnext_small.fb_in22k_ft_in1k`, `52.6M` params), `StratifiedKFold(5)`. |
| **6. Training EffNet-B4** | `M18` – `K18a` | Parsial (`Fold 0–1`) | Fold 0 (`val_macro_f1=0.4752`), Fold 1 (`val_macro_f1=0.3997`), Fold 2 terhenti di ep 4 (`0.4177`, checkpoint tersimpan). |
| **7. Lanjutan** | `K18b`, `M19` – `K22` | Pending | Selesaikan Fold 2–4 `effb4` → `convnext` → TTA → Self-Training → Ensemble + `submission_final.csv`. |

---

## Temuan Kunci EDA yang Membentuk Arsitektur

1. **Ketiadaan Label Ground-Truth:** Seluruh `3.180` `Id` berstatus uji (`sample_submission.csv` berisi placeholder `'Earthquake'`), sehingga digunakan *VLM pseudo-labeling* (`Qwen2.5-VL-7B`) + *student Siamese training*.
2. **Misalignment Spasial Ekstrem (`T1`):** Pada uji `ORB+RANSAC` (`300` sampel), **`76.33%` berstatus `unregistered`** dan pasangan `registered` bergeser median `874.2 px`. Oleh karena itu, arsitektur menggunakan jalur ganda **Concatenation + AbsDiff Channel Attention** pada fitur semantik, bukan *pixel subtraction* mentah.
3. **Distribusi SSIM Unimodal & Imbalance (`T2`):** `SSIM` memiliki `mean=0.1694`, `median=0.1456`, puncak KDE tunggal di `0.1023` (Sarle BC `0.5042 < 0.555`), sehingga diterapkan `class_weights` (`1/freq`) untuk mengoptimalkan **Macro F1**.
4. **Pergeseran Pencahayaan (`Color Shift`):** *KL divergence* 32-bin per pasangan cukup besar (`R=1.88`, `G=2.01`, `B=2.06`), memvalidasi `ColorJitter` independen pada pasangan `pre` dan `post`.

---

## Cara Melanjutkan Sesi Berikutnya

Lihat panduan lengkap dan daftar file artefak di [PIPELINE_HANDOFF.md](PIPELINE_HANDOFF.md).
