# Klasifikasi Fenomena Bencana (— Task 2 Pre/Post Disaster Pairs)

Pipeline *end-to-end* untuk klasifikasi **3.180 pasangan citra udara/satelit** (`pre.jpg` dan `post.jpg`, `1024x1024` RGB) ke dalam 6 kelas bencana (**Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion**) dengan metrik evaluasi **Macro F1-Score**.

---

## Struktur Repositori

```text
.
├── AGENTS.md                           # Kontrak kerja kompetisi & aturan arsitektur terkunci
├── PIPELINE_HANDOFF.md                 # Status sesi terakhir + kode cepat 1-cell resume
├── README.md                           # Ringkasan arsitektur, temuan audit, dan progres
├── notebooks/
│   ├── fenomena_eda_modeling.ipynb     # Notebook Jupyter lengkap (50 cell: 21 Markdown + 29 Kode)
│   └── molab_notebook.py               # Snapshot source code marimo dari Molab
├── reports/
│   ├── PROGRESS_AND_RESUME_GUIDE.md    # Laporan progres lengkap & roadmap menuju 0.80+ Macro F1
│   ├── EXP_SUMMARY.md                  # Laporan temuan EDA & preprocessing (M1–K12)
│   ├── FORENSIC_AUDIT_REPORT.md        # Laporan matematis 4 pengujian forensik (M20–K20)
│   ├── model_hunting.md                # Benchmark RemoteCLIP vs SkyCLIP vs DINOv2 vs SigLIP
│   ├── training_log.md                 # Log training
│   ├── _meta.json                      # Metadata dataset & konfigurasi EDA
│   ├── molab_artifacts_manifest.json   # Manifest file di server Molab
│   └── cells_backup.json               # Backup JSON seluruh 50 cell
├── artifacts_synced/                   # Seluruh .parquet, .npy (<20MB), .csv, .md, .png (siap resume)
└── artifacts_weights_local/            # (Di-ignore Git) Checkpoint .pth & fenomena_resume_pack.tar.gz
```

---

## Ringkasan Progres Terkini

1. **EDA (`M1–K12`):** `3.180` pasang `1024x1024` RGB utuh (`HAS_LABELS = False`). `SSIM` unimodal (`mean=0.1694`, `median=0.1456`), `76.33%` pasangan `unregistered` pada `ORB+RANSAC` (median geser `874.2 px`), dan *KL divergence* warna antarwaktu tinggi (`R=1.88, G=2.01, B=2.06`).
2. **Model Hunting (`M14–K14b`):** **TOP-2:** `RemoteCLIP` (`sil=0.02311`, `ent=1.7533`) & `SkyCLIP` (`sil=0.02151`, `ent=1.7675`). Seluruh embedding `3.180 x 768` (`pre, post, diff, diff_hflip, sym`) tersimpan di `artifacts_synced/`.
3. **Audit Forensik Matematis (`M20–K20`):** Terbukti arah kronologis `pre ↔ post` tersebar hampir simetris (`Z_dir` mean `0.0000`, std `0.5055`, $\Delta\text{BIC}=-36.83$; `34.65%` `RemoteCLIP` & `33.14%` `SkyCLIP` negatif; **`556` pasangan / `17.48%`** konsensus keduanya negatif). Disahkan arsitektur wajib: **Symmetric Temporal-Swap Invariant Fusion + Bidirectional Swap-TTA**.
4. **Multi-Teacher De-Biased V2 (`M19–K19b`):** `Qwen2.5-VL-32B-Instruct` (`448x448` + 2x HFlip TTA + targeted re-pass 556 sampel `[post | pre]`) + `Symmetric CLIP Teacher` menurunkan bias `Fire` dari **`61.57%` → `16.60%`**. Split `folds_skf5.npy` terkunci.
5. **Skrining Fold 0 (`M21–K21c`):**
   - `Iter 0 (Base 384 RGB)`: `f0_logloss=1.1311`, `f0_raw_f1=0.5868`, `f0_cal_f1=0.5603` (`BASELINE`)
   - `Iter 1 (LAB CLAHE 384)`: `f0_logloss=1.1381`, `f0_raw_f1=0.6024`, `f0_cal_f1=0.5999` (`ROLLBACK` — `+0.0396` Calibrated F1, `LogLoss` selisih tipis `+0.007`)
   - `Iter 2 (D4+Cloud Aug 384)`: `f0_logloss=1.1745`, `f0_raw_f1=0.5756`, `f0_cal_f1=0.5891` (`ROLLBACK`)
   - `Iter 3 (448x448 RGB, parsial ep 8/15)`: sempat mencapai `val_ll=1.1160` (ep 6) dan `raw_f1=0.6143` (ep 7) sebelum dihentikan untuk backup.

---

## Panduan Lanjut Sesi Berikutnya

Lihat laporan detail di [reports/PROGRESS_AND_RESUME_GUIDE.md](reports/PROGRESS_AND_RESUME_GUIDE.md) dan kode cepat 1-cell di [PIPELINE_HANDOFF.md](PIPELINE_HANDOFF.md).
