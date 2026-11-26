# PIPELINE HANDOFF — Kompetisi Klasifikasi Bencana (Fenomena)

Dokumen ini menyimpan status lengkap sesi kerja, backup notebook Molab, hasil EDA & modeling sementara, serta langkah persis untuk melanjutkan eksekusi pada sesi berikutnya.

---

## 1. Status Progress Keseluruhan

| Fase | Cell | Status | Ringkasan Hasil |
| :--- | :--- | :--- | :--- |
| **Fase 1: EDA & Preprocessing** | `M1_setup` – `K12_summary` | **SELESAI (100%)** | `3.180` pasangan valid (`1024x1024` RGB), `HAS_LABELS=False`, 10 artefak tersimpan di `/marimo/Fenomena/`. |
| **Fase 2B.1: Instalasi Library** | `M13_install` – `K13_install` | **SELESAI** | `open_clip 4.2.2`, `timm 1.0.31`, `transformers 5.20.1`, `torch 2.11.0+cu130` (`9.42s`). GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`101.98 GB VRAM`). |
| **Fase 2B.2: Model Hunting** | `M14_hunting`, `K14a_hunting`, `K14b_extract` | **SELESAI** | **TOP-2:** `RemoteCLIP` (`sil=0.02311`, `db=4.5010`, `ent=1.7533`) & `SkyCLIP` (`sil=0.02151`, `db=4.4863`, `ent=1.7675`). Embedding penuh `3.180 x 768` (+ `hflip` untuk `RemoteCLIP`) tersimpan. |
| **Fase 2B.3: VLM Pseudo-Labeling** | `M15_pseudo`, `K15a_vlm_load`, `K15b_vlm_infer` | **SELESAI** | `Qwen/Qwen2.5-VL-7B-Instruct` (`235.3s`, `parse_ok=100%`, `conf>=0.7: 3180/3180`). **RED FLAG:** `Fire` = `61.57%`. Tersimpan di `/marimo/Fenomena/pseudo_labels.parquet`. |
| **Fase 2B.4: Arsitektur & Dataset** | `M16_model` – `K17_dataset` | **SELESAI** | `SiameseDualFusion` (`efficientnet_b4`, `34.4M` params) & `SiameseConvNeXt` (`convnext_small.fb_in22k_ft_in1k`, `52.6M` params) teruji. `IMG_CACHE_384` + Stratified 5-Fold (`636` sampel/fold) siap. |
| **Fase 2B.5: Training EffNet-B4** | `M18_train`, `K18a_train_effb4` | **PARSIAL (Fold 0–1 selesai, Fold 2 terhenti di ep 4)** | Fold 0 (`best_ep=26, val_macro_f1=0.4752, 533.3s`), Fold 1 (`best_ep=4, val_macro_f1=0.3997, 195.7s`), Fold 2 (`ep=4, val_f1=0.4177` -> `MarimoInterrupt` di batas waktu ~13.5 menit). Checkpoint `ckpt_effb4_fold0.pth`, `fold1.pth`, `fold2.pth` sudah ada di `/marimo/Fenomena/`. |
| **Fase 2B.6–2B.9: Lanjutan** | `K18b`, `M19–K22` | **BELUM DIJALANKAN** | Lanjutkan besok mulai Fold 2–4 `effb4` → `convnext` → `M19–K22`. |

---

## 2. Backup Lokal yang Tersimpan

Seluruh snapshot notebook dan laporan dari Molab telah disimpan rapi di `notebooks/` dan `reports/`:
- [notebooks/fenomena_eda_modeling.ipynb](notebooks/fenomena_eda_modeling.ipynb) — Notebook Jupyter lengkap (`41` cell: `18` Markdown + `23` Kode).
- [notebooks/molab_notebook.py](notebooks/molab_notebook.py) — File marimo notebook penuh dari `/marimo/notebook.py`.
- [reports/cells_backup.json](reports/cells_backup.json) — Dump JSON seluruh 41 cell (`id`, `name`, `status`, `code`).
- [reports/EXP_SUMMARY.md](reports/EXP_SUMMARY.md) — Ringkasan hasil EDA `M1–K12`.
- [reports/_meta.json](reports/_meta.json) — Metadata EDA terstruktur.
- [reports/model_hunting.md](reports/model_hunting.md) — Tabel benchmark 4 kandidat model & status embedding.
- [reports/training_log.md](reports/training_log.md) — Header log training.
- [reports/molab_artifacts_manifest.json](reports/molab_artifacts_manifest.json) — Daftar seluruh 24 file di `/marimo/Fenomena/` beserta ukuran byte.

---

## 3. Ringkasan Data & Temuan Kunci untuk Sesi Besok

### Temuan EDA (`M1–K12`)
- **Dataset:** `3.180` Id valid di `/marimo/dataset/Fenomena/{Id}/{pre,post}.jpg`, semua `1024x1024` RGB, `0` duplikat MD5, `0` corrupt.
- **Label:** `HAS_LABELS = False` (`sample_submission.csv` seluruhnya placeholder `'Earthquake'`).
- **T1 (Misalignment):** `229/300` (`76.33%`) sampel uji `ORB+RANSAC` berstatus `unregistered`; `71` sampel `registered` bergeser median `874.2 px` (`min 152.8 px`). Solusi: dual fusion `Concat + AbsDiff ChannelAttention` + augmentasi geometrik identik.
- **T2 (SSIM & Class Weights):** SSIM unimodal (`mean=0.1694`, `median=0.1456`, puncak KDE `0.1023`, Sarle BC `0.5042 < 0.555`). Wajib `class_weights` (`1/freq`, ternormalisasi) untuk **Macro F1**.
- **Color Shift:** KL divergence per pasangan tinggi (`R=1.88`, `G=2.01`, `B=2.06`) → `ColorJitter` independen pada `pre` dan `post`.

### Distribusi Pseudo-Label VLM (`K15b`, `Qwen2.5-VL-7B-Instruct`)
- `parse_ok = 100.00%` (`3.180/3.180`), `conf >= 0.7 = 100.00%` (min `0.80`, max `0.95`, `>=0.9` ada `370` sampel).
- **Sebaran kelas (`RED FLAG >60%` pada `Fire`):**
  - `Fire`: `1.958` (`61.57%`), purity KMeans `0.2007`
  - `Earthquake`: `331` (`10.41%`), purity `0.2568`
  - `Volcano`: `301` (`9.47%`), purity `0.2392`
  - `Explosion`: `280` (`8.81%`), purity `0.2893`
  - `Tsunami`: `196` (`6.16%`), purity `0.2041`
  - `Hurricane`: `114` (`3.58%`), purity `0.2632`

---

## 4. Artefak yang Sudah Tersimpan di Server Molab (`/marimo/Fenomena/`)
- Parquet EDA: `df_inventory.parquet`, `df_change.parquet`, `df_registration.parquet`, `color_stats.parquet`
- Visual & Laporan EDA: `sample_grid.png`, `diff_thumb_1..3.png`, `_meta.json`, `EXP_SUMMARY.md`
- Model Hunting: `model_hunting.md`, `emb_RemoteCLIP_{pre,post,diff,diff_hflip}.npy`, `emb_SkyCLIP_{pre,post,diff}.npy`
- Pseudo-Label: `pseudo_labels.parquet`, `_vlm_partial.jsonl`
- Checkpoints `effb4` sementara: `ckpt_effb4_fold0.pth` (`val_f1=0.4752`), `ckpt_effb4_fold1.pth` (`val_f1=0.3997`), `ckpt_effb4_fold2.pth` (parsial ep 4, `val_f1=0.4177` — jalankan ulang Fold 2–4 dengan lengkap).

---

## 5. Panduan Lanjutan Besok (Anti-Timeout & Idempoten)

1. **Koneksi & Cek Kernel:**
   - Minta URL/token Molab terbaru jika sandbox restart (atau gunakan URL/token yang aktif).
   - Cek apakah kernel masih hidup atau fresh:
     - Jika kernel fresh (RAM kosong), jalankan ulang `K13` (bila paket belum terpasang di container), lalu jalankan cell ringkas untuk memuat `df_inv_clean`, `df_pseudo`, `df_train_folds`, `IMG_CACHE_384`, `GPU_PRE_U8`, `GPU_POST_U8` dari file parquet yang **sudah ada** di `/marimo/Fenomena/` (JANGAN ulangi `K14a/K14b` dan `K15a/K15b` karena file `.npy` dan `pseudo_labels.parquet` sudah tersimpan di disk!).
2. **Perbaikan Wajib pada `K18a` & `K18b` (agar < 5 menit/cell dan tidak kena batas eksekusi ~13.5 menit):**
   - Pada `K18a`, Fold 0 memakan `533s` (~`17.7s/epoch`) karena `batch_size=32` dan `F.grid_sample` rotasi di dalam loop. Untuk mempercepat + mencegah timeout:
     - Simpan probabilitas val per fold langsung ke disk (`oof_{tag}_fold{k}.npy`) setiap fold selesai, dan tambahkan logika *skip fold* jika `ckpt_{tag}_fold{k}.pth` dan `oof_{tag}_fold{k}.npy` sudah selesai penuh.
     - Pecah training 5-fold menjadi maksimal **2 fold per cell** (mis. `K18a_1`: Fold 0–1 sudah ada / evaluasi ulang dari `ckpt_effb4_fold0..1.pth` → `oof_effb4_fold0..1.npy`; `K18a_2`: Fold 2–3; `K18a_3`: Fold 4 + gabung `oof_effb4.npy`), atau tingkatkan `batch_size=64` (VRAM 102 GB sangat lega!) jika diizinkan, serta tetap cetak progres per epoch (`flush=True`).
3. **Urutan Eksekusi Tersisa:**
   - **Selesaikan `K18a`** (`effb4` Fold 2, 3, 4 + rekonstruksi `oof_effb4_fold0..1` dari checkpoint yang sudah tersimpan → `oof_effb4.npy`).
   - **Jalankan `K18b`** (`SiameseConvNeXt` Fold 0–4, pecah 2–3 sub-cell agar < 5 menit per cell → `oof_convnext.npy`).
   - **Jalankan `M19–K19`** (Inferensi + TTA horizontal flip pada 3.180 sampel → `probs_effb4.npy`, `probs_convnext.npy`).
   - **Jalankan `M20–K20`** (Self-training iterasi 1: buat `pseudo_labels_v2.parquet`, fine-tune 15 epoch `lr=1e-4` dari checkpoint iterasi 1 — pecah per model/fold bila perlu — evaluasi delta OOF Macro F1, revert jika turun, regenerasi `probs_effb4_v2.npy` & `probs_convnext_v2.npy`).
   - **Jalankan `M21–K21`** (Latih `LogisticRegression(class_weight='balanced', max_iter=1000)` pada `emb_RemoteCLIP_diff.npy` untuk sampel `conf>=0.9` [`370` sampel], TTA average dengan `emb_RemoteCLIP_diff_hflip.npy`, grid search bobot `w1, w2, w3` pada `300` sampel confidence tertinggi, validasi & simpan `/marimo/Fenomena/submission_final.csv`).
   - **Jalankan `M22–K22`** (Tulis `/marimo/Fenomena/MODELING_SUMMARY.md`, kirim toast akhir, dan berikan laporan akhir A–G).
