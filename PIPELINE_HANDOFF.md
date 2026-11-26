# PIPELINE HANDOFF — Task 2 (Fenomena Pre/Post Disaster Pairs)

Dokumen handoff untuk langsung melanjutkan (*resume*) riset & training besok di Molab, Colab, atau Kaggle tanpa mengulang EDA, ekstraksi embedding, maupun VLM-32B pseudo-labeling.

---

## 1. Status Akhir Sesi Hari Ini
- **Selesai 100% & Tersinkronisasi ke `artifacts_synced/`:**
  - EDA (`M1–K12`): `df_inventory.parquet`, `df_change.parquet`, `df_registration.parquet`, `color_stats.parquet`, `sample_grid.png`, `_meta.json`, `EXP_SUMMARY.md`.
  - Model Hunting (`M14–K14b`): `emb_RemoteCLIP_{pre,post,diff,diff_hflip,sym}.npy`, `emb_SkyCLIP_{pre,post,diff,sym}.npy`, `model_hunting.md`.
  - Audit Forensik (`M20–K20`): `forensic_math_audit.parquet`, `forensic_inverted_top12.png`, `FORENSIC_AUDIT_REPORT.md`.
  - De-Biased Multi-Teacher V2 (`M19–K19b`): `clip_teacher_sym_probs.npy`, `vlm32b_raw_logits_{orig,flip}.npy`, `vlm32b_repass556_logits.npy`, `vlm32b_teacher_probs.npy`, `consensus_soft_v1.npy`, `pseudo_labels_v2.parquet`, `folds_skf5.npy`.
  - Skrining Fold 0 (`M21–K21c`): `experiment_tracker.csv` (`Iter 0–2` selesai; `Iter 3` `448x448` sempat hingga ep 8/15 dengan ep 6 `val_ll=1.1160`, ep 7 `raw_f1=0.6143`).
- **Tersimpan Lokal di `artifacts_weights_local/` (tidak di-push ke GitHub):**
  - `ckpt_effb4_fold0.pth`, `ckpt_effb4_fold1.pth`, `ckpt_effb4_fold2.pth` (baseline v1).
  - `ckpt_f0_iter0_base384.pth`, `ckpt_f0_iter1_lab384.pth`, `ckpt_f0_iter2_d4aug384.pth`, `ckpt_f0_iter3_res448.pth` (Stage 1 Fold 0).
  - `fenomena_resume_pack.tar.gz` — arsip seluruh `.parquet`, `.npy`, dan `.csv` siap ekstrak.

---

## 2. Kode Cepat 1-Cell untuk Resume Besok

Jika memulai di environment baru/fresh (atau setelah ekstrak `fenomena_resume_pack.tar.gz` / clone repo ini), jalankan **1 cell Python ini** untuk langsung memuat seluruh state riset tanpa mengulang proses apa pun:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Ubah ART_DIR ke "artifacts_synced" (jika di repo lokal) atau "/marimo/Fenomena" (jika di Molab)
ART_DIR = Path("artifacts_synced") if Path("artifacts_synced").is_dir() else Path("/marimo/Fenomena")

df_inv = pd.read_parquet(ART_DIR / "df_inventory.parquet")
df_change = pd.read_parquet(ART_DIR / "df_change.parquet")
df_forensic = pd.read_parquet(ART_DIR / "forensic_math_audit.parquet")
df_pseudo_v2 = pd.read_parquet(ART_DIR / "pseudo_labels_v2.parquet")
df_tracker = pd.read_csv(ART_DIR / "experiment_tracker.csv")

y_soft = np.load(ART_DIR / "consensus_soft_v1.npy")
folds = np.load(ART_DIR / "folds_skf5.npy")
p_vlm32b = np.load(ART_DIR / "vlm32b_teacher_probs.npy")
p_clip_sym = np.load(ART_DIR / "clip_teacher_sym_probs.npy")

emb_rc_sym = np.load(ART_DIR / "emb_RemoteCLIP_sym.npy")   # (3180, 1536)
emb_sc_sym = np.load(ART_DIR / "emb_SkyCLIP_sym.npy")      # (3180, 1536)

with open(ART_DIR / "_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print("Resume siap:", len(df_inv), "sampel | Folds:", np.bincount(folds).tolist())
print(df_tracker.to_string(index=False))
```

*(Opsional jika memakai arsip `.tar.gz` di server baru: `mkdir -p /marimo/Fenomena && tar -xzf fenomena_resume_pack.tar.gz -C /marimo/Fenomena`)*

---

## 3. Urutan Kerja Besok (Menuju `0.80+` Macro F1)
Lihat rincian teknis lengkap di [reports/PROGRESS_AND_RESUME_GUIDE.md](reports/PROGRESS_AND_RESUME_GUIDE.md):
1. Uji *baseline submission* `Qwen2.5-VL-32B + Symmetric CLIP Teacher` untuk mengunci skor **Public LB nyata**.
2. Terapkan **LAB CLAHE (kanal-L saja)** + resolusi **`512x512`** + ekstrak **Multi-Crop Foundation Embeddings** (`1 global + 4 kuadran` dari `RemoteCLIP`, `SkyCLIP`, `DINOv2-Large`, `SigLIP-SO400M`).
3. Latih **5-Fold penuh** 5 keluarga model heterogen (`ConvNeXt-Base 512`, `Swin/EVA-02 448`, `EfficientNet-B4 512`, `Multi-Crop Foundation MLP`, `Forensic Tabular CatBoost/LightGBM`) dengan `Symmetric Temporal-Swap Invariant Fusion` + `Bidirectional Swap-TTA`.
4. Gabungkan 5 keluarga + `Qwen2.5-VL-32B` → `strict-OOF self-distillation` → kalibrasi **6 Class Multipliers (Nested-OOF)**.
