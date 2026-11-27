# PIPELINE HANDOFF — Task 2 (Fenomena Pre/Post Disaster Pairs)

> **Status Terkini:** **100% Pure-Vision Compliant (FAQ Task 2)** — Seluruh artefak & cell VLM/CLIP serta kandidat `v5/v6/v7` yang skornya rendah telah dibersihkan. Pipeline dikunci pada **Arsitektur View S (Event Structure `K=72`) / View D (`DINOv3 64×64` + `DINOv2 37×37` + Fisika Terarah)**.

---

## 1. Kepatuhan Aturan Panitia (FAQ Task 2)
- **0% VLM** (`Qwen`, `InternVL`, `Mistral`) & **0% CLIP/Text-Image** (`GeoRSCLIP`, `RemoteCLIP`, `SkyCLIP`, `SigLIP`).
- **0% Dataset Eksternal Berlabel** (`xBD`, `BRIGHT`, dll.).
- **0% Penyetelan Berbasis Public LB** & **0% Koreksi Label Per Sampel (`Id`)**.
- **`SEED = 42`** terkunci.

---

## 2. Artefak Tersimpan di Molab (`/marimo/Fenomena/`) & Lokal (`artifacts_synced/`)
| Komponen | Lokasi Molab | Lokasi Lokal | Keterangan |
| :--- | :--- | :--- | :--- |
| **Notebook Marimo & IPYNB** | `/marimo/notebook.py`, `fenomena_molab.ipynb` | `notebooks/molab_notebook.py`, `notebooks/fenomena_eda_modeling.ipynb` | 73 cell bersih (EDA, Pure-Vision Zoo, View S/D `K=72`) |
| **Pure-Vision 5-Crop Zoo** | `emb_DINOv3L_5crop_sym.npy`, `emb_SatMAEPP_5crop_sym.npy`, `emb_ScaleMAE_5crop_sym.npy`, `emb_DINOv2Giant_5crop_sym.npy`, `emb_IJEPA_5crop_sym.npy` | *(tetap di Molab, 52–78 MB/file)* | `(3180, 2*D)`: separuh `:D` = `View S`, separuh `D:` = `|e_A - e_B|` |
| **Spatial Change Heatmaps & Diff** | `dinov3_change_map_64x64.npy`, `dinov2_change_map_37x37.npy`, `emb_dinov3_1024_diff_only.npy`, `emb_dinov2_518_diff_only.npy` | *(tetap di Molab, 8.7–39 MB/file)* | Peta perubahan patch 2D `64×64` (`1024px`) & `37×37` (`518px`) |
| **Contrastive Backbone** | `ckpt_contrastive_cnxb512.pt` (`350 MB`) | *(tetap di Molab)* | `ConvNeXt-Base` 6 epoch `InfoNCE` (`512px LAB CLAHE`) |
| **Fisika & Geometri 2D** | `physical_features_sym.npy` `(3180, 78)`, `spatial_dir_physics.npy` `(3180, 21)` | `artifacts_synced/physical_features_sym.npy`, `artifacts_synced/spatial_dir_physics.npy` | Invarian urutan + 3 rezim sensor (`RGB`, `NIR`, `Grayscale`) |
| **View S `K=72` Cluster Cards** | `cluster_cards_k72.csv`, `cluster_ids_k72_views.npy`, `audit_cluster_cards_k72.png` (`+ p1..p3`) | `artifacts_synced/cluster_cards_k72.csv`, `artifacts_synced/cluster_ids_k72_views.npy`, `artifacts_synced/audit_cluster_cards_k72.png` | `54 RGB + 9 NIR + 9 Grayscale` (`min=6, med=40, max=144`) |

---

## 3. Rencana Lanjutan (Sesi Berikutnya)
1. **Inspeksi Visual & Pemetaan `CLUSTER_TO_CLASS` (`C00..C71`):**
   - Tinjau `artifacts_synced/cluster_cards_k72.csv` dan `artifacts_synced/audit_cluster_cards_k72.png` untuk menyusun dictionary eksplisit `CLUSTER_TO_CLASS` (100% keputusan per klaster *event*, 0 per sampel).
2. **Validasi `GroupKFold(5)` pada `cluster_ids_k72_views.npy`:**
   - Gunakan `GroupKFold(n_splits=5)` dengan `groups=cluster_ids_k72_views.npy` agar *tile* dari *event* yang sama tidak bocor antar *fold*.
3. **Pelatihan Student Pure-Vision + Ensembling:**
   - Latih `MLP-PLR`, `Forensic GBDT` (`CatBoost + LightGBM + XGBoost`), dan `ConvNeXt-Base` (dari `ckpt_contrastive_cnxb512.pt`), lakukan *logit stacking*, validasi 8 gerbang, dan ekspor kandidat submission baru sebelum diunggah ke Kaggle.
