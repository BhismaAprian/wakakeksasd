# EXP_SUMMARY.md — Catatan Eksperimen Kompetisi Fenomena

## Ringkasan Fase Aktif (`Pure-Vision View S / View D`)

| Tahap | Cell Molab | Deskripsi | Temuan / Hasil Utama |
| :--- | :--- | :--- | :--- |
| **EDA & Audit Forensik** | `M1` – `pHFh` | Inventaris `3.180` pasang citra (`1024×1024`), registrasi, audit inversi kronologis | `HAS_LABELS=False`, `76.3%` unregistered, inversi `pre ↔ post` & anomali sensor |
| **Pure-Vision Registry** | `M14` (`XBxa`), `K15` (`hhsi`) | Ekstraksi *5-Crop* simetris `DINOv3-L`, `SatMAE++`, `ScaleMAE`, `DINOv2-Giant`, `I-JEPA` | `0` VLM/CLIP; seluruh embedding `NaN=0` |
| **Audit `v5` & Peta 2D** | `mZME`, `LEme`, `uulT`–`ozxA`, `Ghly` | Audit rasio norma `||mean||/||diff||` (`ScaleMAE 4.756x`), `DINOv3 64×64`, `DINOv2 37×37`, `spatial_dir_physics.npy` | Terbukti vektor statis mendominasi klastering lama; diekstrak `21` fitur geometri 2D & fisika 3 rezim sensor |
| **Diagnostik `View S` & `K=72`** | `M20_claude_view_sd` (`fXiy`), `K20_claude_view_sd` (`BNff`) | Uji komunitas kNN (`k=15`, `SEED=42`), fusi 5 backbone per rezim sensor (`54 RGB + 9 NIR + 9 Gray`), *Cluster Cards* `View D` | Terkonfirmasi struktur *event* rapat (`50` komunitas med `48–52`, seed `ARI=0.7052`); `K=72` klaster (`med=40`) tersimpan di `cluster_cards_k72.csv` & `audit_cluster_cards_k72.png` |

## Rencana Selanjutnya
1. Tinjau `artifacts_synced/cluster_cards_k72.csv` & `artifacts_synced/audit_cluster_cards_k72.png`.
2. Susun dictionary eksplisit `CLUSTER_TO_CLASS` untuk `C00..C71`.
3. Latih student *pure-vision* menggunakan `GroupKFold(5)` berbasis `cluster_ids_k72_views.npy`.
