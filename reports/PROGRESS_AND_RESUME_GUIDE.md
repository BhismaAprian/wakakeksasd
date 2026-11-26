# PROGRESS & RESUME GUIDE — Task 2 (Fenomena Pre/Post Disaster Pairs)

Dokumen ini merangkum seluruh pencapaian riset, temuan matematis, hasil eksperimen Fold 0, serta rencana langkah demi langkah untuk melanjutkan kompetisi menuju **Macro F1 `0.80+`** pada sesi berikutnya.

---

## 1. Ringkasan Eksekutif & Status Setiap Tahap

| Tahap | Cakupan Cell Molab | Status | Temuan & Hasil Kunci |
| :--- | :--- | :--- | :--- |
| **1. EDA & Preprocessing** | `M1_setup` – `K12_summary` | **SELESAI** | `3.180` pasang `1024x1024` RGB valid (`0` corrupt, `0` missing, `0` duplikat MD5). `HAS_LABELS = False`. `SSIM` unimodal (`mean=0.1694`, `median=0.1456`, puncak KDE `0.1023`, Sarle BC `0.5042`). `76.33%` (`229/300`) sampel `ORB+RANSAC` berstatus `unregistered` (median geser `874.2 px`). *KL divergence* warna antarwaktu tinggi (`R=1.88`, `G=2.01`, `B=2.06`). |
| **2. Model Hunting** | `M14_hunting` – `K14b_extract` | **SELESAI** | **TOP-2:** `RemoteCLIP` (`sil=0.02311`, `db=4.5010`, `ent=1.7533`) & `SkyCLIP` (`sil=0.02151`, `db=4.4863`, `ent=1.7675`), mengungguli `DINOv2-large` (`sil=0.01905`). Embedding `3.180 x 768` (`pre, post, diff, diff_hflip, sym`) tersimpan. |
| **3. VLM 7B Awal** | `M15_pseudo` – `K15b_vlm_infer` | **SELESAI (DIGANTI V2)** | `Qwen2.5-VL-7B-Instruct` (`280x280`, `235.3s`) mengalami **RED FLAG bias struktural**: `Fire` mendominasi **`61.57%`** (`1.958/3.180`), sedangkan `Hurricane` hanya `3.58%` (`114`), `Tsunami` `6.16%` (`196`), `Explosion` `8.81%` (`280`), `Volcano` `9.47%` (`301`), `Earthquake` `10.41%` (`331`). |
| **4. Audit Forensik Matematis** | `M20_forensic` – `K20_forensic` | **SELESAI** | Terbukti secara matematis arah waktu visual terdistribusi hampir simetris (`Z_dir` mean `0.0000`, std `0.5055`, $\Delta\text{BIC}=-36.83$). `RemoteCLIP` (`34.65%`) & `SkyCLIP` (`33.14%`) mendeteksi $\Delta S_{\text{sem}} < 0$ ($\rho = 0.4359, p = 1.28\times 10^{-147}$; $\kappa = 0.2676$; **`556` pasangan / `17.48%`** konsensus keduanya negatif). `51.16%` memiliki `post` lebih buram (`d_log_lap < 0`), `52.48%` memiliki $R_{\text{HF}} < 0$, `43.68%` memiliki `post` lebih hijau ($\Delta\text{ExG} > 0$), `~72%` memiliki anomali *scanline* (`>50`) di kedua sisi, dan `40.09%` beda *black padding* `> 2%`. Uji $\chi^2$ membuktikan VLM 7B buta arah (`p=0.268` untuk `is_inverted`, `p=0.880` untuk `is_sem_both_neg`). **Keputusan arsitektur:** **Symmetric Temporal-Swap Invariant Fusion + Bidirectional Swap-TTA**. |
| **5. De-Biased Multi-Teacher V2** | `M19_pseudo_v2` – `K19b_vlm32b_teacher` | **SELESAI** | `Qwen2.5-VL-32B-Instruct` (`448x448` per tile, `896x448` side-by-side + 2x HFlip TTA) + *targeted symmetric `[post | pre]` re-pass* pada `556` sampel `is_sem_both_neg` + *column-mean de-biasing* ($\alpha=0.92$) + fusi dengan `Symmetric CLIP Teacher` (`0.72 * VLM32B + 0.28 * CLIP_Sym`). Bias `Fire` turun drastis dari `61.57%` → **`16.60%`**. `folds_skf5.npy` terkunci. |
| **6. Skrining Fold 0 (Stage 1)** | `M21_fold0_screening` – `K21c_fold0_iter3` | **SELESAI (Iter 0–2) + Parsial Iter 3** | Uji satu-per-satu dengan `SymmetricSiameseNet` (`efficientnet_b4`) pada `folds_skf5[0]` (`consensus_soft_v1.npy`). Lihat tabel di bawah. |

---

## 2. Distribusi Multi-Teacher V2 (`pseudo_labels_v2.parquet`)

`VLM-32B TTA agree = 78.18%` | `VLM-CLIP agree = 34.94%` (`consensus_soft_v1.npy` + `folds_skf5.npy` terkunci):

| Kelas | V1 (`Qwen-7B`) | `VLM32B_Sym_Deb` (%) | `CLIP_Sym` (%) | `Consensus_V2` (n) | `Consensus_V2` (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Earthquake** | `10.41%` | `13.84%` | `17.77%` | `459` | **`14.43%`** |
| **Fire** | `61.57%` | `16.23%` | `19.47%` | `528` | **`16.60%`** |
| **Hurricane** | `3.58%` | `25.03%` | `15.63%` | `788` | **`24.78%`** |
| **Tsunami** | `6.16%` | `5.91%` | `15.94%` | `196` | **`6.16%`** |
| **Volcano** | `9.47%` | `20.06%` | `16.16%` | `634` | **`19.94%`** |
| **Explosion** | `8.81%` | `18.93%` | `15.03%` | `575` | **`18.08%`** |

---

## 3. Tabel Eksperimen Fold 0 (`artifacts_synced/experiment_tracker.csv`)

| Iterasi | Resolusi | LAB CLAHE | Augmentasi | `f0_logloss` | `f0_raw_f1` | `f0_cal_f1` | Keputusan | Durasi |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **Iter 0 (Base 384 RGB)** | `384` | `False` | `basic` | **`1.1311`** | `0.5868` | `0.5603` | `BASELINE` | `287.3s` |
| **Iter 1 (LAB CLAHE 384)** | `384` | `True` | `basic` | `1.1381` | **`0.6024`** | **`0.5999`** | `ROLLBACK` | `287.0s` |
| **Iter 2 (D4+Cloud Aug 384)** | `384` | `False` | `d4_full` | `1.1745` | `0.5756` | `0.5891` | `ROLLBACK` | `289.2s` |
| *Iter 3 (High-Res 448 RGB, parsial ep 8/15)* | `448` | `False` | `basic` | *(ep6: `1.1160`)* | *(ep7: `0.6143`)* | — | *Dihentikan saat backup* | — |

**Catatan Analisis:**
- **Iter 1 (`LAB CLAHE`)** memberi lonjakan `+0.0396` pada `OOS Calibrated Macro F1` (`0.5603 → 0.5999`), hanya tertahan kenaikan tipis `LogLoss` `+0.007` (`1.1311 → 1.1381`).
- **Iter 2 (`d4_full`)** terlalu agresif (`RandomResizedCrop` + `RandomErasing` + `Gamma` + `MixUp` sekaligus dalam 15 epoch) sehingga underfit (`LogLoss 1.1745`).
- **Iter 3 (`448x448`)** sempat mencapai `val_ll = 1.1160` di epoch 6 dan `raw_f1 = 0.6143` di epoch 7 sebelum dihentikan untuk proses backup — indikasi kuat resolusi lebih tinggi membantu pemisahan fitur lokal.

---

## 4. Analisis Bottleneck & Rencana Eksak Menuju `0.80+` Macro F1

1. **Langkah 1 — Jangkar Public LB Nyata:**
   Buat *baseline submission* dari konsensus `Qwen2.5-VL-32B + Symmetric CLIP Teacher` (`pseudo_labels_v2.parquet`) terlebih dahulu (dengan persetujuan user) untuk mengunci skor **Public LB nyata** dan memvalidasi apakah proporsi `Tsunami` (`6.16%`) perlu kalibrasi prior.
2. **Langkah 2 — Peningkatan Fitur & Resolusi:**
   - Gunakan `LAB CLAHE` pada kanal `L*` saja (**tanpa** `RandomErasing`/`Gamma` berlebihan), dan naikkan resolusi input ke **`512x512`**.
   - Ekstrak **Multi-Crop Foundation Embeddings** (`1 global + 4 kuadran` dari citra `512x512`) menggunakan `RemoteCLIP`, `SkyCLIP`, `DINOv2-Large`, dan `SigLIP-SO400M` (dalam format simetris `[0.5*(e_pre+e_post), |e_post-e_pre|]`).
3. **Langkah 3 — Stage 2 (5-Fold Lintas 5 Keluarga Model Heterogen) + Stage 3 Kalibrasi:**
   - Latih 5 keluarga model pada `folds_skf5.npy` menggunakan skrip `nohup python3 -u ...`:
     1. `ConvNeXt-Base` (`512px`, `SymmetricSiamese`)
     2. `Swin-Base` / `EVA-02-Base` (`384/448px`, `SymmetricSiamese`)
     3. `EfficientNet-B4` (`512px`, `SymmetricSiamese`)
     4. `Multi-Crop Foundation Residual MLP` (`RemoteCLIP + SkyCLIP + DINOv2-Large + SigLIP-SO400M`)
     5. `Forensic Tabular CatBoost + LightGBM` (fitur `forensic_math_audit.parquet` invarian urutan + SVD/prompt similarities)
   - Terapkan **Bidirectional Swap + HFlip (4x TTA)**, lalu gabungkan ke-5 keluarga model bersama probabilitas `Qwen2.5-VL-32B`, lakukan 1 putaran **strict-OOF self-distillation**, dan optimalkan **6 Class Multipliers** secara **Nested-OOF** untuk memaksimalkan **Macro F1**.
