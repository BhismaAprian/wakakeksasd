# Model Hunting Report — Kompetisi Fenomena

## Ringkasan Benchmark (300 Sampel, Seed=42)
- **TOP-1:** `RemoteCLIP`
- **TOP-2:** `SkyCLIP`
- **Kriteria:** Silhouette score KMeans `k=6` (lebih tinggi lebih baik), Davies-Bouldin score (lebih rendah lebih baik), serta entropy argmax prompt teks 6 kelas (lebih rendah & tanpa kolaps kelas lebih baik). `Prithvi` dilewati karena input hanya JPEG RGB 3-kanal.

| model | silhouette | db_score | entropy | max_class_share | dim | ok_flag | sec | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RemoteCLIP | 0.02311 | 4.501 | 1.7533 | 0.26 | 768 | True | 21.2 | argmax_counts=[37, 45, 37, 45, 78, 58] |
| SkyCLIP | 0.02151 | 4.4863 | 1.7675 | 0.22 | 768 | True | 32.2 | argmax_counts=[33, 66, 49, 59, 52, 41] |
| DINOv2-large | 0.01905 | 5.0552 | 1.7791 | 0.213 | 1024 | True | 8.9 | vision_only_kmeans_counts=[38, 47, 55, 64, 48, 48] |
| SigLIP-large | nan | nan | nan | nan | 0 | False | 0.4 | ERR: 
SiglipTokenizer requires the SentencePiece library but it was not found in your environment. Check out the instructions on the
installation |

## Embedding Tersimpan (3.180 Sampel)
| model | shape | hflip_saved | cached |
| --- | --- | --- | --- |
| RemoteCLIP | (3180, 768) | True | True |
| SkyCLIP | (3180, 768) | False | True |
