# FORENSIC AUDIT REPORT — Pengujian Matematis Pre vs Post (3.180 Pasangan)

## 1. Uji Asimetri Vektor Semantik (`RemoteCLIP` & `SkyCLIP`)
- **Pasangan dengan $\Delta S_{\text{sem}} < 0$ (`pre` lebih rusak dibanding `post`):**
  - `RemoteCLIP`: **1102 / 3.180** (`34.65%`), mean $\Delta S_{\text{sem}} = 0.0075$ (median `0.0070`)
  - `SkyCLIP`: **1054 / 3.180** (`33.14%`), mean $\Delta S_{\text{sem}} = 0.0068$ (median `0.0059`)
  - **Keduanya $\Delta S_{\text{sem}} < 0$ (`is_sem_both_neg`):** **556 / 3.180** (`17.48%`) | Salah satu `< 0`: `1600` (`50.31%`)
- **Konsistensi Lintas Model:**
  - Korelasi peringkat Spearman $\rho$: **`0.4359`** ($p = 1.28e-147$)
  - Cohen's Kappa $\kappa$ (deteksi $\Delta S_{\text{sem}} < 0$): **`0.2676`** (kesepakatan mentah `67.17%`)
- **Antisymmetric Class Projection Ratio (`F_max - B_max`):**
  - `RemoteCLIP`: mean `0.0114`, fraksi `< 0`: `45.91%`
  - `SkyCLIP`: mean `0.0244`, fraksi `< 0`: `36.16%`

## 2. Forensik Fisika Citra, FFT (2D & 1D), & Geometri Batas
- **Ketajaman & Energi HF ($R_{\text{HF}}$):**
  - Mean $\Delta \ln(\sigma^2_{\nabla^2})$: `-0.0208` (fraksi `post < pre`: `51.16%`)
  - Mean $R_{\text{HF}}$: `-0.0314` (std `0.7157`, fraksi `< 0`: `52.48%`)
- **Artefak Scanline Periodik (1D FFT Vertikal, rasio `peak/median`):**
  - `scanline_pre`: median `3048.81`, P95 `69753.77`, max `1089470.84`
  - `scanline_post`: median `2300.61`, P95 `69712.99`, max `1086085.64`
  - Anomali scanline ekstrem (`ratio > 50`): `pre=2291`, `post=2287`
- **Vegetasi (`ExG`) & Piksel Gelap (`<45`):**
  - Mean $\Delta \text{ExG}$: `-0.0082` (fraksi $\Delta \text{ExG} > 0$ alias `post` lebih hijau dari `pre`: `43.68%`)
  - Mean $\Delta \text{dark}$: `0.0092` (fraksi $\Delta \text{dark} < 0$: `52.39%`)
- **Geometri Padding Hitam (*Warped Orthomosaic*):**
  - Pasangan dengan `pad_frac >= 0.5%`: `pre=2873`, `post=2856`, beda padding `|pre-post| > 2%`: `1275`
  - Mean `concavity` pada citra ber-padding (`>=0.5%`): `pre=0.0077`, `post=0.0082`

## 3. Uji Hipotesis Statistik & Dekomposisi GMM pada $Z_{\text{dir}}$
- **Statistik $Z_{\text{dir}}$:** mean `-0.0000`, std `0.5055`, min `-1.8046`, max `2.1401`
- **Uji Normalitas:** D'Agostino-Pearson $K^2 = 8.90$ ($p = 1.17e-02$), *Skewness* = `0.1011`, *Excess Kurtosis* = `0.1689`
- **Dekomposisi GMM (1-Komponen vs 2-Komponen):**
  - `BIC(1)` = `4701.46` vs `BIC(2)` = `4738.29` → **$\Delta \text{BIC} = -36.83$**
  - Komponen 1 (diduga terbalik): $\mu = -0.2675$, $\sigma = 0.3970$, bobot $= 0.5513$
  - Komponen 2 (normal): $\mu = 0.3287$, $\sigma = 0.4240$, bobot $= 0.4487$
  - **Jumlah pasangan berstatus `is_inverted` ($P_{\text{inv}} > 0.5 \land Z_{\text{dir}} < 0$):** **1624 / 3.180 (`51.07%`)**
  - Uji Mann-Whitney U (`Z_dir`): $U = 0.0$, $p = 0.00e+00$ | KS-test: $D = 1.0000$, $p = 0.00e+00$
  - Uji Mann-Whitney U (`SSIM`): $p = 8.40e-02$ (mean SSIM inverted=`0.1732` vs normal=`0.1654`)

## 4. Audit Silang terhadap `pseudo_labels.parquet` (VLM 7B)
- Chi-square `is_inverted` vs `vlm_label`: $\chi^2 = 6.41$ ($p = 2.68e-01$)
- Chi-square `is_sem_both_neg` (`556` sampel) vs `vlm_label`: $\chi^2 = 1.77$ ($p = 8.80e-01$)
- Distribusi `vlm_label` pada `is_sem_both_neg` vs normal:
```
vlm_label        Earthquake  Explosion  Fire  Hurricane  Tsunami  Volcano
is_sem_both_neg                                                          
False                   271        226  1616         94      167      250
True                     60         54   342         20       29       51
```

## 5. Top-15 Id Paling Terbalik (Berdasarkan $Z_{\text{dir}}$ Terendah)
```
          id   Z_dir  p_inverted_gmm  dS_sem_RemoteCLIP  dS_sem_SkyCLIP  acpr_diff_RemoteCLIP    R_HF   d_exg  d_dark  scanline_pre  scanline_post  pad_frac_pre  pad_frac_post   ssim  vlm_label
c7b66aebf0a7 -1.8046          0.9956            -0.0339         -0.0474                0.0242  0.3274  0.1623 -0.7162    12428.6188     14539.6619        0.1254         0.0125 0.1567       Fire
56bdc1b638b6 -1.7769          0.9954            -0.0480         -0.0341                0.0868  0.4271  0.2330 -0.6031   229274.5379       168.0490        0.0959         0.0500 0.0409       Fire
d9b24ca3488d -1.7385          0.9950            -0.0437         -0.0568                0.1441  0.2863  0.2336 -0.1656      201.2347      3335.5941        0.0900         0.0799 0.1010       Fire
e1598d3a19a9 -1.6554          0.9940            -0.0170         -0.0145                0.0193  0.1167  0.4283 -0.6409     2598.9153     13800.6839        0.0376         0.0066 0.1531       Fire
556361142ae5 -1.6468          0.9939            -0.0120         -0.0290               -0.0285 -1.3064  0.3744 -0.5015       10.2739      2215.2394        0.0130         0.0042 0.1887       Fire
70bcd2ad79cc -1.5910          0.9931            -0.0127         -0.0376               -0.1877  0.6325  0.0408  0.0031     6448.0335         9.6440        0.0049         0.0078 0.2700  Explosion
a95cb9ead6e9 -1.5690          0.9927            -0.0243         -0.0342                0.0742 -0.0606  0.3108 -0.4684       19.0779      3261.8981        0.0308         0.0132 0.1730       Fire
c5f7af3881e5 -1.5252          0.9919            -0.0711         -0.0395                0.0653 -0.0201  0.3056  0.0090        9.6631        12.9538        0.0076         0.0076 0.1408       Fire
7dc95a1b5b03 -1.4751          0.9909            -0.0195         -0.0070                0.0356 -0.2091  0.4058 -0.8102       15.8458       651.6640        0.0071         0.0677 0.2424    Volcano
fe5c2d9002dc -1.4171          0.9896            -0.0172         -0.0186                0.0918  0.6300  0.2287 -0.6930    12413.5977      4575.5149        0.1542         0.0107 0.1697       Fire
7952fee21990 -1.3757          0.9885            -0.0191         -0.0423               -0.2960  1.0120 -0.0793 -0.0015     5405.2563        13.1224        0.0111         0.0116 0.1894    Volcano
bfd337e9cefa -1.3352          0.9873            -0.0206         -0.0109               -0.0285  1.0773  0.3523 -0.5420    17298.0500      8783.8679        0.0480         0.0026 0.0467       Fire
c6d5ef4e36ee -1.3238          0.9870             0.0141         -0.0352               -0.1854  0.1431  0.2279  0.0128    36956.9711     62185.9672        0.0541         0.1075 0.0395 Earthquake
a6c3d31df5bd -1.3039          0.9863            -0.0124         -0.0113               -0.1700 -0.0447  0.5627  0.0027      530.7167      7699.5344        0.0045         0.0051 0.3085       Fire
71cf541cdaf5 -1.2850          0.9856            -0.0373         -0.0311               -0.0975 -0.5921  0.0021  0.0604       14.4444      1402.8259        0.0150         0.0720 0.1677       Fire
```
