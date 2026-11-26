# Audit Forensik Matematis (4 Pengujian Wajib) pada 3.180 pasangan citra (pre.jpg vs post.jpg)
import os, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import open_clip
from huggingface_hub import hf_hub_download

t0 = time.time()
ART = Path("/marimo/Fenomena")
CLASS_NAMES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]

DISASTER_PROMPTS = [
    "aerial satellite image of severe disaster damage, collapsed buildings, burn scars, flooding, blast craters, or volcanic ash",
    "post-disaster satellite view showing ruined infrastructure, debris fields, charred terrain, or submerged neighborhoods",
    "catastrophic destruction visible in overhead imagery after earthquake, wildfire, hurricane, tsunami, volcano, or explosion",
]
INTACT_PROMPTS = [
    "aerial satellite image of intact normal city buildings, healthy green vegetation, and undamaged roads",
    "pre-disaster peaceful satellite view with pristine roofs, clear streets, and thriving forest or coastline",
    "undamaged overhead landscape showing normal residential blocks, agriculture, and intact infrastructure",
]
CLASS_PROMPTS = {
    "Earthquake": [
        "aerial satellite view of collapsed buildings and gray concrete rubble after an earthquake",
        "earthquake structural devastation with crumbled city blocks and fractured roads",
    ],
    "Fire": [
        "aerial satellite image of charred black vegetation burn scar from wildfire",
        "post-wildfire satellite view with blackened burnt terrain and smoke plumes",
    ],
    "Hurricane": [
        "aerial satellite image of hurricane wind damage with stripped roofs and uprooted trees",
        "widespread tropical cyclone damage with scattered debris and inland storm flooding",
    ],
    "Tsunami": [
        "aerial satellite image of tsunami coastal flooding with shoreline buildings swept bare",
        "eroded coastline and muddy seawater inundation extending inland after a tsunami",
    ],
    "Volcano": [
        "aerial satellite image of thick gray volcanic ash blanket covering land and roofs",
        "volcanic eruption aftermath with dark lava flows and pyroclastic deposits",
    ],
    "Explosion": [
        "aerial satellite image of a localized circular explosion crater and radial blast damage",
        "industrial or urban blast epicenter showing radial shockwave destruction",
    ],
}

df_inv = pd.read_parquet(ART / "df_inventory.parquet")
df_chg = pd.read_parquet(ART / "df_change.parquet")
df_v1 = pd.read_parquet(ART / "pseudo_labels.parquet")

print(f"[{time.strftime('%H:%M:%S')}] Test 1: Semantic Vector Asymmetry (RemoteCLIP & SkyCLIP, CPU prompt encoding)...", flush=True)

def encode_anchor_vectors(repo_id, filename):
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14")
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    raw = {k.replace("module.", ""): v for k, v in raw.items()}
    model.load_state_dict(raw, strict=True)
    tok = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    with torch.no_grad():
        def enc_mean(plist):
            tf = F.normalize(model.encode_text(tok(plist)).float(), dim=-1)
            return F.normalize(tf.mean(dim=0, keepdim=True), dim=-1).numpy()[0]
        u_dis = enc_mean(DISASTER_PROMPTS)
        u_int = enc_mean(INTACT_PROMPTS)
        c_mat = np.stack([enc_mean(CLASS_PROMPTS[c]) for c in CLASS_NAMES], axis=0)
    del model
    return u_dis, u_int, c_mat

sem_res = {}
for mname, repo, fn in [
    ("RemoteCLIP", "chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"),
    ("SkyCLIP", "furiousteabag/SkyCLIP", "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS_epoch_20.pt"),
]:
    u_dis, u_int, c_mat = encode_anchor_vectors(repo, fn)
    e_pre = np.load(ART / f"emb_{mname}_pre.npy")
    e_post = np.load(ART / f"emb_{mname}_post.npy")
    s_post = (e_post @ u_dis) - (e_post @ u_int)
    s_pre = (e_pre @ u_dis) - (e_pre @ u_int)
    dS_sem = (s_post - s_pre).astype(np.float64)

    d_vec = e_post - e_pre
    d_vec = d_vec / (np.linalg.norm(d_vec, axis=1, keepdims=True) + 1e-8)
    proj = d_vec @ c_mat.T
    fwd_proj = proj.max(axis=1).astype(np.float64)
    rev_proj = (-proj).max(axis=1).astype(np.float64)
    acpr_diff = fwd_proj - rev_proj
    acpr_ratio = fwd_proj / (np.abs(rev_proj) + 1e-6)
    sem_res[mname] = {
        "dS_sem": dS_sem,
        "fwd_proj": fwd_proj,
        "rev_proj": rev_proj,
        "acpr_diff": acpr_diff,
        "acpr_ratio": acpr_ratio,
    }

rc_neg = sem_res["RemoteCLIP"]["dS_sem"] < 0
sc_neg = sem_res["SkyCLIP"]["dS_sem"] < 0
rho_sem, p_rho_sem = stats.spearmanr(sem_res["RemoteCLIP"]["dS_sem"], sem_res["SkyCLIP"]["dS_sem"])
kappa_neg = cohen_kappa_score(rc_neg.astype(int), sc_neg.astype(int))
both_neg = int((rc_neg & sc_neg).sum())
either_neg = int((rc_neg | sc_neg).sum())
agree_neg_pct = float((rc_neg == sc_neg).mean() * 100.0)

print(f"  RemoteCLIP dS_sem < 0: {int(rc_neg.sum())}/3180 ({rc_neg.mean()*100:.2f}%)", flush=True)
print(f"  SkyCLIP    dS_sem < 0: {int(sc_neg.sum())}/3180 ({sc_neg.mean()*100:.2f}%)", flush=True)
print(f"  Spearman rho(RC, SC): {rho_sem:.4f} (p={p_rho_sem:.2e}) | Cohen kappa(<0): {kappa_neg:.4f} | both<0: {both_neg} | agree: {agree_neg_pct:.2f}%", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] Test 2: Image Physics, 2D/1D FFT & Boundary Geometry (3180 pairs)...", flush=True)

# Precompute radial HF mask for 512x512 2D FFT
_yy, _xx = np.ogrid[:512, :512]
_dist = np.sqrt((_yy - 256) ** 2 + (_xx - 256) ** 2)
HF_MASK_512 = (_dist >= 0.25 * 256)

def analyze_img_physics(img_bgr):
    # Full 1024x1024 for black-padding geometry & 1D row scanline FFT
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray1024 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 1D Vertical Scanline FFT
    row_prof = gray1024.mean(axis=1).astype(np.float64)
    row_prof = (row_prof - row_prof.mean()) * np.hanning(len(row_prof))
    ps1d = np.abs(np.fft.rfft(row_prof)) ** 2
    ps1d_hi = ps1d[8:]  # exclude low-frequency shading
    scanline_ratio = float(ps1d_hi.max() / (np.median(ps1d_hi) + 1e-8))

    # Black padding mask & concavity
    pad_mask = (rgb[:, :, 0] < 5) & (rgb[:, :, 1] < 5) & (rgb[:, :, 2] < 5)
    pad_frac = float(pad_mask.mean())
    concavity = 0.0
    if pad_frac >= 0.005:
        fg = (~pad_mask).astype(np.uint8)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cmax = max(cnts, key=cv2.contourArea)
            a_obj = float(cv2.contourArea(cmax))
            hull = cv2.convexHull(cmax)
            a_hull = float(cv2.contourArea(hull))
            if a_hull > 1.0:
                concavity = float(max(0.0, 1.0 - a_obj / a_hull))

    # Downsample 512x512 for fast Laplacian, 2D FFT, ExG, Dark fraction
    rgb512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray512 = cv2.resize(gray1024, (512, 512), interpolation=cv2.INTER_AREA)
    lap_var = float(cv2.Laplacian(gray512, cv2.CV_64F).var())

    f2 = np.fft.fftshift(np.fft.fft2(gray512.astype(np.float32) / 255.0))
    ps2 = np.abs(f2) ** 2
    e_hf = float(ps2[HF_MASK_512].mean())

    exg = float((2.0 * rgb512[:, :, 1] - rgb512[:, :, 0] - rgb512[:, :, 2]).mean())
    dark_frac = float((gray512 < 45).mean())

    return lap_var, e_hf, scanline_ratio, exg, dark_frac, pad_frac, concavity

def analyze_pair(row):
    p1 = cv2.imread(row["path_pre"], cv2.IMREAD_COLOR)
    p2 = cv2.imread(row["path_post"], cv2.IMREAD_COLOR)
    lv1, ehf1, sc1, exg1, dk1, pf1, cc1 = analyze_img_physics(p1)
    lv2, ehf2, sc2, exg2, dk2, pf2, cc2 = analyze_img_physics(p2)
    return {
        "id": row["id"],
        "lap_var_pre": lv1, "lap_var_post": lv2,
        "d_log_lap": float(np.log((lv2 + 1e-6) / (lv1 + 1e-6))),
        "e_hf_pre": ehf1, "e_hf_post": ehf2,
        "R_HF": float(np.log((ehf2 + 1e-8) / (ehf1 + 1e-8))),
        "scanline_pre": sc1, "scanline_post": sc2,
        "d_log_scanline": float(np.log((sc2 + 1e-6) / (sc1 + 1e-6))),
        "exg_pre": exg1, "exg_post": exg2,
        "d_exg": float(exg2 - exg1),
        "dark_pre": dk1, "dark_post": dk2,
        "d_dark": float(dk2 - dk1),
        "pad_frac_pre": pf1, "pad_frac_post": pf2,
        "concavity_pre": cc1, "concavity_post": cc2,
    }

with ThreadPoolExecutor(max_workers=16) as ex:
    phys_rows = list(ex.map(analyze_pair, df_inv.to_dict("records")))
df_phys = pd.DataFrame(phys_rows)
print(f"  Physics audit finished in {time.time()-t0:.1f}s total", flush=True)

# Merge all signals
df_audit = df_phys.copy()
for mname in ["RemoteCLIP", "SkyCLIP"]:
    for k, arr in sem_res[mname].items():
        df_audit[f"{k}_{mname}"] = arr

df_audit = df_audit.merge(df_chg[["id", "ssim", "mae_r", "mae_g", "mae_b", "d_bright"]], on="id", how="left")
df_audit = df_audit.merge(df_v1[["id", "vlm_label", "vlm_conf"]], on="id", how="left")

# Test 3: Composite Chronological Direction Score Z_dir + Normality + 2-GMM
def znorm(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

z_components = [
    znorm(df_audit["dS_sem_RemoteCLIP"].values),
    znorm(df_audit["dS_sem_SkyCLIP"].values),
    znorm(df_audit["acpr_diff_RemoteCLIP"].values),
    znorm(df_audit["acpr_diff_SkyCLIP"].values),
    znorm(-df_audit["d_exg"].values),      # vegetation loss = normal forward transition
    znorm(df_audit["d_dark"].values),      # darkening = normal forward transition
]
Z_dir = np.mean(np.stack(z_components, axis=0), axis=0)
df_audit["Z_dir"] = Z_dir

k2_stat, k2_p = stats.normaltest(Z_dir)
skew_z = float(stats.skew(Z_dir))
kurt_z = float(stats.kurtosis(Z_dir, fisher=True))

X_z = Z_dir.reshape(-1, 1)
gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X_z)
gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=10).fit(X_z)
bic1 = float(gmm1.bic(X_z))
bic2 = float(gmm2.bic(X_z))
delta_bic = bic1 - bic2

means2 = gmm2.means_.ravel()
stds2 = np.sqrt(gmm2.covariances_.ravel())
weights2 = gmm2.weights_.ravel()
inv_comp = int(np.argmin(means2))
norm_comp = 1 - inv_comp
p_inv_gmm = gmm2.predict_proba(X_z)[:, inv_comp]
df_audit["p_inverted_gmm"] = p_inv_gmm

# Suspected inverted: posterior > 0.5 AND Z_dir < 0 (and also check semantic consensus < 0)
df_audit["is_inverted"] = (p_inv_gmm > 0.5) & (Z_dir < 0)
df_audit["is_sem_both_neg"] = (df_audit["dS_sem_RemoteCLIP"] < 0) & (df_audit["dS_sem_SkyCLIP"] < 0)

inv_mask = df_audit["is_inverted"].values
n_inv = int(inv_mask.sum())
if 0 < n_inv < len(df_audit):
    mw_u, mw_p = stats.mannwhitneyu(Z_dir[inv_mask], Z_dir[~inv_mask], alternative="less")
    ks_d, ks_p = stats.ks_2samp(Z_dir[inv_mask], Z_dir[~inv_mask])
    mw_ssim_u, mw_ssim_p = stats.mannwhitneyu(df_audit.loc[inv_mask, "ssim"], df_audit.loc[~inv_mask, "ssim"])
else:
    mw_u = mw_p = ks_d = ks_p = mw_ssim_u = mw_ssim_p = np.nan

# Cross-check vs pseudo_labels.parquet (VLM 7B)
ct_inv = pd.crosstab(df_audit["is_inverted"], df_audit["vlm_label"])
chi2_vlm, p_chi2_vlm, _, _ = stats.chi2_contingency(ct_inv) if n_inv > 0 else (np.nan, np.nan, 0, None)
ct_sem = pd.crosstab(df_audit["is_sem_both_neg"], df_audit["vlm_label"])
chi2_sem, p_chi2_sem, _, _ = stats.chi2_contingency(ct_sem)

df_audit.to_parquet(ART / "forensic_math_audit.parquet", index=False)

# Visual grid: Top-12 most inverted pairs ([pre | post])
top15 = df_audit.sort_values("Z_dir", ascending=True).head(15).reset_index(drop=True)
top12 = top15.head(12)

fig, axes = plt.subplots(4, 3, figsize=(16, 11))
for idx, r in top12.iterrows():
    ri, ci = divmod(idx, 3)
    row_inv = df_inv[df_inv["id"] == r["id"]].iloc[0]
    p1 = cv2.cvtColor(cv2.resize(cv2.imread(row_inv["path_pre"]), (224, 224)), cv2.COLOR_BGR2RGB)
    p2 = cv2.cvtColor(cv2.resize(cv2.imread(row_inv["path_post"]), (224, 224)), cv2.COLOR_BGR2RGB)
    sep = np.full((224, 4, 3), 255, dtype=np.uint8)
    strip = np.hstack([p1, sep, p2])
    axes[ri, ci].imshow(strip)
    axes[ri, ci].set_title(
        f"{r['id']} | Z={r['Z_dir']:.2f} | RC={r['dS_sem_RemoteCLIP']:.2f} SC={r['dS_sem_SkyCLIP']:.2f}\ndExG={r['d_exg']:+.2f} | VLM7B={r['vlm_label']}",
        fontsize=8.5,
    )
    axes[ri, ci].axis("off")
fig.suptitle("Top-12 Pasangan dengan Skor Inversi Kronologis Tertinggi [Kiri=pre.jpg | Kanan=post.jpg]", fontsize=12)
fig.tight_layout()
fig.savefig(ART / "forensic_inverted_top12.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Build report
cols_top15 = [
    "id", "Z_dir", "p_inverted_gmm", "dS_sem_RemoteCLIP", "dS_sem_SkyCLIP",
    "acpr_diff_RemoteCLIP", "R_HF", "d_exg", "d_dark", "scanline_pre", "scanline_post",
    "pad_frac_pre", "pad_frac_post", "ssim", "vlm_label"
]

report = f"""# FORENSIC AUDIT REPORT — Pengujian Matematis Pre vs Post (3.180 Pasangan)

## 1. Uji Asimetri Vektor Semantik (`RemoteCLIP` & `SkyCLIP`)
- **Pasangan dengan $\\Delta S_{{\\text{{sem}}}} < 0$ (`pre` lebih rusak dibanding `post`):**
  - `RemoteCLIP`: **{int(rc_neg.sum())} / 3.180** (`{rc_neg.mean()*100:.2f}%`), mean $\\Delta S_{{\\text{{sem}}}} = {sem_res['RemoteCLIP']['dS_sem'].mean():.4f}$ (median `{np.median(sem_res['RemoteCLIP']['dS_sem']):.4f}`)
  - `SkyCLIP`: **{int(sc_neg.sum())} / 3.180** (`{sc_neg.mean()*100:.2f}%`), mean $\\Delta S_{{\\text{{sem}}}} = {sem_res['SkyCLIP']['dS_sem'].mean():.4f}$ (median `{np.median(sem_res['SkyCLIP']['dS_sem']):.4f}`)
  - **Keduanya $\\Delta S_{{\\text{{sem}}}} < 0$ (`is_sem_both_neg`):** **{both_neg} / 3.180** (`{both_neg/3180*100:.2f}%`) | Salah satu `< 0`: `{either_neg}` (`{either_neg/3180*100:.2f}%`)
- **Konsistensi Lintas Model:**
  - Korelasi peringkat Spearman $\\rho$: **`{rho_sem:.4f}`** ($p = {p_rho_sem:.2e}$)
  - Cohen's Kappa $\\kappa$ (deteksi $\\Delta S_{{\\text{{sem}}}} < 0$): **`{kappa_neg:.4f}`** (kesepakatan mentah `{agree_neg_pct:.2f}%`)
- **Antisymmetric Class Projection Ratio (`F_max - B_max`):**
  - `RemoteCLIP`: mean `{sem_res['RemoteCLIP']['acpr_diff'].mean():.4f}`, fraksi `< 0`: `{(sem_res['RemoteCLIP']['acpr_diff'] < 0).mean()*100:.2f}%`
  - `SkyCLIP`: mean `{sem_res['SkyCLIP']['acpr_diff'].mean():.4f}`, fraksi `< 0`: `{(sem_res['SkyCLIP']['acpr_diff'] < 0).mean()*100:.2f}%`

## 2. Forensik Fisika Citra, FFT (2D & 1D), & Geometri Batas
- **Ketajaman & Energi HF ($R_{{\\text{{HF}}}}$):**
  - Mean $\\Delta \\ln(\\sigma^2_{{\\nabla^2}})$: `{df_audit['d_log_lap'].mean():.4f}` (fraksi `post < pre`: `{(df_audit['d_log_lap'] < 0).mean()*100:.2f}%`)
  - Mean $R_{{\\text{{HF}}}}$: `{df_audit['R_HF'].mean():.4f}` (std `{df_audit['R_HF'].std():.4f}`, fraksi `< 0`: `{(df_audit['R_HF'] < 0).mean()*100:.2f}%`)
- **Artefak Scanline Periodik (1D FFT Vertikal, rasio `peak/median`):**
  - `scanline_pre`: median `{df_audit['scanline_pre'].median():.2f}`, P95 `{df_audit['scanline_pre'].quantile(0.95):.2f}`, max `{df_audit['scanline_pre'].max():.2f}`
  - `scanline_post`: median `{df_audit['scanline_post'].median():.2f}`, P95 `{df_audit['scanline_post'].quantile(0.95):.2f}`, max `{df_audit['scanline_post'].max():.2f}`
  - Anomali scanline ekstrem (`ratio > 50`): `pre={int((df_audit['scanline_pre']>50).sum())}`, `post={int((df_audit['scanline_post']>50).sum())}`
- **Vegetasi (`ExG`) & Piksel Gelap (`<45`):**
  - Mean $\\Delta \\text{{ExG}}$: `{df_audit['d_exg'].mean():.4f}` (fraksi $\\Delta \\text{{ExG}} > 0$ alias `post` lebih hijau dari `pre`: `{(df_audit['d_exg'] > 0).mean()*100:.2f}%`)
  - Mean $\\Delta \\text{{dark}}$: `{df_audit['d_dark'].mean():.4f}` (fraksi $\\Delta \\text{{dark}} < 0$: `{(df_audit['d_dark'] < 0).mean()*100:.2f}%`)
- **Geometri Padding Hitam (*Warped Orthomosaic*):**
  - Pasangan dengan `pad_frac >= 0.5%`: `pre={int((df_audit['pad_frac_pre']>=0.005).sum())}`, `post={int((df_audit['pad_frac_post']>=0.005).sum())}`, beda padding `|pre-post| > 2%`: `{int((np.abs(df_audit['pad_frac_pre']-df_audit['pad_frac_post'])>0.02).sum())}`
  - Mean `concavity` pada citra ber-padding (`>=0.5%`): `pre={df_audit.loc[df_audit['pad_frac_pre']>=0.005, 'concavity_pre'].mean():.4f}`, `post={df_audit.loc[df_audit['pad_frac_post']>=0.005, 'concavity_post'].mean():.4f}`

## 3. Uji Hipotesis Statistik & Dekomposisi GMM pada $Z_{{\\text{{dir}}}}$
- **Statistik $Z_{{\\text{{dir}}}}$:** mean `{Z_dir.mean():.4f}`, std `{Z_dir.std():.4f}`, min `{Z_dir.min():.4f}`, max `{Z_dir.max():.4f}`
- **Uji Normalitas:** D'Agostino-Pearson $K^2 = {k2_stat:.2f}$ ($p = {k2_p:.2e}$), *Skewness* = `{skew_z:.4f}`, *Excess Kurtosis* = `{kurt_z:.4f}`
- **Dekomposisi GMM (1-Komponen vs 2-Komponen):**
  - `BIC(1)` = `{bic1:.2f}` vs `BIC(2)` = `{bic2:.2f}` → **$\\Delta \\text{{BIC}} = {delta_bic:.2f}$**
  - Komponen 1 (diduga terbalik): $\\mu = {means2[inv_comp]:.4f}$, $\\sigma = {stds2[inv_comp]:.4f}$, bobot $= {weights2[inv_comp]:.4f}$
  - Komponen 2 (normal): $\\mu = {means2[norm_comp]:.4f}$, $\\sigma = {stds2[norm_comp]:.4f}$, bobot $= {weights2[norm_comp]:.4f}$
  - **Jumlah pasangan berstatus `is_inverted` ($P_{{\\text{{inv}}}} > 0.5 \\land Z_{{\\text{{dir}}}} < 0$):** **{n_inv} / 3.180 (`{n_inv/3180*100:.2f}%`)**
  - Uji Mann-Whitney U (`Z_dir`): $U = {mw_u:.1f}$, $p = {mw_p:.2e}$ | KS-test: $D = {ks_d:.4f}$, $p = {ks_p:.2e}$
  - Uji Mann-Whitney U (`SSIM`): $p = {mw_ssim_p:.2e}$ (mean SSIM inverted=`{df_audit.loc[inv_mask, 'ssim'].mean():.4f}` vs normal=`{df_audit.loc[~inv_mask, 'ssim'].mean():.4f}`)

## 4. Audit Silang terhadap `pseudo_labels.parquet` (VLM 7B)
- Chi-square `is_inverted` vs `vlm_label`: $\\chi^2 = {chi2_vlm:.2f}$ ($p = {p_chi2_vlm:.2e}$)
- Chi-square `is_sem_both_neg` (`{both_neg}` sampel) vs `vlm_label`: $\\chi^2 = {chi2_sem:.2f}$ ($p = {p_chi2_sem:.2e}$)
- Distribusi `vlm_label` pada `is_sem_both_neg` vs normal:
```
{ct_sem.to_string()}
```

## 5. Top-15 Id Paling Terbalik (Berdasarkan $Z_{{\\text{{dir}}}}$ Terendah)
```
{top15[cols_top15].round(4).to_string(index=False)}
```
"""

(ART / "FORENSIC_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
print(report)
print(f"\n=== AUDIT FORENSIK SELESAI dalam {time.time()-t0:.1f}s ===", flush=True)
