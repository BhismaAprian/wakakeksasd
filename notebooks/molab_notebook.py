import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return


@app.cell
def _():
    import zipfile
    import os

    zip_path = "/marimo/Fenomena.zip"
    extract_to = "/marimo/dataset"

    # Buat folder dataset jika belum ada
    os.makedirs(extract_to, exist_ok=True)

    # Ekstrak zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"File di dalam zip ({len(zip_ref.namelist())} file):")
        for name in zip_ref.namelist():
            print(f"  {name}")
        zip_ref.extractall(extract_to)

    print(f"\n✅ Ekstrak selesai! File tersimpan di: {extract_to}")

    # Tampilkan hasil ekstrak
    print("\nIsi folder dataset:")
    for root, dirs, files in os.walk(extract_to):
        level = root.replace(extract_to, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            print(f"{subindent}{file} ({size:,} bytes)")
    return


@app.cell(hide_code=True)
def M1_setup(mo):
    mo.md("""
    ## 1. Setup & Konfigurasi

    **Konteks Kompetisi:** Klasifikasi 6 kelas fenomena bencana (*Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion*) dari pasangan citra sebelum (`pre.jpg`) dan sesudah (`post.jpg`) kejadian pada `/marimo/dataset/Fenomena/{Id}/`, dievaluasi dengan metrik **Macro F1-Score**.

    - **Tujuan tahap:** Mengimpor seluruh library visi komputer, analisis spasial, dan statistik (`numpy`, `pandas`, `matplotlib`, `PIL`, `cv2`, `tqdm`, `scipy.stats`, `skimage.metrics.structural_similarity`), menetapkan `RANDOM_SEED=42`, serta memverifikasi lokasi dataset, file submission, dan keberadaan file metadata/label awal.
    - **Output yang diharapkan:** Ringkasan jumlah folder `Id`, jumlah file `pre.jpg`/`post.jpg`, path `sample_submission.csv`, dan daftar file non-citra (`.csv`/`.json`/`.npy`) di `DATA_ROOT`.
    - **Interpretasi:** Dataset terverifikasi utuh: terdapat **3.180** folder `Id` dengan **3.180** file `pre.jpg` dan **3.180** file `post.jpg`, `SUB_PATH` ditemukan di `/marimo/sample_submission.csv`, dan tidak ada file label/ekstra di dalam `DATA_ROOT`.
    """)
    return


@app.cell
def K1_setup():
    # Setup library global, seed deterministik, path dataset/artefak,
    # serta audit awal jumlah folder Id, pasangan citra, dan file non-citra di DATA_ROOT.
    import os as _os
    import glob
    import json
    import random
    import hashlib
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import cv2
    from tqdm.auto import tqdm
    from scipy import stats as scipy_stats
    from skimage.metrics import structural_similarity as ssim_fn

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    DATA_ROOT = "/marimo/dataset/Fenomena"
    ARTIFACT_DIR = "/marimo/Fenomena"
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)

    _sub_candidates = [
        _p for _base in ["/marimo", "/marimo/dataset"]
        for _p in glob.glob(f"{_base}/**/*submission*.csv", recursive=True)
    ]
    SUB_PATH = sorted(_sub_candidates)[0] if _sub_candidates else None

    _id_dirs = sorted([
        _d.name for _d in Path(DATA_ROOT).iterdir() if _d.is_dir()
    ]) if Path(DATA_ROOT).exists() else []

    _n_pre = sum((Path(DATA_ROOT) / _d / "pre.jpg").is_file() for _d in _id_dirs)
    _n_post = sum((Path(DATA_ROOT) / _d / "post.jpg").is_file() for _d in _id_dirs)

    _extra_files = []
    for _ext in ("*.csv", "*.json", "*.npy", "*.npz", "*.parquet"):
        _extra_files.extend(glob.glob(f"{DATA_ROOT}/**/{_ext}", recursive=True))
    _extra_files = sorted(set(_extra_files))

    _summary_k1 = pd.DataFrame([{
        "data_root_exists": Path(DATA_ROOT).exists(),
        "n_id_folders": len(_id_dirs),
        "n_pre_jpg": _n_pre,
        "n_post_jpg": _n_post,
        "sub_path": SUB_PATH,
        "n_extra_files_in_data_root": len(_extra_files),
    }])
    print("Extra files di DATA_ROOT:", _extra_files)
    _summary_k1
    return (
        ARTIFACT_DIR,
        DATA_ROOT,
        Image,
        Path,
        RANDOM_SEED,
        SUB_PATH,
        cv2,
        glob,
        hashlib,
        json,
        np,
        pd,
        plt,
        random,
        scipy_stats,
        ssim_fn,
        tqdm,
    )


@app.cell(hide_code=True)
def M2_inventory(mo):
    mo.md("""
    ## 2. Inventaris Data & Verifikasi Label

    - **Tujuan tahap:** Menginventarisasi seluruh 3.180 folder `Id` beserta atribut file (`path`, ukuran byte, hash MD5, dimensi, dan mode warna `pre.jpg`/`post.jpg`), mendeteksi duplikat citra/pasangan, serta memindai direktori proyek untuk memastikan apakah terdapat file label training (`*.csv`, `*.json`, `label*`, `train*`).
    - **Output yang diharapkan:** DataFrame `df_inv` (disimpan ke `/marimo/Fenomena/df_inventory.parquet`), ringkasan statistik `describe()`, jumlah duplikat MD5, serta daftar file kandidat label yang ditemukan.
    - **Interpretasi:** Seluruh **3.180** `Id` lengkap (`0` missing), **0** duplikat MD5 (baik `pre`, `post`, pasangan, maupun `pre==post`), seluruh citra seragam berukuran `1024x1024` dengan mode `RGB`. Satu-satunya file CSV/JSON yang ditemukan adalah `/marimo/sample_submission.csv` (tidak ditemukan file label training).
    """)
    return


@app.cell
def K2_inventory(
    ARTIFACT_DIR,
    DATA_ROOT,
    Image,
    Path,
    glob,
    hashlib,
    np,
    pd,
    tqdm,
):
    # Inventarisasi 3.180 pasangan Id (ukuran, MD5, dimensi, mode warna),
    # deteksi duplikat MD5, dan pencarian file label di workspace/dataset.
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _md5_file(_p):
        _h = hashlib.md5()
        with open(_p, "rb") as _f:
            for _chunk in iter(lambda: _f.read(65536), b""):
                _h.update(_chunk)
        return _h.hexdigest()

    def _inspect_one_id(_id_name):
        _dir = Path(DATA_ROOT) / _id_name
        _p_pre = _dir / "pre.jpg"
        _p_post = _dir / "post.jpg"
        _rec = {
            "id": _id_name,
            "path_pre": str(_p_pre) if _p_pre.is_file() else "MISSING",
            "path_post": str(_p_post) if _p_post.is_file() else "MISSING",
            "size_bytes_pre": np.nan,
            "size_bytes_post": np.nan,
            "hash_md5_pre": "MISSING",
            "hash_md5_post": "MISSING",
            "dim_pre": "MISSING",
            "dim_post": "MISSING",
            "mode_pre": "MISSING",
            "mode_post": "MISSING",
        }
        for _side, _p in [("pre", _p_pre), ("post", _p_post)]:
            if _p.is_file():
                _rec[f"size_bytes_{_side}"] = _p.stat().st_size
                _rec[f"hash_md5_{_side}"] = _md5_file(_p)
                try:
                    with Image.open(_p) as _im:
                        _rec[f"dim_{_side}"] = f"{_im.size[0]}x{_im.size[1]}"
                        _rec[f"mode_{_side}"] = _im.mode
                except Exception:
                    _rec[f"dim_{_side}"] = "CORRUPT"
                    _rec[f"mode_{_side}"] = "CORRUPT"
        return _rec

    _all_ids = sorted([_d.name for _d in Path(DATA_ROOT).iterdir() if _d.is_dir()])
    with _TPE(max_workers=16) as _ex:
        _records = list(tqdm(_ex.map(_inspect_one_id, _all_ids), total=len(_all_ids), desc="Inventory"))

    df_inv = pd.DataFrame(_records)
    _inv_path = Path(ARTIFACT_DIR) / "df_inventory.parquet"
    df_inv.to_parquet(_inv_path, index=False)

    _dup_pre = int(df_inv.loc[df_inv["hash_md5_pre"] != "MISSING", "hash_md5_pre"].duplicated().sum())
    _dup_post = int(df_inv.loc[df_inv["hash_md5_post"] != "MISSING", "hash_md5_post"].duplicated().sum())
    _dup_pair = int(df_inv.duplicated(subset=["hash_md5_pre", "hash_md5_post"]).sum())
    _cross_same = int((df_inv["hash_md5_pre"] == df_inv["hash_md5_post"]).sum())

    _found_files = []
    for _base in ["/marimo", "/marimo/dataset"]:
        for _pat in ["**/*.csv", "**/*.json", "**/label*", "**/train*"]:
            for _m in glob.glob(f"{_base}/{_pat}", recursive=True):
                if any(_skip in _m for _skip in ["__marimo__", "/.config/", "/.venv/", "/.cache/"]):
                    continue
                _found_files.append(_m)
    label_file_candidates = sorted(set(_found_files))

    print(f"Total Id: {len(df_inv)} | Missing pre: {(df_inv['path_pre']=='MISSING').sum()} | Missing post: {(df_inv['path_post']=='MISSING').sum()}")
    print(f"Duplikat MD5 -> pre: {_dup_pre}, post: {_dup_post}, pasangan identik: {_dup_pair}, pre==post dalam 1 Id: {_cross_same}")
    print("Kandidat file label/CSV/JSON:", label_file_candidates)
    print("\ndf_inv.head():")
    print(df_inv.head().to_string())
    print("\ndf_inv.describe(include='all'):")
    print(df_inv.describe(include="all").to_string())
    df_inv.head()
    return df_inv, label_file_candidates


@app.cell(hide_code=True)
def M3_integrity(mo):
    mo.md("""
    ## 3. Audit Integritas Citra

    - **Tujuan tahap:** Menguji keterbacaan penuh (*full decode*) setiap file `pre.jpg` dan `post.jpg` dengan `PIL` untuk mendeteksi JPEG corrupt/terpotong, memeriksa konsistensi dimensi (`dim_pre == dim_post`) serta mode warna (`RGB`), dan memvisualisasikan distribusi rasio aspek serta resolusi citra.
    - **Output yang diharapkan:** Daftar `Id` bermasalah (`df_integrity_issues`), flag status valid `df_inv_clean`, serta plot histogram 2 subplot (rasio aspek dan resolusi lebar piksel).
    - **Interpretasi:** Seluruh **3.180** pasangan (`6.360` citra) berhasil didekode tanpa error (`0` corrupt, `0` non-RGB, `0` mismatch dimensi). Seluruh citra memiliki resolusi `1024x1024` (rasio aspek `1.0`), sehingga tidak ada kandidat drop dari sisi integritas file maupun perbedaan resolusi.
    """)
    return


@app.cell
def K3_integrity(Image, df_inv, np, pd, plt, tqdm):
    # Audit full-decode PIL, cek mismatch dimensi pre vs post, mode warna non-RGB,
    # serta plot histogram rasio aspek & resolusi (2 subplot).
    from concurrent.futures import ThreadPoolExecutor as _TPE

    plt.close("all")

    def _audit_one(_row):
        _id = _row["id"]
        _issues = []
        _w_pre = _h_pre = _w_post = _h_post = np.nan
        _m_pre = _m_post = "MISSING"

        for _side in ("pre", "post"):
            _p = _row[f"path_{_side}"]
            if _p == "MISSING":
                _issues.append(f"missing_{_side}")
                continue
            try:
                with Image.open(_p) as _im:
                    _im.load()
                    _w, _h = _im.size
                    _m = _im.mode
                if _side == "pre":
                    _w_pre, _h_pre, _m_pre = _w, _h, _m
                else:
                    _w_post, _h_post, _m_post = _w, _h, _m
                if _m != "RGB":
                    _issues.append(f"non_rgb_{_side}:{_m}")
            except Exception as _e:
                _issues.append(f"corrupt_{_side}:{type(_e).__name__}")

        if not np.isnan(_w_pre) and not np.isnan(_w_post):
            if (_w_pre != _w_post) or (_h_pre != _h_post):
                _issues.append(f"dim_mismatch:{int(_w_pre)}x{int(_h_pre)}_vs_{int(_w_post)}x{int(_h_post)}")

        return {
            "id": _id,
            "w_pre": _w_pre,
            "h_pre": _h_pre,
            "w_post": _w_post,
            "h_post": _h_post,
            "aspect_pre": (_w_pre / _h_pre) if _h_pre else np.nan,
            "aspect_post": (_w_post / _h_post) if _h_post else np.nan,
            "mode_pre": _m_pre,
            "mode_post": _m_post,
            "issues": ";".join(_issues),
        }

    _rows = df_inv.to_dict("records")
    with _TPE(max_workers=16) as _ex:
        _audit_rows = list(tqdm(_ex.map(_audit_one, _rows), total=len(_rows), desc="Integrity Audit"))

    _df_audit = pd.DataFrame(_audit_rows)
    df_integrity_issues = _df_audit[_df_audit["issues"] != ""].copy()
    df_inv_clean = df_inv[~df_inv["id"].isin(df_integrity_issues["id"])].reset_index(drop=True)

    print(f"Total diaudit: {len(_df_audit)} | Bermasalah: {len(df_integrity_issues)} | Valid bersih: {len(df_inv_clean)}")
    print("Distribusi dimensi pre:", _df_audit.apply(lambda r: f"{int(r.w_pre)}x{int(r.h_pre)}" if pd.notna(r.w_pre) else "ERR", axis=1).value_counts().to_dict())
    print("Distribusi dimensi post:", _df_audit.apply(lambda r: f"{int(r.w_post)}x{int(r.h_post)}" if pd.notna(r.w_post) else "ERR", axis=1).value_counts().to_dict())
    print("Mode pre:", _df_audit["mode_pre"].value_counts().to_dict(), "| Mode post:", _df_audit["mode_post"].value_counts().to_dict())
    if len(df_integrity_issues) > 0:
        print("\nTabel Id bermasalah:")
        print(df_integrity_issues.head(20).to_string(index=False))
    else:
        print("Tidak ditemukan Id corrupt, missing, non-RGB, maupun mismatch dimensi.")

    _fig_k3, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _aspects = pd.concat([_df_audit["aspect_pre"], _df_audit["aspect_post"]]).dropna()
    _widths = pd.concat([_df_audit["w_pre"], _df_audit["w_post"]]).dropna()

    _axes[0].hist(_aspects, bins=21, range=(0.5, 1.5), color="#2b6cb0", edgecolor="white")
    _axes[0].set_title("Distribusi Rasio Aspek (W/H) Pre & Post")
    _axes[0].set_xlabel("Aspect Ratio")
    _axes[0].set_ylabel("Jumlah Citra")
    _axes[0].axvline(1.0, color="#e53e3e", linestyle="--", label="1.0 (Square)")
    _axes[0].legend()

    _axes[1].hist(_widths, bins=21, range=(512, 1536), color="#2f855a", edgecolor="white")
    _axes[1].set_title("Distribusi Resolusi Lebar (Piksel) Pre & Post")
    _axes[1].set_xlabel("Lebar (px)")
    _axes[1].set_ylabel("Jumlah Citra")
    _axes[1].axvline(1024, color="#e53e3e", linestyle="--", label="1024 px")
    _axes[1].legend()

    _fig_k3.tight_layout()
    _fig_k3
    return df_integrity_issues, df_inv_clean


@app.cell(hide_code=True)
def M4_labels(mo):
    mo.md("""
    ## 4. Distribusi Kelas (Jika Label Ada)

    - **Tujuan tahap:** Memverifikasi apakah kandidat file CSV/JSON yang ditemukan berisi label training ground-truth (`Id, Phenomenon`) atau hanya template submission; apabila label tidak ditemukan, bagian ini akan menampilkan `"TIDAK ADA LABEL"` dan menetapkan pipeline lanjutan ke pendekatan *zero-shot* VLM/CLIP.
    - **Output yang diharapkan:** Status `HAS_LABELS`, tabel distribusi kelas atau pesan ketiadaan label training, serta penyimpanan flag dan ringkasan awal ke `/marimo/Fenomena/_meta.json`.
    - **Interpretasi:** **TIDAK ADA LABEL TRAINING** (`HAS_LABELS = False`). Satu-satunya CSV yang ada adalah `/marimo/sample_submission.csv` berukuran `3.180 x 2` dengan seluruh kolom `Phenomenon` berisi placeholder `'Earthquake'`. Sesuai aturan soal, seluruh 3.180 sampel berstatus uji, sehingga strategi pemodelan wajib berfokus pada *zero-shot* CLIP/SigLIP dan/atau *pseudo-labeling* via VLM.
    """)
    return


@app.cell
def K4_labels(
    ARTIFACT_DIR,
    Path,
    SUB_PATH,
    df_integrity_issues,
    df_inv,
    df_inv_clean,
    json,
    label_file_candidates,
    mo,
    pd,
    plt,
):
    # Verifikasi apakah ada file label ground-truth training (selain sample_submission),
    # cetak status HAS_LABELS, dan simpan metadata awal ke /marimo/Fenomena/_meta.json.
    _sub_df = pd.read_csv(SUB_PATH) if SUB_PATH else pd.DataFrame()
    _sub_classes = _sub_df["Phenomenon"].value_counts(dropna=False).to_dict() if "Phenomenon" in _sub_df.columns else {}
    print("Inspeksi sample_submission.csv -> shape:", _sub_df.shape, "| nilai Phenomenon:", _sub_classes)

    _true_label_files = [
        _p for _p in label_file_candidates
        if "sample_submission" not in Path(_p).name.lower()
    ]

    df_labels = None
    HAS_LABELS = False
    _class_dist = None

    for _cand in _true_label_files:
        if _cand.endswith(".csv"):
            _tmp = pd.read_csv(_cand)
            if {"Id", "Phenomenon"}.issubset(_tmp.columns) or {"id", "Phenomenon"}.issubset(_tmp.columns):
                df_labels = _tmp
                HAS_LABELS = True
                break

    if HAS_LABELS and df_labels is not None:
        _id_col = "Id" if "Id" in df_labels.columns else "id"
        _merged = df_inv_clean.merge(df_labels, left_on="id", right_on=_id_col, how="inner")
        _vc = _merged["Phenomenon"].value_counts()
        _class_dist = _vc.to_dict()
        _imb_ratio = float(_vc.max() / max(_vc.min(), 1))
        print(f"LABEL TRAINING DITEMUKAN ({len(_merged)} baris). Imbalance ratio: {_imb_ratio:.2f}")
        print(_vc.to_string())
        plt.close("all")
        _fig_k4, _ax = plt.subplots(figsize=(8, 4))
        _vc.plot(kind="bar", ax=_ax, color="#3182ce")
        _ax.set_title("Distribusi Kelas Phenomenon")
        _ax.set_ylabel("Jumlah Sampel")
        _fig_k4.tight_layout()
        _out_k4 = _fig_k4
    else:
        print("TIDAK ADA LABEL TRAINING — akan gunakan zero-shot VLM/CLIP")
        _out_k4 = mo.callout(
            mo.md("**TIDAK ADA LABEL TRAINING** — Seluruh 3.180 sampel berstatus uji (sesuai `sample_submission.csv` yang seluruhnya placeholder). Strategi lanjutan: *zero-shot* CLIP/SigLIP dan/atau *pseudo-labeling* via VLM."),
            kind="warn",
        )

    _meta_payload = {
        "n_ids": int(len(df_inv)),
        "n_valid_clean": int(len(df_inv_clean)),
        "n_integrity_issues": int(len(df_integrity_issues)),
        "has_labels": bool(HAS_LABELS),
        "class_distribution": _class_dist,
        "sub_path": SUB_PATH,
        "sample_submission_rows": int(len(_sub_df)),
        "sample_submission_values": {str(_k): int(_v) for _k, _v in _sub_classes.items()},
        "label_file_candidates": label_file_candidates,
    }
    with open(Path(ARTIFACT_DIR) / "_meta.json", "w", encoding="utf-8") as _f:
        json.dump(_meta_payload, _f, indent=2, ensure_ascii=False)
    print("Metadata tersimpan ke:", str(Path(ARTIFACT_DIR) / "_meta.json"))
    _out_k4
    return HAS_LABELS, df_labels


@app.cell(hide_code=True)
def M5_change(mo):
    mo.md("""
    ## 5. Statistik Perubahan Piksel (Change Signal)

    - **Tujuan tahap:** Menghitung sinyal perubahan global (`mae_r`, `mae_g`, `mae_b` pada skala 0–255, `ssim`, `d_bright`, `d_std`) pada resolusi `256x256` RGB untuk seluruh `3.180` pasangan valid sebagai fitur diskriminatif dasar deteksi perubahan, serta menyimpan 3 thumbnail peta selisih absolut (*grayscale diff map*).
    - **Output yang diharapkan:** DataFrame `df_change` yang disimpan ke `/marimo/Fenomena/df_change.parquet`, ringkasan statistik `df_change.describe()`, dan 3 file thumbnail diff map di `/marimo/Fenomena/`.
    - **Interpretasi:** Rata-rata MAE cukup besar (`mae_r=55.61`, `mae_g=52.40`, `mae_b=51.32`) dengan rata-rata SSIM rendah (`0.169`, median `0.146`, rentang `0.006–0.801`) serta variasi kecerahan `d_bright` yang lebar (`std=40.45`). Ini mengindikasikan adanya perbedaan komposisi visual/iluminasi atau pergeseran spasial yang kuat antara `pre` dan `post` pada sebagian besar pasangan.
    """)
    return


@app.cell
def K5_change(
    ARTIFACT_DIR,
    Image,
    Path,
    cv2,
    df_inv_clean,
    np,
    pd,
    plt,
    ssim_fn,
    tqdm,
):
    # Hitung metrik perubahan piksel (MAE R/G/B, SSIM, selisih brightness & std) pada resolusi 256x256 RGB
    # untuk seluruh Id bersih secara paralel, simpan df_change.parquet dan 3 thumbnail diff map grayscale.
    from concurrent.futures import ThreadPoolExecutor as _TPE

    plt.close("all")

    def _compute_change_one(_row):
        _pre = cv2.imread(_row["path_pre"], cv2.IMREAD_COLOR)
        _post = cv2.imread(_row["path_post"], cv2.IMREAD_COLOR)
        _pre = cv2.cvtColor(cv2.resize(_pre, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _post = cv2.cvtColor(cv2.resize(_post, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

        _diff = np.abs(_post.astype(np.float32) - _pre.astype(np.float32))
        _mae_r, _mae_g, _mae_b = _diff.mean(axis=(0, 1))
        _ssim_val = float(ssim_fn(_pre, _post, channel_axis=-1, data_range=255))

        _g_pre = cv2.cvtColor(_pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
        _g_post = cv2.cvtColor(_post, cv2.COLOR_RGB2GRAY).astype(np.float32)
        _d_bright = float(_g_post.mean() - _g_pre.mean())
        _d_std = float(_g_post.std() - _g_pre.std())

        return {
            "id": _row["id"],
            "mae_r": float(_mae_r),
            "mae_g": float(_mae_g),
            "mae_b": float(_mae_b),
            "ssim": _ssim_val,
            "d_bright": _d_bright,
            "d_std": _d_std,
        }

    _rows_k5 = df_inv_clean.to_dict("records")
    with _TPE(max_workers=16) as _ex:
        _change_recs = list(tqdm(_ex.map(_compute_change_one, _rows_k5), total=len(_rows_k5), desc="Change Metrics"))

    df_change = pd.DataFrame(_change_recs)
    _change_path = Path(ARTIFACT_DIR) / "df_change.parquet"
    df_change.to_parquet(_change_path, index=False)

    # Simpan 3 thumbnail diff map (abs diff grayscale) untuk 3 sampel pertama
    _thumb_paths = []
    _fig_k5, _axes_k5 = plt.subplots(1, 3, figsize=(10, 3.3))
    for _i in range(3):
        _r = df_inv_clean.iloc[_i]
        _p1 = cv2.cvtColor(cv2.resize(cv2.imread(_r["path_pre"]), (256, 256)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        _p2 = cv2.cvtColor(cv2.resize(cv2.imread(_r["path_post"]), (256, 256)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        _gdiff = np.abs(_p2 - _p1).astype(np.uint8)
        _tp = Path(ARTIFACT_DIR) / f"diff_thumb_{_i+1}_{_r['id']}.png"
        Image.fromarray(_gdiff).save(_tp)
        _thumb_paths.append(str(_tp))
        _axes_k5[_i].imshow(_gdiff, cmap="gray", vmin=0, vmax=255)
        _axes_k5[_i].set_title(f"Diff {_r['id']}\nSSIM={df_change.iloc[_i]['ssim']:.3f}")
        _axes_k5[_i].axis("off")

    _fig_k5.tight_layout()
    print("3 thumbnail diff map tersimpan:", _thumb_paths)
    print("\ndf_change.describe():")
    print(df_change.describe().to_string())
    _fig_k5
    return (df_change,)


@app.cell(hide_code=True)
def M6_change_dist(mo):
    mo.md("""
    ## 6. Analisis Perubahan per Kelas (Jika Label Ada)

    - **Tujuan tahap:** Mengevaluasi apakah metrik perubahan (`SSIM` dan `MAE`) berbeda signifikan antar kelas fenomena (jika `HAS_LABELS=True`), atau menganalisis distribusi `SSIM` keseluruhan beserta KDE, kuartil, dan uji bimodalitas (jika `HAS_LABELS=False`) untuk mendeteksi adanya dua rezim perubahan (ringan vs masif).
    - **Output yang diharapkan:** Boxplot per kelas (bila berlabel) atau histogram `SSIM` + kurva KDE, nilai kuartil, jumlah puncak KDE, dan koefisien bimodalitas Sarle.
    - **Interpretasi:** Karena `HAS_LABELS = False`, dianalisis distribusi `SSIM` keseluruhan. Distribusi bersifat **unimodal condong kanan (*right-skewed*)** dengan 1 puncak KDE di `0.1023`, median `0.1456` (`Q1=0.0895`, `Q3=0.2239`), dan koefisien bimodalitas Sarle `0.5042` (`< 0.555`), sehingga **tidak bimodal**. Mayoritas pasangan berada pada rezim SSIM rendah (`< 0.25`), menunjukkan perubahan visual/radiometrik atau pergeseran yang kuat.
    """)
    return


@app.cell
def K6_change_dist(HAS_LABELS, df_change, df_labels, np, plt, scipy_stats):
    # Analisis distribusi SSIM/MAE per kelas jika HAS_LABELS=True,
    # atau plot histogram SSIM + KDE, kuartil, dan cek bimodalitas jika HAS_LABELS=False.
    from scipy.signal import find_peaks as _find_peaks

    plt.close("all")

    if HAS_LABELS and df_labels is not None:
        _id_col = "Id" if "Id" in df_labels.columns else "id"
        _dfc = df_change.merge(df_labels, left_on="id", right_on=_id_col, how="inner")
        _dfc["mae_mean"] = _dfc[["mae_r", "mae_g", "mae_b"]].mean(axis=1)
        _grp = _dfc.groupby("Phenomenon")[["ssim", "mae_mean", "mae_r", "mae_g", "mae_b"]].agg(["mean", "std"])
        print(_grp.to_string())
        _fig_k6, _ax_k6 = plt.subplots(figsize=(9, 4.5))
        _dfc.boxplot(column="ssim", by="Phenomenon", ax=_ax_k6)
        _ax_k6.set_title("Distribusi SSIM per Kelas Phenomenon")
        plt.suptitle("")
        _fig_k6.tight_layout()
    else:
        _ssim_vals = df_change["ssim"].dropna().values
        _q = np.quantile(_ssim_vals, [0.0, 0.25, 0.5, 0.75, 1.0])
        _skew = float(scipy_stats.skew(_ssim_vals))
        _kurt_pearson = float(scipy_stats.kurtosis(_ssim_vals, fisher=False))
        _sarle_bc = (_skew ** 2 + 1.0) / _kurt_pearson

        _xs = np.linspace(0.0, max(0.85, float(_ssim_vals.max())), 400)
        _kde = scipy_stats.gaussian_kde(_ssim_vals)
        _ys = _kde(_xs)
        _peaks, _ = _find_peaks(_ys, prominence=0.05 * _ys.max())
        _peak_xs = [round(float(_xs[_p]), 4) for _p in _peaks]
        is_ssim_bimodal = len(_peaks) >= 2

        print("Kuartil SSIM -> min: {:.4f}, Q1(25%): {:.4f}, median(50%): {:.4f}, Q3(75%): {:.4f}, max: {:.4f}".format(*_q))
        print(f"Skewness: {_skew:.4f} | Pearson Kurtosis: {_kurt_pearson:.4f} | Sarle Bimodality Coeff: {_sarle_bc:.4f} (ambang >0.555)")
        print(f"Puncak KDE terdeteksi ({len(_peak_xs)}): {_peak_xs} -> Bimodal (KDE >=2 puncak)? {is_ssim_bimodal}")

        _fig_k6, _ax_k6 = plt.subplots(figsize=(9, 4.5))
        _ax_k6.hist(_ssim_vals, bins=40, density=True, alpha=0.55, color="#3182ce", edgecolor="white", label="Histogram SSIM")
        _ax_k6.plot(_xs, _ys, color="#c53030", linewidth=2.2, label="KDE")
        for _px in _peak_xs:
            _ax_k6.axvline(_px, color="#2f855a", linestyle="--", linewidth=1.5, label=f"Puncak KDE: {_px:.3f}")
        _ax_k6.axvline(float(_q[2]), color="#1a202c", linestyle=":", linewidth=1.5, label=f"Median: {_q[2]:.3f}")
        _ax_k6.set_title("Distribusi SSIM Keseluruhan (3.180 Pasangan) + Kurva KDE")
        _ax_k6.set_xlabel("SSIM (Pre vs Post)")
        _ax_k6.set_ylabel("Densitas")
        _ax_k6.legend()
        _fig_k6.tight_layout()

    _fig_k6
    return (is_ssim_bimodal,)


@app.cell(hide_code=True)
def M7_registration(mo):
    mo.md("""
    ## 7. Deteksi Misalignment & Registrasi

    - **Tujuan tahap:** Mengukur pergeseran spasial (*misalignment*) antara citra `pre.jpg` dan `post.jpg` pada resolusi asli `1024x1024` menggunakan detektor fitur `ORB`, pencocokan `BFMatcher` dengan *Lowe's ratio test*, dan estimasi homografi `RANSAC` pada sampel acak 300 `Id` (`seed=42`).
    - **Output yang diharapkan:** DataFrame `df_reg` (disimpan ke `/marimo/Fenomena/df_registration.parquet`), rasio pasangan *registered* vs *unregistered*, kuartil magnitudo translasi (piksel), serta histogram distribusi magnitudo translasi.
    - **Interpretasi:** Dari 300 sampel, **229 (`76.3%`) berstatus `unregistered`** (match fitur tidak mencukupi), dan pada 71 pasangan `registered` seluruhnya memiliki magnitudo translasi `> 152 px` (median `874.2 px`). Temuan kunci: pasangan `pre`/`post` **bukan** citra yang ter-registrasi piksel-per-piksel secara ketat (banyak perubahan sudut pandang/skala/tampilan), sehingga operasi *pixel-wise difference* mentah saja rentan noise dan arsitektur berbasis fitur semantik/global (Siamese CNN/ViT atau CLIP/VLM) jauh lebih tepat daripada *pixel subtraction* murni.
    """)
    return


@app.cell
def K7_registration(
    ARTIFACT_DIR,
    Path,
    RANDOM_SEED,
    cv2,
    df_inv_clean,
    np,
    pd,
    plt,
    tqdm,
):
    # Sampling 300 Id secara acak (seed=42), ukur pergeseran spasial pre vs post
    # menggunakan ORB + BFMatcher ratio test + RANSAC homography, lalu simpan df_registration.parquet.
    plt.close("all")

    _sample_reg = df_inv_clean.sample(n=min(300, len(df_inv_clean)), random_state=RANDOM_SEED)
    _orb = cv2.ORB_create(nfeatures=1000)
    _bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    _reg_records = []
    for _row in tqdm(_sample_reg.to_dict("records"), desc="ORB+RANSAC Reg"):
        _g1 = cv2.imread(_row["path_pre"], cv2.IMREAD_GRAYSCALE)
        _g2 = cv2.imread(_row["path_post"], cv2.IMREAD_GRAYSCALE)
        _kp1, _des1 = _orb.detectAndCompute(_g1, None)
        _kp2, _des2 = _orb.detectAndCompute(_g2, None)

        _status = "unregistered"
        _n_good = 0
        _n_inliers = 0
        _tx = np.nan
        _ty = np.nan
        _tmag = np.nan

        if _des1 is not None and _des2 is not None and len(_kp1) >= 10 and len(_kp2) >= 10:
            _knn = _bf.knnMatch(_des1, _des2, k=2)
            _good = [_m for _pair in _knn if len(_pair) == 2 for _m, _n in [_pair] if _m.distance < 0.75 * _n.distance]
            _n_good = len(_good)
            if _n_good >= 10:
                _src_pts = np.float32([_kp1[_m.queryIdx].pt for _m in _good]).reshape(-1, 1, 2)
                _dst_pts = np.float32([_kp2[_m.trainIdx].pt for _m in _good]).reshape(-1, 1, 2)
                _H, _mask = cv2.findHomography(_src_pts, _dst_pts, cv2.RANSAC, 5.0)
                if _H is not None and _mask is not None:
                    _n_inliers = int(_mask.ravel().sum())
                    if _n_inliers >= 6:
                        _tx = float(_H[0, 2])
                        _ty = float(_H[1, 2])
                        _tmag = float(np.hypot(_tx, _ty))
                        _status = "registered"

        _reg_records.append({
            "id": _row["id"],
            "status": _status,
            "n_good_matches": _n_good,
            "n_inliers": _n_inliers,
            "tx": _tx,
            "ty": _ty,
            "translation_mag": _tmag,
        })

    df_reg = pd.DataFrame(_reg_records)
    df_reg.to_parquet(Path(ARTIFACT_DIR) / "df_registration.parquet", index=False)

    _vc_reg = df_reg["status"].value_counts().to_dict()
    _reg_ok = df_reg[df_reg["status"] == "registered"]["translation_mag"].dropna()
    print("Status registrasi (300 sampel):", _vc_reg)
    if len(_reg_ok) > 0:
        print("Statistik translation_mag (px, seluruh registered):")
        print(_reg_ok.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
        _n_shift_gt20 = int((_reg_ok > 20).sum())
        print(f"Pasangan registered dengan translasi > 20 px: {_n_shift_gt20} / {len(_reg_ok)}")

    _fig_k7, _ax_k7 = plt.subplots(figsize=(9, 4.2))
    if len(_reg_ok) > 0:
        _clip_hi = float(min(max(_reg_ok.quantile(0.95) * 1.2, 20.0), 500.0))
        _clipped = _reg_ok.clip(upper=_clip_hi)
        _ax_k7.hist(_clipped, bins=30, color="#805ad5", edgecolor="white")
        _ax_k7.axvline(float(_reg_ok.median()), color="#e53e3e", linestyle="--", label=f"Median: {_reg_ok.median():.2f} px")
        _ax_k7.set_title(f"Distribusi Translation Magnitude ORB+RANSAC (n={len(_reg_ok)} registered, clip={_clip_hi:.0f} px)")
        _ax_k7.set_xlabel("Translation Magnitude (piksel, 1024x1024)")
        _ax_k7.set_ylabel("Jumlah Pasangan")
        _ax_k7.legend()
    else:
        _ax_k7.text(0.5, 0.5, "Semua sampel berstatus unregistered", ha="center", va="center")
        _ax_k7.set_title("Distribusi Translation Magnitude ORB+RANSAC")
    _fig_k7.tight_layout()
    _fig_k7
    return (df_reg,)


@app.cell(hide_code=True)
def M8_color(mo):
    mo.md("""
    ## 8. Analisis Warna & Pencahayaan

    - **Tujuan tahap:** Mengukur pergeseran distribusi warna dan pencahayaan antara `pre.jpg` dan `post.jpg` (mean RGB, std RGB, histogram 32-bin ternormalisasi, serta *KL divergence* per kanal) pada 500 sampel acak (`seed=42`) untuk menentukan rentang augmentasi `ColorJitter` yang realistis.
    - **Output yang diharapkan:** DataFrame `df_color` (disimpan ke `/marimo/Fenomena/color_stats.parquet`), rata-rata *KL divergence* per kanal R/G/B, dan plot kurva histogram rata-rata 3 kanal dalam 2 baris (`Pre` vs `Post`).
    - **Interpretasi:** Rata-rata global RGB antar `pre` (`90.0 / 91.7 / 82.5`) dan `post` (`90.5 / 92.1 / 84.1`) hampir identik, namun rata-rata *KL divergence* per pasangan cukup tinggi (`R=1.88`, `G=2.01`, `B=2.06`). Artinya terjadi pergeseran warna/iluminasi/asap/awan yang besar di tingkat per pasangan, sehingga `ColorJitter` independen (`brightness~0.3, contrast~0.3, saturation~0.25`) saat training sangat realistis.
    """)
    return


@app.cell
def K8_color(
    ARTIFACT_DIR,
    Path,
    RANDOM_SEED,
    cv2,
    df_inv_clean,
    np,
    pd,
    plt,
    scipy_stats,
    tqdm,
):
    # Sampling 500 Id (seed=42), hitung mean/std RGB, histogram 32-bin ternormalisasi,
    # serta KL divergence pre vs post per kanal, lalu plot 3 kanal x 2 baris dan simpan color_stats.parquet.
    plt.close("all")

    _sample_col = df_inv_clean.sample(n=min(500, len(df_inv_clean)), random_state=RANDOM_SEED)
    _eps = 1e-8
    _hist_pre_acc = np.zeros((3, 32), dtype=np.float64)
    _hist_post_acc = np.zeros((3, 32), dtype=np.float64)
    _col_records = []

    for _row in tqdm(_sample_col.to_dict("records"), desc="Color Stats (500)"):
        _p1 = cv2.cvtColor(cv2.resize(cv2.imread(_row["path_pre"]), (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _p2 = cv2.cvtColor(cv2.resize(cv2.imread(_row["path_post"]), (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

        _rec = {"id": _row["id"]}
        for _ci, _ch in enumerate(("r", "g", "b")):
            _c1 = _p1[:, :, _ci]
            _c2 = _p2[:, :, _ci]
            _rec[f"mean_{_ch}_pre"] = float(_c1.mean())
            _rec[f"std_{_ch}_pre"] = float(_c1.std())
            _rec[f"mean_{_ch}_post"] = float(_c2.mean())
            _rec[f"std_{_ch}_post"] = float(_c2.std())

            _h1, _ = np.histogram(_c1, bins=32, range=(0, 256))
            _h2, _ = np.histogram(_c2, bins=32, range=(0, 256))
            _h1 = _h1.astype(np.float64) + _eps
            _h2 = _h2.astype(np.float64) + _eps
            _h1 /= _h1.sum()
            _h2 /= _h2.sum()
            _hist_pre_acc[_ci] += _h1
            _hist_post_acc[_ci] += _h2
            _rec[f"kl_{_ch}"] = float(scipy_stats.entropy(_h1, _h2))

        _col_records.append(_rec)

    _hist_pre_mean = _hist_pre_acc / len(_col_records)
    _hist_post_mean = _hist_post_acc / len(_col_records)

    df_color = pd.DataFrame(_col_records)
    df_color.to_parquet(Path(ARTIFACT_DIR) / "color_stats.parquet", index=False)

    _kl_means = df_color[["kl_r", "kl_g", "kl_b"]].mean().to_dict()
    _rgb_summary = df_color[[
        "mean_r_pre", "mean_g_pre", "mean_b_pre",
        "mean_r_post", "mean_g_post", "mean_b_post",
        "std_r_pre", "std_g_pre", "std_b_pre",
        "std_r_post", "std_g_post", "std_b_post",
    ]].mean()
    print("Rata-rata KL Divergence (Pre || Post) 32-bin:", {k: round(float(v), 4) for k, v in _kl_means.items()})
    print("\nRata-rata Mean & Std RGB (Pre vs Post):")
    print(_rgb_summary.round(2).to_string())

    _bin_centers = np.linspace(4, 252, 32)
    _colors = ["#e53e3e", "#38a169", "#3182ce"]
    _ch_names = ["Red (R)", "Green (G)", "Blue (B)"]
    _fig_k8, _axes_k8 = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for _ci in range(3):
        _axes_k8[0, _ci].plot(_bin_centers, _hist_pre_mean[_ci], color=_colors[_ci], linewidth=2)
        _axes_k8[0, _ci].fill_between(_bin_centers, _hist_pre_mean[_ci], color=_colors[_ci], alpha=0.2)
        _axes_k8[0, _ci].set_title(f"PRE — {_ch_names[_ci]}")
        _axes_k8[0, _ci].grid(alpha=0.25)

        _axes_k8[1, _ci].plot(_bin_centers, _hist_post_mean[_ci], color=_colors[_ci], linewidth=2, linestyle="--")
        _axes_k8[1, _ci].fill_between(_bin_centers, _hist_post_mean[_ci], color=_colors[_ci], alpha=0.2)
        _axes_k8[1, _ci].set_title(f"POST — {_ch_names[_ci]} (KL={_kl_means[f'kl_{_ch_names[_ci][0].lower()}']:.3f})")
        _axes_k8[1, _ci].set_xlabel("Intensitas Piksel (0-255)")
        _axes_k8[1, _ci].grid(alpha=0.25)

    _axes_k8[0, 0].set_ylabel("Probabilitas Rata-rata")
    _axes_k8[1, 0].set_ylabel("Probabilitas Rata-rata")
    _fig_k8.suptitle("Kurva Histogram Warna Rata-rata (32-bin) Pre vs Post (n=500)", fontsize=13)
    _fig_k8.tight_layout()
    _fig_k8
    return (df_color,)


@app.cell(hide_code=True)
def M9_grid(mo):
    mo.md("""
    ## 9. Visualisasi Sampel Perubahan

    - **Tujuan tahap:** Memverifikasi secara visual sinyal perubahan pada 8 `Id` dengan `SSIM` terendah (perubahan paling masif) dan 8 `Id` dengan `SSIM` tertinggi (perubahan paling minimal) dalam format grid `4x4` (baris 1–2: perubahan masif; baris 3–4: perubahan minimal; tiap sel menampilkan komposit `[pre | post | diff]`).
    - **Output yang diharapkan:** Tabel ringkas 16 `Id` ekstrem beserta skor `SSIM` dan figur grid `4x4` yang disimpan ke `/marimo/Fenomena/sample_grid.png`.
    - **Interpretasi:** Grid `/marimo/Fenomena/sample_grid.png` memverifikasi kontras yang jelas antara kelompok masif (`SSIM` `0.0064–0.0185`, MAE `~61–108`) dan kelompok minimal (`SSIM` `0.6115–0.8007`, MAE `~10–40`). Pada sampel `861244f6031e` terlihat `mae_r == mae_g == mae_b` (indikasi citra grayscale tersimpan sebagai RGB).
    """)
    return


@app.cell
def K9_grid(ARTIFACT_DIR, Path, cv2, df_change, df_inv_clean, np, pd, plt):
    # Pilih 8 Id dengan SSIM terendah dan 8 Id dengan SSIM tertinggi,
    # susun grid 4x4 komposit [pre | post | diff] dan simpan ke /marimo/Fenomena/sample_grid.png.
    plt.close("all")

    _df_sorted = df_change.merge(df_inv_clean[["id", "path_pre", "path_post"]], on="id").sort_values("ssim")
    _low8 = _df_sorted.head(8).copy()
    _low8["grup"] = "masif_ssim_rendah"
    _high8 = _df_sorted.tail(8).iloc[::-1].copy()
    _high8["grup"] = "minimal_ssim_tinggi"
    _sel16 = pd.concat([_low8, _high8], ignore_index=True)

    print("16 sampel ekstrem (8 terendah + 8 tertinggi):")
    print(_sel16[["grup", "id", "ssim", "mae_r", "mae_g", "mae_b", "d_bright"]].to_string(index=False))

    def _make_strip(_row, _sz=160):
        _p1 = cv2.cvtColor(cv2.resize(cv2.imread(_row["path_pre"]), (_sz, _sz), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _p2 = cv2.cvtColor(cv2.resize(cv2.imread(_row["path_post"]), (_sz, _sz), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        _df = np.abs(_p2.astype(np.int16) - _p1.astype(np.int16)).mean(axis=2).astype(np.uint8)
        _df_rgb = cv2.applyColorMap(_df, cv2.COLORMAP_INFERNO)
        _df_rgb = cv2.cvtColor(_df_rgb, cv2.COLOR_BGR2RGB)
        _sep = np.full((_sz, 3, 3), 255, dtype=np.uint8)
        return np.hstack([_p1, _sep, _p2, _sep, _df_rgb])

    _fig_k9, _axes_k9 = plt.subplots(4, 4, figsize=(18, 8.5))
    for _idx, _row in _sel16.iterrows():
        _r_i, _c_i = divmod(_idx, 4)
        _ax = _axes_k9[_r_i, _c_i]
        _ax.imshow(_make_strip(_row))
        _tag = "MASIF" if _r_i < 2 else "MINIMAL"
        _ax.set_title(f"[{_tag}] {_row['id']} | SSIM={_row['ssim']:.3f}", fontsize=9)
        _ax.axis("off")

    _fig_k9.suptitle("Grid 4x4 Pasangan [Pre | Post | Diff] — Baris 1-2: SSIM Terendah (Masif) | Baris 3-4: SSIM Tertinggi (Minimal)", fontsize=12)
    _fig_k9.tight_layout()
    _grid_path = Path(ARTIFACT_DIR) / "sample_grid.png"
    _fig_k9.savefig(_grid_path, dpi=150, bbox_inches="tight")
    print("Sample grid tersimpan ke:", str(_grid_path))
    _fig_k9
    return


@app.cell(hide_code=True)
def M10_preprocess(mo):
    mo.md("""
    ## 10. Pipeline Preprocessing (Artefak)

    - **Tujuan tahap:** Mendefinisikan fungsi preprocessing dan augmentasi pasangan citra `pre`/`post` yang konsisten dan siap digunakan pada notebook training terpisah (`load_pair`, `build_diff_channels` untuk fusion 9-kanal, `augment_pair` dengan transformasi geometrik identik dan `ColorJitter` independen, serta `compute_ndvi_if_available`). Karena `HAS_LABELS=False`, tahap ini tidak melakukan training model.
    - **Output yang diharapkan:** Verifikasi unit-test pada 3 `Id` acak (cek `assert` shape `(3, 384, 384)`, gabungan 9-kanal `(9, 384, 384)`, `dtype=torch.float32`, dan hasil `compute_ndvi_if_available`).
    - **Interpretasi:** Seluruh unit-test pada 3 `Id` acak (`4f45ca15185b`, `767037fc8705`, `3c622cfc60c0`) lolos 100%: `load_pair` dan `augment_pair` menghasilkan tensor `(3, 384, 384)` `float32`, `build_diff_channels` membentuk fitur 9-kanal `(9, 384, 384)`, dan `compute_ndvi_if_available` mengembalikan `None` karena citra input bertipe JPEG RGB 3-kanal tanpa kanal NIR.
    """)
    return


@app.cell
def K10_preprocess(
    DATA_ROOT,
    Image,
    Path,
    RANDOM_SEED,
    df_inv_clean,
    pd,
    random,
):
    # Definisi fungsi pipeline preprocessing pasangan pre/post (load_pair, build_diff_channels,
    # augment_pair, compute_ndvi_if_available) dan uji assert shape & dtype pada 3 Id acak.
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms import ColorJitter

    def load_pair(sample_id, size=384):
        """Load pre.jpg & post.jpg untuk sample_id, resize ke (size, size), kembalikan tensor float32 RGB [0, 1]."""
        _p_pre = Path(DATA_ROOT) / str(sample_id) / "pre.jpg"
        _p_post = Path(DATA_ROOT) / str(sample_id) / "post.jpg"
        with Image.open(_p_pre) as _im1:
            _im1 = _im1.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            _t1 = TF.to_tensor(_im1).to(torch.float32)
        with Image.open(_p_post) as _im2:
            _im2 = _im2.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            _t2 = TF.to_tensor(_im2).to(torch.float32)
        return _t1, _t2

    def build_diff_channels(pre, post):
        """Bangun 3 kanal diff (abs_diff, gradient_x, gradient_y) untuk input fusion 9-kanal."""
        _abs_map = torch.abs(post - pre).mean(dim=0, keepdim=True)  # (1, H, W)
        _gx = torch.zeros_like(_abs_map)
        _gy = torch.zeros_like(_abs_map)
        _gx[:, :, 1:] = _abs_map[:, :, 1:] - _abs_map[:, :, :-1]
        _gy[:, 1:, :] = _abs_map[:, 1:, :] - _abs_map[:, :-1, :]
        return torch.cat([_abs_map, _gx, _gy], dim=0).to(torch.float32)

    def augment_pair(pre, post, mode="train"):
        """Terapkan flip/rotasi identik ke pre & post, serta ColorJitter independen pada mode train."""
        if mode != "train":
            return pre.clone(), post.clone()
        _p1, _p2 = pre.clone(), post.clone()
        if random.random() < 0.5:
            _p1 = TF.hflip(_p1)
            _p2 = TF.hflip(_p2)
        if random.random() < 0.5:
            _p1 = TF.vflip(_p1)
            _p2 = TF.vflip(_p2)
        _k_rot = random.choice([0, 90, 180, 270])
        if _k_rot != 0:
            _p1 = TF.rotate(_p1, _k_rot)
            _p2 = TF.rotate(_p2, _k_rot)
        _jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.04)
        _p1 = _jitter(_p1)
        _p2 = _jitter(_p2)
        return _p1.clamp(0.0, 1.0), _p2.clamp(0.0, 1.0)

    def compute_ndvi_if_available(pre):
        """Hitung NDVI (NIR-Red)/(NIR+Red) jika tensor memiliki >=4 kanal multispektral, selain itu None."""
        if pre.ndim == 3 and pre.shape[0] >= 4:
            _red = pre[0]
            _nir = pre[3]
            return (_nir - _red) / (_nir + _red + 1e-6)
        return None

    _test_ids_k10 = df_inv_clean.sample(n=3, random_state=RANDOM_SEED)["id"].tolist()
    _k10_rows = []
    for _tid in _test_ids_k10:
        _pre_t, _post_t = load_pair(_tid, size=384)
        _pre_aug, _post_aug = augment_pair(_pre_t, _post_t, mode="train")
        _diff_t = build_diff_channels(_pre_aug, _post_aug)
        _fused9 = torch.cat([_pre_aug, _post_aug, _diff_t], dim=0)
        _ndvi = compute_ndvi_if_available(_pre_t)

        assert _pre_t.shape == (3, 384, 384) and _pre_t.dtype == torch.float32
        assert _post_t.shape == (3, 384, 384) and _post_t.dtype == torch.float32
        assert _diff_t.shape == (3, 384, 384) and _diff_t.dtype == torch.float32
        assert _fused9.shape == (9, 384, 384) and _fused9.dtype == torch.float32
        assert _ndvi is None

        _k10_rows.append({
            "id": _tid,
            "pre_shape": tuple(_pre_aug.shape),
            "post_shape": tuple(_post_aug.shape),
            "diff_shape": tuple(_diff_t.shape),
            "fused9_shape": tuple(_fused9.shape),
            "dtype": str(_fused9.dtype),
            "ndvi": str(_ndvi),
        })

    _df_k10_test = pd.DataFrame(_k10_rows)
    print("Uji pipeline preprocessing pada 3 Id acak BERHASIL:")
    print(_df_k10_test.to_string(index=False))
    _df_k10_test
    return ColorJitter, TF, torch


@app.cell(hide_code=True)
def M11_save(mo):
    mo.md("""
    ## 11. Simpan Artefak & Metadata

    - **Tujuan tahap:** Menyimpan ulang seluruh DataFrame hasil EDA ke format Parquet (`df_inventory.parquet`, `df_change.parquet`, `df_registration.parquet`, `color_stats.parquet`), memperbarui metadata JSON (`/marimo/Fenomena/_meta.json`), serta menulis laporan ringkas `/marimo/Fenomena/EXP_SUMMARY.md` agar notebook training/inference berikutnya dapat langsung memuat artefak secara idempoten.
    - **Output yang diharapkan:** Tabel verifikasi keberadaan & ukuran seluruh file artefak di `/marimo/Fenomena/` serta konfirmasi penulisan `_meta.json` dan `EXP_SUMMARY.md`.
    - **Interpretasi:** Seluruh **7** artefak utama (`df_inventory.parquet` 402 KB, `df_change.parquet` 194 KB, `df_registration.parquet` 11.7 KB, `color_stats.parquet` 79.3 KB, `sample_grid.png` 3.44 MB, `_meta.json` 1.18 KB, `EXP_SUMMARY.md` 2.28 KB) ditambah 3 thumbnail diff map berhasil tersimpan lengkap di `/marimo/Fenomena/`.
    """)
    return


@app.cell
def K11_save(
    ARTIFACT_DIR,
    HAS_LABELS,
    Path,
    df_change,
    df_color,
    df_integrity_issues,
    df_inv,
    df_inv_clean,
    df_labels,
    df_reg,
    is_ssim_bimodal,
    json,
    pd,
    torch,
):
    # Menyimpan semua DataFrame ke Parquet, memperbarui _meta.json,
    # dan menulis ringkasan temuan lengkap ke /marimo/Fenomena/EXP_SUMMARY.md.
    _art_dir = Path(ARTIFACT_DIR)
    df_inv.to_parquet(_art_dir / "df_inventory.parquet", index=False)
    df_change.to_parquet(_art_dir / "df_change.parquet", index=False)
    df_reg.to_parquet(_art_dir / "df_registration.parquet", index=False)
    df_color.to_parquet(_art_dir / "color_stats.parquet", index=False)

    _mean_ssim = float(df_change["ssim"].mean())
    _med_ssim = float(df_change["ssim"].median())
    _mean_mae = df_change[["mae_r", "mae_g", "mae_b"]].mean().to_dict()
    _unreg_n = int((df_reg["status"] == "unregistered").sum())
    _unreg_pct = float(_unreg_n / len(df_reg) * 100.0)
    _reg_med_t = float(df_reg.loc[df_reg["status"] == "registered", "translation_mag"].median())
    _kl_mean = df_color[["kl_r", "kl_g", "kl_b"]].mean().to_dict()

    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0

    _rek_arsitektur = (
        "Supervised Siamese Network (EfficientNet-B4, abs-diff & concat fusion, Stratified 5-Fold, Macro F1)"
        if HAS_LABELS else
        "Zero-shot VLM/CLIP (SigLIP / OpenCLIP ViT-L/H + VLM Qwen2-VL/InternVL untuk pseudo-label terkalibrasi) -> Student Siamese Network (EfficientNet-B4/ConvNeXt) dengan pseudo-label berkeyakinan tinggi"
    )

    _meta_final = {
        "n_ids": int(len(df_inv)),
        "n_valid_clean": int(len(df_inv_clean)),
        "n_integrity_issues": int(len(df_integrity_issues)),
        "ada_label": bool(HAS_LABELS),
        "distribusi_kelas": df_labels["Phenomenon"].value_counts().to_dict() if (HAS_LABELS and df_labels is not None) else None,
        "mean_ssim": round(_mean_ssim, 6),
        "median_ssim": round(_med_ssim, 6),
        "ssim_bimodal": bool(is_ssim_bimodal),
        "mean_mae_rgb": {k: round(float(v), 4) for k, v in _mean_mae.items()},
        "unregistered_ratio": round(_unreg_pct / 100.0, 4),
        "registered_median_translation_px": round(_reg_med_t, 2),
        "kl_divergence_mean": {k: round(float(v), 4) for k, v in _kl_mean.items()},
        "gpu_device": _gpu_name,
        "gpu_vram_gb": _gpu_vram,
        "rekomendasi_arsitektur": _rek_arsitektur,
        "artefak": {
            "inventory": str(_art_dir / "df_inventory.parquet"),
            "change": str(_art_dir / "df_change.parquet"),
            "registration": str(_art_dir / "df_registration.parquet"),
            "color_stats": str(_art_dir / "color_stats.parquet"),
            "sample_grid": str(_art_dir / "sample_grid.png"),
            "meta": str(_art_dir / "_meta.json"),
            "exp_summary": str(_art_dir / "EXP_SUMMARY.md"),
        },
    }

    with open(_art_dir / "_meta.json", "w", encoding="utf-8") as _f:
        json.dump(_meta_final, _f, indent=2, ensure_ascii=False)

    _exp_md = f"""# EXP_SUMMARY — EDA & Preprocessing Kompetisi Fenomena Bencana

    ## 1. Status Data & Verifikasi Label
    - **Total Id:** {len(df_inv):,} pasangan (`pre.jpg` + `post.jpg`), seluruhnya valid (`0` missing, `0` corrupt, `0` duplikat MD5).
    - **Resolusi & Mode Warna:** Seragam `1024x1024` RGB (`3.180/3.180`), tanpa kanal multispektral NIR (`NDVI = None`).
    - **Status Label:** **TIDAK ADA LABEL TRAINING** (`HAS_LABELS = False`). Satu-satunya CSV yang tersedia adalah `/marimo/sample_submission.csv` (`3.180` baris, seluruhnya placeholder `'Earthquake'`).

    ## 2. Temuan Kunci EDA
    1. **Ketiadaan Label Ground-Truth:** Seluruh 3.180 sampel berstatus uji sehingga supervised training langsung tidak dapat dilakukan tanpa pseudo-labeling.
    2. **Sinyal Perubahan Global (SSIM & MAE):** Rata-rata `SSIM = {_mean_ssim:.4f}` (median `{_med_ssim:.4f}`, rentang `0.0064–0.8007`), dengan rata-rata MAE RGB (`R={_mean_mae['mae_r']:.2f}`, `G={_mean_mae['mae_g']:.2f}`, `B={_mean_mae['mae_b']:.2f}` pada skala 0–255).
    3. **Distribusi SSIM Unimodal:** Uji KDE menunjukkan 1 puncak dominan di `0.1023` (Sarle BC `0.5042 < 0.555`) — **tidak bimodal**.
    4. **Misalignment Spasial Tinggi:** Dari 300 sampel uji `ORB+RANSAC`, **{_unreg_n}/300 ({_unreg_pct:.1f}%) berstatus `unregistered`**, dan pasangan `registered` memiliki median translasi **{_reg_med_t:.1f} px** (min `152.8 px`). Artinya pasangan citra bukan tile satelit ter-registrasi piksel-per-piksel.
    5. **Pergeseran Warna/Pencahayaan Kuat:** Rata-rata *KL divergence* 32-bin antara `pre` dan `post` mencapai `R={_kl_mean['kl_r']:.2f}`, `G={_kl_mean['kl_g']:.2f}`, `B={_kl_mean['kl_b']:.2f}`, mengindikasikan variasi iluminasi, asap, dan awan yang signifikan.

    ## 3. Rekomendasi Strategi & Estimasi Compute
    - **Hardware Terdeteksi:** `{_gpu_name}` (`{_gpu_vram} GB VRAM`).
    - **Strategi Utama:** Zero-shot CLIP/SigLIP (membandingkan embedding `post` dan `post - pre` terhadap prompt 6 kelas) + pseudo-labeling terkalibrasi menggunakan VLM (mis. Qwen2-VL/Qwen2.5-VL pada citra gabungan `[pre | post]`), diikuti *self-training* Siamese Network (`EfficientNet-B4` / `ConvNeXt`, fusion *concat* + *abs-diff*) menggunakan Stratified 5-Fold pada pseudo-label berkeyakinan tinggi.
    - **Kandidat Drop:** `0` dari audit integritas (`df_integrity_issues` kosong), namun `~76.3%` pasangan `unregistered` pada ORB menandakan fusion *pixel-difference* mentah harus dilengkapi fitur semantik/global.
    """

    with open(_art_dir / "EXP_SUMMARY.md", "w", encoding="utf-8") as _f:
        _f.write(_exp_md)

    _expected_files = [
        "df_inventory.parquet",
        "df_change.parquet",
        "df_registration.parquet",
        "color_stats.parquet",
        "sample_grid.png",
        "_meta.json",
        "EXP_SUMMARY.md",
    ]
    _artifact_check = pd.DataFrame([
        {
            "file": _fn,
            "path": str(_art_dir / _fn),
            "exists": (_art_dir / _fn).is_file(),
            "size_bytes": (_art_dir / _fn).stat().st_size if (_art_dir / _fn).is_file() else 0,
        }
        for _fn in _expected_files
    ])
    print(_artifact_check.to_string(index=False))
    _artifact_check
    return


@app.cell(hide_code=True)
def M12_summary(mo):
    mo.md("""
    ## 12. Ringkasan & Rekomendasi Strategi

    - **Tujuan tahap:** Merangkum seluruh temuan EDA, status ketersediaan label, karakteristik distribusi `SSIM`/registrasi/warna, kandidat `Id` bermasalah, rekomendasi arsitektur, serta estimasi waktu eksekusi pada GPU Molab.
    - **Output yang diharapkan:** Ringkasan otomatis yang dibaca langsung dari `/marimo/Fenomena/_meta.json` dan notifikasi toast penyelesaian EDA.

    ### Temuan Utama & Rekomendasi
    1. **Ketersediaan Label:** **TIDAK ADA LABEL TRAINING** (`HAS_LABELS = False`). Seluruh 3.180 sampel (`pre.jpg` + `post.jpg`) berstatus uji; `/marimo/sample_submission.csv` hanya berisi template dengan nilai dummy `'Earthquake'` pada seluruh 3.180 baris.
    2. **Rekomendasi Arsitektur:** Jangan melatih *supervised* Siamese Network dari nol tanpa label. Gunakan:
       - **Tahap 1 (Zero-Shot Baseline):** Ekstraksi embedding `CLIP` / `SigLIP` (`open_clip` / `transformers`) untuk fitur `post` dan selisih semantik `(e_post - e_pre)` terhadap prompt terkurasi 6 kelas (`Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion`).
       - **Tahap 2 (VLM Pseudo-Labeling):** Gunakan VLM (mis. `Qwen2.5-VL` / `Qwen2-VL`) pada komposit `[pre | post]` untuk menghasilkan *pseudo-label* beserta skor keyakinan & *calibration* distribusi 6 kelas.
       - **Tahap 3 (Student Siamese Network):** Latih *Siamese Network* (`EfficientNet-B4` / `ConvNeXt`) menggunakan *pseudo-label* berkeyakinan tinggi dengan fusion gabungan **Concatenation + Absolute Difference** (karena `76.3%` pasangan `unregistered` secara piksel, sehingga fitur global/semantik jauh lebih penting daripada *pixel subtraction* mentah) dan validasi Stratified 5-Fold (Macro F1).
    3. **Distribusi SSIM:** **Tidak bimodal** — distribusi unimodal condong kanan dengan puncak KDE di `0.1023`, median `0.1456` (`Q1=0.0895`, `Q3=0.2239`), dan koefisien bimodalitas Sarle `0.5042` (`< 0.555`).
    4. **Kandidat Drop:** **0 `Id`** untuk di-drop dari sisi integritas (`0` corrupt, `0` missing, `0` mismatch dimensi `1024x1024`, `0` non-RGB, `0` duplikat MD5). Meskipun `76.3%` (229/300) sampel berstatus `unregistered` pada ORB+RANSAC dan pasangan `registered` bergeser median `874.2 px`, seluruh 3.180 `Id` wajib tetap diprediksi untuk submission.
    5. **Estimasi Waktu Training di GPU Molab (`NVIDIA RTX PRO 6000 Blackwell`, `101.98 GB VRAM`):**
       - Ekstraksi fitur zero-shot CLIP/SigLIP (6.360 citra, FP16/BF16): **~15–30 detik**.
       - VLM *pseudo-labeling* lokal (mis. Qwen2.5-VL-7B/32B, 3.180 pasangan): **~10–25 menit**.
       - Training Siamese EfficientNet-B4 (384x384, Stratified 5-Fold × 10 epoch, AMP BF16): **~2–4 menit per fold (~10–20 menit total 5-fold)**.
    """)
    return


@app.cell
def K12_summary(ARTIFACT_DIR, Path, json, mo):
    # Membaca ringkasan akhir dari /marimo/Fenomena/_meta.json,
    # mencetak status akhir terstruktur, dan mengirim toast konfirmasi ke UI marimo.
    with open(Path(ARTIFACT_DIR) / "_meta.json", "r", encoding="utf-8") as _f:
        _final_meta = json.load(_f)

    print("=== RINGKASAN AKHIR EDA FENOMENA (_meta.json) ===")
    print(json.dumps(_final_meta, indent=2, ensure_ascii=False))

    mo.status.toast(title="Fenomena EDA Done", description="EDA selesai! Cek EXP_SUMMARY.md")
    mo.callout(
        mo.md(f"**EDA & Preprocessing Selesai:** `{_final_meta['n_valid_clean']}/{_final_meta['n_ids']}` pasang valid | `ada_label={_final_meta['ada_label']}` | `mean_ssim={_final_meta['mean_ssim']}` | GPU: `{_final_meta['gpu_device']}` (`{_final_meta['gpu_vram_gb']} GB`)"),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def M13_install(mo):
    mo.md("""
    ## 13. Instalasi Library Modeling

    - **Tujuan tahap:** Mengatasi temuan EDA `(T3)` dengan memasang library yang belum tersedia (`open_clip_torch`, `timm`, `bitsandbytes`, `qwen-vl-utils`, `accelerate`, `sentencepiece`, serta memastikan `scikit-learn` dan `huggingface_hub`) agar dapat menjalankan *model hunting* backbone satelit/multimodal dan *pseudo-labeling* dengan VLM (`Qwen2.5-VL`).
    - **Output yang diharapkan:** Konfirmasi versi `open_clip`, `timm`, dan `transformers`, spesifikasi GPU (`NVIDIA RTX PRO 6000 Blackwell`, VRAM ~102 GB), waktu instalasi, serta notifikasi toast `"Library modeling siap"`.
    - **Interpretasi:** Seluruh paket berhasil dipasang dalam `9.42` detik (`open_clip 4.2.2`, `timm 1.0.31`, `transformers 5.20.1`, `torch 2.11.0+cu130`) di atas GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`101.98 GB VRAM`).
    """)
    return


@app.cell
def K13_install(mo, pd, torch):
    # Pemasangan library modeling secara idempoten (open_clip_torch, timm, bitsandbytes,
    # qwen-vl-utils, accelerate, sentencepiece, scikit-learn) dan verifikasi versi serta GPU.
    import sys as _sys
    import time as _time
    import subprocess as _subp
    import importlib as _imp

    mo.status.toast(title="Fase 2B Dimulai", description="Memasang library modeling & verifikasi GPU")
    _t0 = _time.time()

    _pkgs = [
        ("open_clip", "open_clip_torch"),
        ("timm", "timm"),
        ("bitsandbytes", "bitsandbytes"),
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("sklearn", "scikit-learn"),
        ("huggingface_hub", "huggingface_hub"),
    ]

    _missing = []
    for _mod_name, _pip_name in _pkgs:
        try:
            _imp.import_module(_mod_name)
        except ImportError:
            _missing.append(_pip_name)

    if _missing:
        print("Memasang paket yang belum tersedia:", _missing)
        try:
            _subp.check_call([_sys.executable, "-m", "pip", "install", "--quiet", *_missing])
        except Exception:
            _subp.check_call(["uv", "pip", "install", "--quiet", "--python", _sys.executable, *_missing])
    else:
        print("Semua library modeling sudah terpasang (idempoten skip).")

    import open_clip
    import timm
    import transformers

    _elapsed = _time.time() - _t0
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0

    _env_df = pd.DataFrame([{
        "open_clip": getattr(open_clip, "__version__", "installed"),
        "timm": timm.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu_device": _gpu_name,
        "vram_gb": _vram_gb,
        "install_time_sec": round(_elapsed, 2),
    }])
    print(_env_df.to_string(index=False))
    mo.status.toast(title="Library modeling siap", description=f"Selesai dalam {_elapsed:.1f} detik")
    _env_df
    return open_clip, timm, transformers


@app.cell(hide_code=True)
def M14_hunting(mo):
    mo.md("""
    ## 14. Model Hunting: Memilih Backbone Terbaik untuk Citra Satelit

    - **Tujuan tahap:** Mengevaluasi kandidat backbone spesialis citra satelit & representasi visual (`RemoteCLIP`, `SkyCLIP`, `DINOv2-large`, `SigLIP-large`; `Prithvi` dilewati karena data hanya RGB 3-kanal tanpa multispektral) pada 300 sampel acak (`seed=42`). CLIP standar tidak dipakai sebagai final karena kurang spesifik untuk sudut pandang udara/satelit.
    - **Kriteria pemilihan:** Pada embedding selisih `e_diff = norm(e_post - e_pre)` dihitung `silhouette_score` dan `davies_bouldin_score` (KMeans `k=6`), serta `entropy` distribusi prediksi argmax *cosine similarity* terhadap 6 prompt teks spesifik bencana satelit. TOP-2 dipilih berdasarkan kombinasi `silhouette` tertinggi dan `entropy` terendah (tanpa kolaps ke satu kelas), lalu diekstrak embedding penuh untuk 3.180 sampel.
    - **Output yang diharapkan:** DataFrame `df_hunting`, pemilihan `top1` dan `top2`, file `/marimo/Fenomena/model_hunting.md`, serta file embedding `.npy` untuk seluruh 3.180 sampel (`emb_{name}_pre.npy`, `_post.npy`, `_diff.npy`).
    - **Interpretasi:** **TOP-2 terpilih adalah `RemoteCLIP` (`silhouette=0.02311`, `db=4.5010`, `entropy=1.7533`) dan `SkyCLIP` (`silhouette=0.02151`, `db=4.4863`, `entropy=1.7675`)**, keduanya mengungguli `DINOv2-large` (`silhouette=0.01905`, `db=5.0552`, `entropy=1.7791`) tanpa kolaps kelas (`max_class_share` `0.26` dan `0.22`). `SigLIP-large` gagal dimuat pada sesi kernel berjalan karena `sentencepiece` baru terpasang di K13 sebelum restart kernel.
    """)
    return


@app.cell
def K14a_hunting(
    Image,
    RANDOM_SEED,
    df_inv_clean,
    mo,
    np,
    open_clip,
    pd,
    scipy_stats,
    torch,
    transformers,
):
    # Benchmark 4 kandidat backbone (RemoteCLIP, SkyCLIP, DINOv2-large, SigLIP-large)
    # pada 300 sampel acak (seed=42) dan pilih TOP-2 untuk ekstraksi embedding penuh.
    import gc as _gc
    import time as _time
    import torch.nn.functional as _F
    from huggingface_hub import hf_hub_download as _hf_dl
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.metrics import silhouette_score as _sil_score, davies_bouldin_score as _db_score

    mo.status.toast(title="Model Hunting Dimulai", description="Mengevaluasi RemoteCLIP, SkyCLIP, DINOv2-large, SigLIP-large")

    CLASS_NAMES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
    HUNT_PROMPTS = [
        "aerial satellite image of earthquake damage and collapsed structures",
        "aerial satellite image of wildfire burn scar and charred vegetation",
        "aerial satellite image of hurricane wind damage",
        "aerial satellite image of tsunami coastal flooding",
        "aerial satellite image of volcanic ash and lava deposits",
        "aerial satellite image of explosion crater and radial blast damage",
    ]

    _sample300 = df_inv_clean.sample(n=300, random_state=RANDOM_SEED).reset_index(drop=True)
    _imgs_pre_300 = [Image.open(_p).convert("RGB") for _p in _sample300["path_pre"]]
    _imgs_post_300 = [Image.open(_p).convert("RGB") for _p in _sample300["path_post"]]

    def _load_candidate(_cname):
        if _cname == "RemoteCLIP":
            _ckpt = _hf_dl(repo_id="chendelong/RemoteCLIP", filename="RemoteCLIP-ViT-L-14.pt")
            _model, _, _preproc = open_clip.create_model_and_transforms("ViT-L-14")
            _sd = torch.load(_ckpt, map_location="cpu", weights_only=False)
            _model.load_state_dict(_sd, strict=True)
            _tok = open_clip.get_tokenizer("ViT-L-14")
            _model = _model.cuda().eval().to(torch.bfloat16)
            return ("open_clip", _model, _preproc, _tok)
        if _cname == "SkyCLIP":
            _last_err = None
            for _repo, _fn in [
                ("furiousteabag/SkyCLIP", "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS_epoch_20.pt"),
                ("BiliSakura/SkyCLIP-ViT-L-14", "SkyCLIP-ViT-L-14.pth"),
            ]:
                try:
                    _ckpt = _hf_dl(repo_id=_repo, filename=_fn)
                    _model, _, _preproc = open_clip.create_model_and_transforms("ViT-L-14")
                    _raw = torch.load(_ckpt, map_location="cpu", weights_only=False)
                    if isinstance(_raw, dict) and "state_dict" in _raw:
                        _raw = _raw["state_dict"]
                    elif isinstance(_raw, dict) and "model" in _raw:
                        _raw = _raw["model"]
                    _raw = {_k.replace("module.", ""): _v for _k, _v in _raw.items()}
                    _model.load_state_dict(_raw, strict=True)
                    _tok = open_clip.get_tokenizer("ViT-L-14")
                    _model = _model.cuda().eval().to(torch.bfloat16)
                    return ("open_clip", _model, _preproc, _tok)
                except Exception as _e:
                    _last_err = _e
            raise RuntimeError(f"SkyCLIP gagal dimuat: {_last_err}")
        if _cname == "DINOv2-large":
            _proc = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            _model = transformers.AutoModel.from_pretrained("facebook/dinov2-large", torch_dtype=torch.bfloat16).cuda().eval()
            return ("dinov2", _model, _proc, None)
        if _cname == "SigLIP-large":
            _proc = transformers.AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
            _model = transformers.AutoModel.from_pretrained("google/siglip-large-patch16-384", torch_dtype=torch.bfloat16).cuda().eval()
            return ("siglip", _model, _proc, None)
        raise ValueError(_cname)

    @torch.no_grad()
    def _extract_bundle(_bundle, _pil_list, _batch_size=32):
        _kind, _model, _proc, _tok = _bundle
        _feats = []
        for _i in range(0, len(_pil_list), _batch_size):
            _batch = _pil_list[_i:_i + _batch_size]
            if _kind == "open_clip":
                _x = torch.stack([_proc(_im) for _im in _batch]).cuda().to(torch.bfloat16)
                _f = _model.encode_image(_x)
            elif _kind == "dinov2":
                _inp = _proc(images=_batch, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
                _out = _model(**_inp)
                _f = _out.last_hidden_state[:, 0]
            elif _kind == "siglip":
                _inp = _proc(images=_batch, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
                _out = _model.get_image_features(**_inp)
                _f = _out if isinstance(_out, torch.Tensor) else (_out.pooler_output if hasattr(_out, "pooler_output") else _out[0])
            _f = _F.normalize(_f.float(), dim=-1)
            _feats.append(_f.cpu().numpy())
        return np.vstack(_feats)

    @torch.no_grad()
    def _extract_text_bundle(_bundle, _prompts):
        _kind, _model, _proc, _tok = _bundle
        if _kind == "open_clip":
            _t = _tok(_prompts).cuda()
            _tf = _model.encode_text(_t)
            return _F.normalize(_tf.float(), dim=-1).cpu().numpy()
        if _kind == "siglip":
            _inp = _proc(text=_prompts, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            _out = _model.get_text_features(**_inp)
            _tf = _out if isinstance(_out, torch.Tensor) else (_out.pooler_output if hasattr(_out, "pooler_output") else _out[0])
            return _F.normalize(_tf.float(), dim=-1).cpu().numpy()
        return None

    _hunt_rows = []
    for _cname in ["RemoteCLIP", "SkyCLIP", "DINOv2-large", "SigLIP-large"]:
        _t_s = _time.time()
        try:
            _bundle = _load_candidate(_cname)
            _e_pre = _extract_bundle(_bundle, _imgs_pre_300, _batch_size=64)
            _e_post = _extract_bundle(_bundle, _imgs_post_300, _batch_size=64)
            _diff_raw = _e_post - _e_pre
            _norms = np.linalg.norm(_diff_raw, axis=1, keepdims=True) + 1e-8
            _e_diff = _diff_raw / _norms

            _km = _KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit(_e_diff)
            _sil = float(_sil_score(_e_diff, _km.labels_))
            _db = float(_db_score(_e_diff, _km.labels_))

            _tf = _extract_text_bundle(_bundle, HUNT_PROMPTS)
            if _tf is not None:
                _sims = _e_diff @ _tf.T
                _preds = _sims.argmax(axis=1)
                _counts = np.bincount(_preds, minlength=6)
                _probs_d = _counts / _counts.sum()
                _entropy = float(scipy_stats.entropy(_probs_d + 1e-12))
                _max_share = float(_probs_d.max())
                _prompt_note = f"argmax_counts={_counts.tolist()}"
            else:
                _counts = np.bincount(_km.labels_, minlength=6)
                _probs_d = _counts / _counts.sum()
                _entropy = float(scipy_stats.entropy(_probs_d + 1e-12))
                _max_share = float(_probs_d.max())
                _prompt_note = f"vision_only_kmeans_counts={_counts.tolist()}"

            _hunt_rows.append({
                "model": _cname,
                "silhouette": round(_sil, 5),
                "db_score": round(_db, 4),
                "entropy": round(_entropy, 4),
                "max_class_share": round(_max_share, 3),
                "dim": int(_e_diff.shape[1]),
                "ok_flag": True,
                "sec": round(_time.time() - _t_s, 1),
                "notes": _prompt_note,
            })
            del _bundle
            _gc.collect()
            torch.cuda.empty_cache()
        except Exception as _e:
            _hunt_rows.append({
                "model": _cname,
                "silhouette": np.nan,
                "db_score": np.nan,
                "entropy": np.nan,
                "max_class_share": np.nan,
                "dim": 0,
                "ok_flag": False,
                "sec": round(_time.time() - _t_s, 1),
                "notes": f"ERR: {str(_e)[:140]}",
            })
            _gc.collect()
            torch.cuda.empty_cache()

    df_hunting = pd.DataFrame(_hunt_rows)
    _ok_df = df_hunting[df_hunting["ok_flag"]].copy()
    if len(_ok_df) >= 2:
        # Skor gabungan: silhouette tinggi (+) & entropy rendah (-), penalti jika kolaps (>0.75 di 1 kelas)
        _sil_norm = (_ok_df["silhouette"] - _ok_df["silhouette"].min()) / max(_ok_df["silhouette"].max() - _ok_df["silhouette"].min(), 1e-8)
        _ent_norm = (_ok_df["entropy"] - _ok_df["entropy"].min()) / max(_ok_df["entropy"].max() - _ok_df["entropy"].min(), 1e-8)
        _collapse_pen = (_ok_df["max_class_share"] > 0.75).astype(float) * 2.0
        _ok_df["rank_score"] = _sil_norm + (1.0 - _ent_norm) - _collapse_pen
        _ok_df = _ok_df.sort_values("rank_score", ascending=False)
        top1, top2 = _ok_df["model"].iloc[0], _ok_df["model"].iloc[1]
    elif len(_ok_df) == 1:
        top1 = top2 = _ok_df["model"].iloc[0]
    else:
        top1, top2 = "DINOv2-large", "SigLIP-large"

    print("=== HASIL MODEL HUNTING (300 SAMPEL) ===")
    print(df_hunting.to_string(index=False))
    print(f"\nTOP-2 TERPILIH: {top1}, {top2}")
    df_hunting
    return CLASS_NAMES, HUNT_PROMPTS, df_hunting, top1, top2


@app.cell
def K14b_extract(
    ARTIFACT_DIR,
    Image,
    Path,
    df_hunting,
    df_inv_clean,
    mo,
    np,
    open_clip,
    pd,
    top1,
    top2,
    torch,
    tqdm,
    transformers,
):
    # Ekstraksi embedding penuh 3.180 sampel (pre, post, diff, dan diff_hflip untuk TTA)
    # bagi TOP-2 model terpilih (top1, top2), serta simpan df_hunting ke model_hunting.md.
    import gc as _gc
    import time as _time
    import torch.nn.functional as _F
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from huggingface_hub import hf_hub_download as _hf_dl

    def _df_to_md(_df):
        _cols = list(_df.columns)
        _hdr = "| " + " | ".join(str(_c) for _c in _cols) + " |"
        _sep = "| " + " | ".join("---" for _ in _cols) + " |"
        _lines = [_hdr, _sep]
        for _, _r in _df.iterrows():
            _lines.append("| " + " | ".join(str(_r[_c]) for _c in _cols) + " |")
        return "\n".join(_lines)

    def _load_top_model(_cname):
        if _cname == "RemoteCLIP":
            _ckpt = _hf_dl(repo_id="chendelong/RemoteCLIP", filename="RemoteCLIP-ViT-L-14.pt")
            _m, _, _pp = open_clip.create_model_and_transforms("ViT-L-14")
            _m.load_state_dict(torch.load(_ckpt, map_location="cpu", weights_only=False), strict=True)
            return ("open_clip", _m.cuda().eval().to(torch.bfloat16), _pp)
        if _cname == "SkyCLIP":
            for _repo, _fn in [
                ("furiousteabag/SkyCLIP", "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS_epoch_20.pt"),
                ("BiliSakura/SkyCLIP-ViT-L-14", "SkyCLIP-ViT-L-14.pth"),
            ]:
                try:
                    _ckpt = _hf_dl(repo_id=_repo, filename=_fn)
                    _m, _, _pp = open_clip.create_model_and_transforms("ViT-L-14")
                    _raw = torch.load(_ckpt, map_location="cpu", weights_only=False)
                    if isinstance(_raw, dict) and "state_dict" in _raw:
                        _raw = _raw["state_dict"]
                    elif isinstance(_raw, dict) and "model" in _raw:
                        _raw = _raw["model"]
                    _raw = {_k.replace("module.", ""): _v for _k, _v in _raw.items()}
                    _m.load_state_dict(_raw, strict=True)
                    return ("open_clip", _m.cuda().eval().to(torch.bfloat16), _pp)
                except Exception:
                    continue
        if _cname == "DINOv2-large":
            _pp = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            _m = transformers.AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.bfloat16).cuda().eval()
            return ("dinov2", _m, _pp)
        raise ValueError(f"Unsupported model for full extraction: {_cname}")

    @torch.no_grad()
    def _extract_full(_cname, _paths_pre, _paths_post, _save_hflip=False, _bs=64):
        _kind, _model, _pp = _load_top_model(_cname)

        def _prep(_p):
            with Image.open(_p) as _im:
                _rgb = _im.convert("RGB")
                if _kind == "open_clip":
                    _t0 = _pp(_rgb)
                    _t1 = _pp(_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)) if _save_hflip else None
                    return _t0, _t1
                return _rgb, (_rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if _save_hflip else None)

        def _run_paths(_paths):
            _feats_orig, _feats_flip = [], []
            for _i in tqdm(range(0, len(_paths), _bs), desc=f"{_cname} extract"):
                _chunk = _paths[_i:_i + _bs]
                with _TPE(max_workers=8) as _ex:
                    _res = list(_ex.map(_prep, _chunk))
                if _kind == "open_clip":
                    _x0 = torch.stack([_r[0] for _r in _res]).cuda().to(torch.bfloat16)
                    _f0 = _F.normalize(_model.encode_image(_x0).float(), dim=-1).cpu().numpy()
                    _feats_orig.append(_f0)
                    if _save_hflip:
                        _x1 = torch.stack([_r[1] for _r in _res]).cuda().to(torch.bfloat16)
                        _f1 = _F.normalize(_model.encode_image(_x1).float(), dim=-1).cpu().numpy()
                        _feats_flip.append(_f1)
                else:
                    _inp0 = _pp(images=[_r[0] for _r in _res], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
                    _f0 = _F.normalize(_model(**_inp0).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
                    _feats_orig.append(_f0)
                    if _save_hflip:
                        _inp1 = _pp(images=[_r[1] for _r in _res], return_tensors="pt").to("cuda", dtype=torch.bfloat16)
                        _f1 = _F.normalize(_model(**_inp1).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
                        _feats_flip.append(_f1)
            _o = np.vstack(_feats_orig).astype(np.float32)
            _fl = np.vstack(_feats_flip).astype(np.float32) if _save_hflip else None
            return _o, _fl

        _pre, _pre_fl = _run_paths(_paths_pre)
        _post, _post_fl = _run_paths(_paths_post)
        _diff_raw = _post - _pre
        _diff = (_diff_raw / (np.linalg.norm(_diff_raw, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        _diff_fl = None
        if _save_hflip:
            _dfl = _post_fl - _pre_fl
            _diff_fl = (_dfl / (np.linalg.norm(_dfl, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        del _model
        _gc.collect()
        torch.cuda.empty_cache()
        return _pre, _post, _diff, _diff_fl

    _paths_pre_all = df_inv_clean["path_pre"].tolist()
    _paths_post_all = df_inv_clean["path_post"].tolist()
    _art = Path(ARTIFACT_DIR)

    _emb_summary = []
    for _idx_m, _mname in enumerate([top1, top2]):
        _is_top1 = (_idx_m == 0)
        _p_pre = _art / f"emb_{_mname}_pre.npy"
        _p_post = _art / f"emb_{_mname}_post.npy"
        _p_diff = _art / f"emb_{_mname}_diff.npy"
        _p_fl = _art / f"emb_{_mname}_diff_hflip.npy"
        if _p_pre.is_file() and _p_post.is_file() and _p_diff.is_file() and (not _is_top1 or _p_fl.is_file()):
            _ediff = np.load(_p_diff, mmap_mode="r")
            if _ediff.shape[0] == len(_paths_pre_all):
                _emb_summary.append({"model": _mname, "shape": tuple(_ediff.shape), "hflip_saved": _p_fl.is_file(), "cached": True})
                continue
        _epre, _epost, _ediff, _ediff_fl = _extract_full(_mname, _paths_pre_all, _paths_post_all, _save_hflip=_is_top1)
        np.save(_p_pre, _epre)
        np.save(_p_post, _epost)
        np.save(_p_diff, _ediff)
        if _ediff_fl is not None:
            np.save(_p_fl, _ediff_fl)
        _emb_summary.append({"model": _mname, "shape": tuple(_ediff.shape), "hflip_saved": _ediff_fl is not None, "cached": False})

    _hunt_md = f"""# Model Hunting Report — Kompetisi Fenomena

    ## Ringkasan Benchmark (300 Sampel, Seed=42)
    - **TOP-1:** `{top1}`
    - **TOP-2:** `{top2}`
    - **Kriteria:** Silhouette score KMeans `k=6` (lebih tinggi lebih baik), Davies-Bouldin score (lebih rendah lebih baik), serta entropy argmax prompt teks 6 kelas (lebih rendah & tanpa kolaps kelas lebih baik). `Prithvi` dilewati karena input hanya JPEG RGB 3-kanal.

    {_df_to_md(df_hunting)}

    ## Embedding Tersimpan (3.180 Sampel)
    {_df_to_md(pd.DataFrame(_emb_summary))}
    """
    with open(_art / "model_hunting.md", "w", encoding="utf-8") as _f:
        _f.write(_hunt_md)

    print("Ekstraksi embedding selesai:", _emb_summary)
    mo.status.toast(title=f"Model hunting selesai: TOP={top1},{top2}", description="Embedding 3.180 sampel tersimpan")
    pd.DataFrame(_emb_summary)
    return


@app.cell(hide_code=True)
def M15_pseudo(mo):
    mo.md("""
    ## 15. Pseudo-Label dengan VLM Kuat (Qwen2.5-VL)

    - **Tujuan tahap:** Karena `HAS_LABELS = False`, pseudo-label guru dibuat menggunakan VLM `Qwen2.5-VL-7B-Instruct` yang mengamati pasangan komposit *side-by-side* `[pre | post]` secara langsung—mengatasi temuan `(T1)` (`76.3%` pasangan *unregistered*) tanpa bergantung pada *pixel diff* mentah.
    - **Mekanisme & Validasi:** Output JSON (`label`, `confidence`, `reason`) di-parse secara robust, diaudit distribusi kelasnya (`%parse_ok`, `%conf >= 0.7`, serta deteksi RED FLAG jika satu kelas `> 60%`), dan di-*cross-check* *purity*-nya terhadap klaster KMeans pada embedding terbaik (`RemoteCLIP`). Jika VLM gagal dimuat/kolaps (`> 80%`), pipeline otomatis beralih ke fallback clustering + *prompt cosine similarity*.
    - **Output yang diharapkan:** DataFrame `df_pseudo` (disimpan ke `/marimo/Fenomena/pseudo_labels.parquet`), `df_purity`, dan notifikasi toast jumlah sampel *high-confidence*.
    - **Interpretasi:** `Qwen2.5-VL-7B-Instruct` selesai dalam `235.3` detik dengan `parse_ok = 100.00%` dan `conf >= 0.7 = 3.180/3.180 (100.00%)`. **RED FLAG terdeteksi:** kelas `Fire` mendominasi `1.958` sampel (`61.57% > 60%`), sementara `Earthquake` `331` (`10.41%`), `Volcano` `301` (`9.47%`), `Explosion` `280` (`8.81%`), `Tsunami` `196` (`6.16%`), dan `Hurricane` `114` (`3.58%`). Purity per kelas terhadap klaster KMeans `RemoteCLIP` berkisar `0.20–0.29`. Temuan ini menegaskan perlunya *class weights* (sesuai `T2`) dan *self-training* iteratif untuk mengoreksi bias VLM pada kelas `Fire`.
    """)
    return


@app.cell
def K15a_vlm_load(Image, df_inv_clean, mo, torch, transformers):
    # Memuat Qwen/Qwen2.5-VL-7B-Instruct dalam bfloat16 ke GPU (tanpa tergantung cache accelerate)
    # dan menguji generasi JSON pada 2 sampel komposit [pre | post].
    import time as _time
    import importlib as _imp

    mo.status.toast(title="VLM Pseudo-Label Dimulai", description="Memuat Qwen2.5-VL-7B-Instruct...")
    _t0 = _time.time()

    VLM_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    VLM_PROMPT = (
        "You are analyzing a side-by-side satellite image pair (left=pre-disaster, right=post-disaster) "
        "of the same location. Classify the disaster into exactly one of: "
        "Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion. "
        "Respond ONLY with valid JSON: "
        "{\"label\": \"<one of the six>\", \"confidence\": <0.0-1.0>, \"reason\": \"<short>\"}."
    )

    def _make_side_by_side(_p1, _p2, _tile_sz=336):
        with Image.open(_p1) as _im1, Image.open(_p2) as _im2:
            _r1 = _im1.convert("RGB").resize((_tile_sz, _tile_sz), Image.Resampling.BILINEAR)
            _r2 = _im2.convert("RGB").resize((_tile_sz, _tile_sz), Image.Resampling.BILINEAR)
        _canvas = Image.new("RGB", (_tile_sz * 2, _tile_sz))
        _canvas.paste(_r1, (0, 0))
        _canvas.paste(_r2, (_tile_sz, 0))
        return _canvas

    VLM_OK = False
    vlm_processor = None
    vlm_model = None
    vlm_load_note = ""

    try:
        _imp.reload(transformers.utils.import_utils)
        _cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) or getattr(transformers, "AutoModelForImageTextToText", None)
        vlm_processor = transformers.AutoProcessor.from_pretrained(
            VLM_MODEL_ID,
            trust_remote_code=True,
            min_pixels=200 * 28 * 28,
            max_pixels=400 * 28 * 28,
        )
        if hasattr(vlm_processor, "tokenizer"):
            vlm_processor.tokenizer.padding_side = "left"
        vlm_model = _cls.from_pretrained(
            VLM_MODEL_ID,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        _test_imgs = [_make_side_by_side(df_inv_clean.iloc[_i]["path_pre"], df_inv_clean.iloc[_i]["path_post"]) for _i in range(2)]
        _msgs = [[{
            "role": "user",
            "content": [{"type": "image", "image": _im}, {"type": "text", "text": VLM_PROMPT}],
        }] for _im in _test_imgs]
        _texts = [vlm_processor.apply_chat_template(_m, tokenize=False, add_generation_prompt=True) for _m in _msgs]
        _inp = vlm_processor(text=_texts, images=_test_imgs, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            _out = vlm_model.generate(**_inp, max_new_tokens=80, do_sample=False)
        _gen = vlm_processor.batch_decode(_out[:, _inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print("Smoke test VLM output:", _gen)
        VLM_OK = True
        vlm_load_note = f"Loaded {VLM_MODEL_ID} in {_time.time()-_t0:.1f}s"
    except Exception as _e:
        vlm_load_note = f"VLM load failed ({_time.time()-_t0:.1f}s): {str(_e)[:250]}"
        print(vlm_load_note)

    print("VLM_OK:", VLM_OK, "|", vlm_load_note)
    return VLM_OK, VLM_PROMPT, vlm_model, vlm_processor


@app.cell
def K15b_vlm_infer(
    ARTIFACT_DIR,
    CLASS_NAMES,
    HUNT_PROMPTS,
    Image,
    Path,
    RANDOM_SEED,
    VLM_OK,
    VLM_PROMPT,
    df_inv_clean,
    json,
    mo,
    np,
    open_clip,
    pd,
    top1,
    torch,
    tqdm,
    vlm_model,
    vlm_processor,
):
    # Inferensi VLM batch (Qwen2.5-VL-7B) pada 3.180 komposit [pre | post], parse JSON robust,
    # validasi distribusi & purity vs klaster embedding K14, lalu simpan pseudo_labels.parquet.
    import gc as _gc
    import re as _re
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from sklearn.cluster import KMeans as _KMeans

    _t0 = _time.time()
    _art = Path(ARTIFACT_DIR)
    _final_pq = _art / "pseudo_labels.parquet"
    _partial_jsonl = _art / "_vlm_partial.jsonl"

    def _build_sbs(_row, _tile_sz=280):
        with Image.open(_row["path_pre"]) as _im1, Image.open(_row["path_post"]) as _im2:
            _r1 = _im1.convert("RGB").resize((_tile_sz, _tile_sz), Image.Resampling.BILINEAR)
            _r2 = _im2.convert("RGB").resize((_tile_sz, _tile_sz), Image.Resampling.BILINEAR)
        _c = Image.new("RGB", (_tile_sz * 2, _tile_sz))
        _c.paste(_r1, (0, 0))
        _c.paste(_r2, (_tile_sz, 0))
        return _c

    _canon = {_c.lower(): _c for _c in CLASS_NAMES}

    def _parse_vlm_json(_txt):
        _clean = _re.sub(r"```(?:json)?|```", "", str(_txt)).strip()
        _m_obj = _re.search(r"\{.*\}", _clean, flags=_re.DOTALL)
        if _m_obj:
            try:
                _d = json.loads(_m_obj.group(0))
                _lbl_raw = str(_d.get("label", "")).strip().lower()
                _lbl = _canon.get(_lbl_raw)
                if _lbl is None:
                    for _k, _v in _canon.items():
                        if _k in _lbl_raw:
                            _lbl = _v
                            break
                _conf = float(_d.get("confidence", 0.5))
                _conf = float(np.clip(_conf, 0.0, 1.0))
                _rsn = str(_d.get("reason", ""))[:200]
                if _lbl is not None:
                    return _lbl, _conf, _rsn, True
            except Exception:
                pass
        # Regex fallback
        _lbl = None
        for _k, _v in _canon.items():
            if _re.search(rf"\b{_k}\b", _clean, flags=_re.IGNORECASE):
                _lbl = _v
                break
        _mc = _re.search(r"\"?confidence\"?\s*:\s*([0-9]*\.?[0-9]+)", _clean)
        _conf = float(np.clip(float(_mc.group(1)), 0.0, 1.0)) if _mc else (0.55 if _lbl else 0.0)
        if _lbl is not None:
            return _lbl, _conf, _clean[:160], True
        return "Earthquake", 0.0, _clean[:160], False

    _records_by_id = {}
    if _final_pq.is_file():
        _cached_df = pd.read_parquet(_final_pq)
        if len(_cached_df) == len(df_inv_clean):
            for _r in _cached_df.to_dict("records"):
                _records_by_id[_r["id"]] = _r
    elif _partial_jsonl.is_file():
        with open(_partial_jsonl, "r", encoding="utf-8") as _f:
            for _line in _f:
                if _line.strip():
                    _r = json.loads(_line)
                    _records_by_id[_r["id"]] = _r

    _todo_rows = [_r for _r in df_inv_clean.to_dict("records") if _r["id"] not in _records_by_id]

    if VLM_OK and len(_todo_rows) > 0:
        _bs = 32
        with open(_partial_jsonl, "a", encoding="utf-8") as _f_log:
            for _i in tqdm(range(0, len(_todo_rows), _bs), desc="VLM Pseudo-Label"):
                _chunk = _todo_rows[_i:_i + _bs]
                with _TPE(max_workers=8) as _ex:
                    _imgs = list(_ex.map(_build_sbs, _chunk))
                _msgs = [[{
                    "role": "user",
                    "content": [{"type": "image", "image": _im}, {"type": "text", "text": VLM_PROMPT}],
                }] for _im in _imgs]
                _texts = [vlm_processor.apply_chat_template(_m, tokenize=False, add_generation_prompt=True) for _m in _msgs]
                _inp = vlm_processor(text=_texts, images=_imgs, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    _out = vlm_model.generate(**_inp, max_new_tokens=72, do_sample=False)
                _gens = vlm_processor.batch_decode(_out[:, _inp["input_ids"].shape[1]:], skip_special_tokens=True)
                for _row, _raw_txt in zip(_chunk, _gens):
                    _lbl, _conf, _rsn, _ok = _parse_vlm_json(_raw_txt)
                    _rec = {
                        "id": _row["id"],
                        "vlm_label": _lbl,
                        "vlm_conf": _conf,
                        "vlm_reason": _rsn,
                        "parse_ok": _ok,
                    }
                    _records_by_id[_row["id"]] = _rec
                    _f_log.write(json.dumps(_rec, ensure_ascii=False) + "\n")
                _f_log.flush()

    # Susun sesuai urutan df_inv_clean
    if len(_records_by_id) == len(df_inv_clean):
        df_pseudo = pd.DataFrame([_records_by_id[_id] for _id in df_inv_clean["id"]])
    else:
        df_pseudo = pd.DataFrame(columns=["id", "vlm_label", "vlm_conf", "vlm_reason", "parse_ok"])

    # Cek fallback jika VLM gagal total / kolaps >80%
    _need_fallback = (not VLM_OK) or (len(df_pseudo) != len(df_inv_clean)) or (df_pseudo["parse_ok"].mean() < 0.05)
    if not _need_fallback:
        _top_share = float(df_pseudo["vlm_label"].value_counts(normalize=True).max())
        if _top_share > 0.80:
            print(f"RED FLAG KRITIS: Satu kelas mendominasi {_top_share*100:.1f}% (>80%) -> aktifkan fallback embedding+KMeans!")
            _need_fallback = True

    if _need_fallback:
        print("Menjalankan FALLBACK pseudo-label: embedding model terbaik + KMeans + prompt cosine similarity...")
        _emb_diff = np.load(_art / f"emb_{top1}_diff.npy")
        _km = _KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit(_emb_diff)
        _ckpt_name = "chendelong/RemoteCLIP" if top1 == "RemoteCLIP" else None
        _m_oc, _, _ = open_clip.create_model_and_transforms("ViT-L-14")
        if top1 == "RemoteCLIP":
            from huggingface_hub import hf_hub_download as _hf_dl
            _m_oc.load_state_dict(torch.load(_hf_dl("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt"), map_location="cpu", weights_only=False))
        _tok_oc = open_clip.get_tokenizer("ViT-L-14")
        _m_oc = _m_oc.cuda().eval()
        with torch.no_grad():
            _tf = _m_oc.encode_text(_tok_oc(HUNT_PROMPTS).cuda()).float()
            _tf = (_tf / _tf.norm(dim=-1, keepdim=True)).cpu().numpy()
        del _m_oc
        _cent = _km.cluster_centers_ / (np.linalg.norm(_km.cluster_centers_, axis=1, keepdims=True) + 1e-8)
        _sim_c = _cent @ _tf.T
        # Greedy bipartite mapping cluster -> class agar tiap klaster dapat kelas unik
        _used_cls = set()
        _cl_to_cls = {}
        for _ci in np.argsort(-_sim_c.max(axis=1)):
            for _cj in np.argsort(-_sim_c[_ci]):
                if int(_cj) not in _used_cls:
                    _cl_to_cls[int(_ci)] = CLASS_NAMES[int(_cj)]
                    _used_cls.add(int(_cj))
                    break
        _sample_sims = _emb_diff @ _tf.T
        _fb_rows = []
        for _idx, _id in enumerate(df_inv_clean["id"]):
            _lbl = _cl_to_cls[int(_km.labels_[_idx])]
            _cidx = CLASS_NAMES.index(_lbl)
            _csim = float(np.clip((_sample_sims[_idx, _cidx] + 1.0) / 2.0, 0.0, 1.0))
            _fb_rows.append({
                "id": _id,
                "vlm_label": _lbl,
                "vlm_conf": _csim,
                "vlm_reason": f"fallback_kmeans_cluster_{_km.labels_[_idx]}",
                "parse_ok": True,
            })
        df_pseudo = pd.DataFrame(_fb_rows)

    # Validasi distribusi & cross-check purity dengan klaster KMeans pada emb_diff terbaik
    _vc = df_pseudo["vlm_label"].value_counts()
    _vc_pct = (_vc / len(df_pseudo) * 100.0).round(2)
    _parse_pct = float(df_pseudo["parse_ok"].mean() * 100.0)
    _n_high_conf = int((df_pseudo["vlm_conf"] >= 0.7).sum())
    _high_conf_pct = float(_n_high_conf / len(df_pseudo) * 100.0)
    _red_flag_60 = bool((_vc_pct > 60.0).any())

    _emb_best = np.load(_art / f"emb_{top1}_diff.npy")
    _km_eval = _KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init=10).fit(_emb_best)
    _df_x = df_pseudo.copy()
    _df_x["cluster"] = _km_eval.labels_

    _purity_rows = []
    for _lbl in CLASS_NAMES:
        _sub = _df_x[_df_x["vlm_label"] == _lbl]
        if len(_sub) > 0:
            _dom_cl = int(_sub["cluster"].value_counts().idxmax())
            _dom_cnt = int(_sub["cluster"].value_counts().max())
            _pur = float(_dom_cnt / len(_sub))
        else:
            _dom_cl, _dom_cnt, _pur = -1, 0, 0.0
        _purity_rows.append({
            "vlm_label": _lbl,
            "count": len(_sub),
            "pct": round(len(_sub) / len(_df_x) * 100.0, 2),
            "dominant_cluster": _dom_cl,
            "dominant_cluster_count": _dom_cnt,
            "purity": round(_pur, 4),
        })
    df_purity = pd.DataFrame(_purity_rows)

    df_pseudo.to_parquet(_final_pq, index=False)

    # Bebaskan VRAM model VLM setelah selesai
    try:
        if vlm_model is not None:
            vlm_model.cpu()
    except Exception:
        pass
    _gc.collect()
    torch.cuda.empty_cache()

    print(f"VLM Pseudo-labeling selesai dalam {_time.time()-_t0:.1f}s | parse_ok: {_parse_pct:.2f}% | conf>=0.7: {_n_high_conf}/3180 ({_high_conf_pct:.2f}%)")
    if _red_flag_60:
        print("RED FLAG: Salah satu kelas melebihi 60% distribusi pseudo-label!")
    print("\nDistribusi & Purity vs KMeans Cluster (emb_diff terbaik):")
    print(df_purity.to_string(index=False))
    mo.status.toast(title=f"Pseudo-label selesai: {_n_high_conf}/3180 high-conf", description=f"parse_ok={_parse_pct:.1f}%")
    df_purity
    return (df_pseudo,)


@app.cell(hide_code=True)
def M16_model(mo):
    mo.md("""
    ## 16. Arsitektur Siamese: Dual Fusion (Concat + Attention)

    - **Tujuan tahap:** Mengatasi temuan `(T1)` (`76.3%` pasangan *unregistered*) dengan membangun arsitektur Siamese *shared-weights* yang menggabungkan dua jalur fitur semantik: **Jalur A (Concatenation)** untuk menangkap konteks global `[f_pre, f_post]` dan **Jalur B (Absolute Difference + Channel Attention)** untuk menonjolkan fitur perubahan paling informatif.
    - **Kandidat Backbone:** `SiameseDualFusion` (`efficientnet_b4`) dan `SiameseConvNeXt` (`convnext_small.fb_in22k_ft_in1k`) untuk keragaman arsitektur CNN modern + klasik pada tahap ensemble.
    - **Output yang diharapkan:** Verifikasi *forward pass* dummy `[2, 3, 384, 384]` → `[2, 6]` beserta jumlah parameter masing-masing model.
    """)
    return


@app.cell
def K16_model(pd, timm, torch):
    # Definisi arsitektur SiameseDualFusion (EfficientNet-B4) dan SiameseConvNeXt (ConvNeXt-Small)
    # dengan jalur ganda Concat + AbsDiff Channel Attention, lalu uji forward pass dummy [2, 3, 384, 384].
    import torch.nn as nn

    class ChannelAttention1D(nn.Module):
        """Channel attention ringan untuk menyeleksi dimensi perubahan semantik pada vektor fitur 1D."""
        def __init__(self, dim, reduction=8):
            super().__init__()
            _hid = max(dim // reduction, 16)
            self.net = nn.Sequential(
                nn.Linear(dim, _hid),
                nn.GELU(),
                nn.Linear(_hid, dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * self.net(x)

    class _BaseSiameseDualFusion(nn.Module):
        """Backbone shared-weights dengan dual fusion: (A) Concat + (B) AbsDiff Channel Attention."""
        def __init__(self, backbone_name, num_classes=6, pretrained=True):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, features_only=False
            )
            _d = self.backbone.num_features
            self.proj_concat = nn.Sequential(nn.Linear(_d * 2, _d), nn.GELU())
            self.diff_attn = ChannelAttention1D(_d, reduction=8)
            self.proj_diff = nn.Sequential(nn.Linear(_d, _d), nn.GELU())
            self.head = nn.Sequential(
                nn.Linear(_d * 2, _d),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(_d, num_classes),
            )

        def forward(self, pre, post):
            _f_pre = self.backbone(pre)
            _f_post = self.backbone(post)
            _a = self.proj_concat(torch.cat([_f_pre, _f_post], dim=-1))
            _b = self.proj_diff(self.diff_attn(torch.abs(_f_post - _f_pre)))
            return self.head(torch.cat([_a, _b], dim=-1))

    class SiameseDualFusion(_BaseSiameseDualFusion):
        """Siamese Dual Fusion berbasis backbone EfficientNet-B4."""
        def __init__(self, num_classes=6, pretrained=True):
            super().__init__("efficientnet_b4", num_classes=num_classes, pretrained=pretrained)

    class SiameseConvNeXt(_BaseSiameseDualFusion):
        """Siamese Dual Fusion berbasis backbone ConvNeXt-Small."""
        def __init__(self, num_classes=6, pretrained=True):
            _ok_name = None
            for _cand in ["convnext_small.fb_in22k_ft_in1k", "convnext_small.in12k_ft_in1k", "convnext_small"]:
                try:
                    _tmp = timm.create_model(_cand, pretrained=False, num_classes=0)
                    del _tmp
                    _ok_name = _cand
                    break
                except Exception:
                    continue
            if _ok_name is None:
                _ok_name = "convnext_small"
            self.backbone_id = _ok_name
            super().__init__(_ok_name, num_classes=num_classes, pretrained=pretrained)

    _d_pre = torch.randn(2, 3, 384, 384)
    _d_post = torch.randn(2, 3, 384, 384)

    _m_eff = SiameseDualFusion(pretrained=False).eval()
    _m_cnx = SiameseConvNeXt(pretrained=False).eval()
    with torch.no_grad():
        _o_eff = _m_eff(_d_pre, _d_post)
        _o_cnx = _m_cnx(_d_pre, _d_post)

    assert _o_eff.shape == (2, 6), f"Unexpected EffNet shape: {_o_eff.shape}"
    assert _o_cnx.shape == (2, 6), f"Unexpected ConvNeXt shape: {_o_cnx.shape}"

    _model_test_df = pd.DataFrame([
        {"model": "SiameseDualFusion (efficientnet_b4)", "params": sum(p.numel() for p in _m_eff.parameters()), "out_shape": tuple(_o_eff.shape)},
        {"model": f"SiameseConvNeXt ({_m_cnx.backbone_id})", "params": sum(p.numel() for p in _m_cnx.parameters()), "out_shape": tuple(_o_cnx.shape)},
    ])
    del _m_eff, _m_cnx, _d_pre, _d_post
    print(_model_test_df.to_string(index=False))
    _model_test_df
    return SiameseDualFusion, nn


@app.cell(hide_code=True)
def M17_dataset(mo):
    mo.md("""
    ## 17. Dataset & Augmentasi Konsisten pre/post

    - **Tujuan tahap:** Membangun `FenomenaDataset` yang menerapkan transformasi geometrik **identik** pada pasangan `pre`/`post` (`RandomHorizontalFlip`, `RandomRotation(10)`), namun `ColorJitter` **independen** untuk mensimulasikan temuan EDA pergeseran warna/iluminasi tinggi (*KL divergence* `1.88–2.06`), serta melakukan pembagian `StratifiedKFold(n_splits=5)` pada sampel *high-confidence*.
    - **Output yang diharapkan:** Tabel distribusi `vlm_conf`, split `df_train_folds` (kolom `fold` 0–4) dan `df_test_only`, cache citra `384x384` di RAM untuk akselerasi GPU, serta verifikasi assert shape `(3, 384, 384)` pada 4 sampel.
    """)
    return


@app.cell
def K17_dataset(
    CLASS_NAMES,
    ColorJitter,
    DATA_ROOT,
    Image,
    Path,
    RANDOM_SEED,
    TF,
    df_inv_clean,
    df_pseudo,
    np,
    pd,
    random,
    torch,
    tqdm,
):
    # Definisi FenomenaDataset dengan augmentasi geometrik identik & ColorJitter independen,
    # Stratified 5-Fold split pada pseudo-label high_conf, cache 384x384 RAM, dan uji 4 sampel.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import StratifiedKFold

    LABEL2IDX = {_c: _i for _i, _c in enumerate(CLASS_NAMES)}
    IDX2LABEL = {_i: _c for _i, _c in enumerate(CLASS_NAMES)}
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    IMG_CACHE_384 = {}
    def _load_cache_one(_row):
        with Image.open(_row["path_pre"]) as _i1, Image.open(_row["path_post"]) as _i2:
            _a1 = np.asarray(_i1.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR), dtype=np.uint8)
            _a2 = np.asarray(_i2.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR), dtype=np.uint8)
        return _row["id"], (_a1, _a2)

    with _TPE(max_workers=16) as _ex:
        for _cid, _pair in tqdm(_ex.map(_load_cache_one, df_inv_clean.to_dict("records")), total=len(df_inv_clean), desc="Cache 384x384 RAM"):
            IMG_CACHE_384[_cid] = _pair

    class FenomenaDataset(Dataset):
        """Dataset pasangan pre/post dengan augmentasi geometrik identik dan ColorJitter independen."""
        def __init__(self, df, data_root=DATA_ROOT, size=384, mode="train", cache_dict=None, return_weight=False, hflip_tta=False):
            self.df = df.reset_index(drop=True)
            self.data_root = Path(data_root)
            self.size = size
            self.mode = mode
            self.cache_dict = cache_dict
            self.return_weight = return_weight
            self.hflip_tta = hflip_tta
            self.jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            _row = self.df.iloc[idx]
            _sid = _row["id"]
            if self.cache_dict is not None and _sid in self.cache_dict and self.size == 384:
                _a1, _a2 = self.cache_dict[_sid]
                _t1 = torch.from_numpy(_a1.copy()).permute(2, 0, 1).float().div_(255.0)
                _t2 = torch.from_numpy(_a2.copy()).permute(2, 0, 1).float().div_(255.0)
            else:
                with Image.open(self.data_root / _sid / "pre.jpg") as _f1:
                    _t1 = TF.to_tensor(_f1.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))
                with Image.open(self.data_root / _sid / "post.jpg") as _f2:
                    _t2 = TF.to_tensor(_f2.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR))

            if self.mode == "train":
                if random.random() < 0.5:
                    _t1 = TF.hflip(_t1)
                    _t2 = TF.hflip(_t2)
                _angle = random.uniform(-10.0, 10.0)
                _t1 = TF.rotate(_t1, _angle)
                _t2 = TF.rotate(_t2, _angle)
                _t1 = self.jitter(_t1)
                _t2 = self.jitter(_t2)
            elif self.hflip_tta:
                _t1 = TF.hflip(_t1)
                _t2 = TF.hflip(_t2)

            _t1 = (_t1 - IMAGENET_MEAN) / IMAGENET_STD
            _t2 = (_t2 - IMAGENET_MEAN) / IMAGENET_STD
            _lbl_str = _row["vlm_label"] if "vlm_label" in _row and pd.notna(_row["vlm_label"]) else "Earthquake"
            _lbl = LABEL2IDX.get(_lbl_str, 0)
            if self.return_weight:
                _w = float(_row["sample_weight"]) if "sample_weight" in _row else 1.0
                return _t1, _t2, _lbl, _sid, np.float32(_w)
            return _t1, _t2, _lbl, _sid

    _df_base = df_inv_clean[["id", "path_pre", "path_post"]].merge(df_pseudo, on="id", how="inner")
    _hc_mask = (_df_base["vlm_conf"] >= 0.7) & (_df_base["parse_ok"] == True)
    df_train_folds = _df_base[_hc_mask].copy().reset_index(drop=True)
    df_test_only = _df_base[~_hc_mask].copy().reset_index(drop=True)

    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    df_train_folds["fold"] = -1
    for _f_idx, (_, _val_idx) in enumerate(_skf.split(df_train_folds, df_train_folds["vlm_label"])):
        df_train_folds.loc[_val_idx, "fold"] = _f_idx

    print(f"Split selesai -> df_train_folds: {len(df_train_folds)} | df_test_only: {len(df_test_only)}")
    _ds_test = FenomenaDataset(df_train_folds.head(4), mode="train", cache_dict=IMG_CACHE_384)
    for _j in range(4):
        _p1, _p2, _y, _sid = _ds_test[_j]
        assert _p1.shape == (3, 384, 384) and _p2.shape == (3, 384, 384)
    print("Uji FenomenaDataset (4 sampel) BERHASIL: shape=(3, 384, 384).")
    return (
        IMAGENET_MEAN,
        IMAGENET_STD,
        IMG_CACHE_384,
        LABEL2IDX,
        df_train_folds,
    )


@app.cell(hide_code=True)
def M18_train(mo):
    mo.md("""
    ## 18. Training Siamese Network (Dual Architecture)

    - **Tujuan tahap:** Melatih dua arsitektur Siamese berbeda (`SiameseDualFusion` berbasis `efficientnet_b4` dan `SiameseConvNeXt` berbasis `convnext_small.fb_in22k_ft_in1k`) dengan Stratified 5-Fold CV pada `df_train_folds`, menggunakan *class weights* (`1/freq`, ternormalisasi) sesuai temuan `(T2)`, `label_smoothing=0.1`, *cosine schedule* + *warmup* 5 epoch, AMP `bfloat16`, dan *early stopping* (`patience=7`).
    - **Output yang diharapkan:** Checkpoint terbaik tiap fold (`/marimo/Fenomena/ckpt_{model}_fold{k}.pth`), probabilitas OOF (`/marimo/Fenomena/oof_{model}.npy`), ringkasan **OOF Macro F1** per fold dan per kelas, serta log di `/marimo/Fenomena/training_log.md`.
    """)
    return


@app.cell
def K18a_train_effb4(
    ARTIFACT_DIR,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_CACHE_384,
    LABEL2IDX,
    Path,
    SiameseDualFusion,
    df_inv_clean,
    df_train_folds,
    mo,
    nn,
    np,
    pd,
    torch,
):
    # Definisi fungsi training 5-fold terpadu berkecepatan tinggi (GPU-cached tensor + augmentasi
    # geometrik identik & ColorJitter independen di GPU, AMP BF16, class weights, cosine warmup) dan training EffNet-B4.
    import math as _math
    import time as _time
    import gc as _gc
    import torch.nn.functional as _F
    from sklearn.metrics import f1_score

    # Preload seluruh 3.180 pasang uint8 (2.8 GB) ke VRAM (102 GB) agar 0 bottleneck CPU/I/O
    _ID_TO_POS = {_sid: _i for _i, _sid in enumerate(df_inv_clean["id"])}
    _pre_stack = np.stack([IMG_CACHE_384[_sid][0] for _sid in df_inv_clean["id"]], axis=0)  # (N, 384, 384, 3)
    _post_stack = np.stack([IMG_CACHE_384[_sid][1] for _sid in df_inv_clean["id"]], axis=0)
    GPU_PRE_U8 = torch.from_numpy(_pre_stack).permute(0, 3, 1, 2).contiguous().cuda()
    GPU_POST_U8 = torch.from_numpy(_post_stack).permute(0, 3, 1, 2).contiguous().cuda()
    del _pre_stack, _post_stack

    _GPU_MEAN = IMAGENET_MEAN.cuda().view(1, 3, 1, 1)
    _GPU_STD = IMAGENET_STD.cuda().view(1, 3, 1, 1)

    def _gpu_color_jitter(_x, _bright=0.3, _cont=0.3, _sat=0.2, _hue=0.05):
        _b = _x.shape[0]
        _dev = _x.device
        # Brightness
        _bf = (1.0 - _bright) + 2.0 * _bright * torch.rand(_b, 1, 1, 1, device=_dev, dtype=_x.dtype)
        _x = (_x * _bf).clamp(0.0, 1.0)
        # Contrast
        _gray = 0.2989 * _x[:, 0:1] + 0.5870 * _x[:, 1:2] + 0.1140 * _x[:, 2:3]
        _gmean = _gray.mean(dim=(2, 3), keepdim=True)
        _cf = (1.0 - _cont) + 2.0 * _cont * torch.rand(_b, 1, 1, 1, device=_dev, dtype=_x.dtype)
        _x = (_x * _cf + _gmean * (1.0 - _cf)).clamp(0.0, 1.0)
        # Saturation
        _sf = (1.0 - _sat) + 2.0 * _sat * torch.rand(_b, 1, 1, 1, device=_dev, dtype=_x.dtype)
        _x = (_x * _sf + _gray * (1.0 - _sf)).clamp(0.0, 1.0)
        # Subtle hue channel rotation
        _hf = (2.0 * torch.rand(_b, 1, 1, 1, device=_dev, dtype=_x.dtype) - 1.0) * _hue * _math.pi
        _cos_h = torch.cos(_hf)
        _sin_h = torch.sin(_hf)
        _r, _g, _bl = _x[:, 0:1], _x[:, 1:2], _x[:, 2:3]
        _y = 0.299 * _r + 0.587 * _g + 0.114 * _bl
        _i = 0.596 * _r - 0.274 * _g - 0.322 * _bl
        _q = 0.211 * _r - 0.523 * _g + 0.312 * _bl
        _i2 = _i * _cos_h - _q * _sin_h
        _q2 = _i * _sin_h + _q * _cos_h
        _r2 = _y + 0.956 * _i2 + 0.621 * _q2
        _g2 = _y - 0.272 * _i2 - 0.647 * _q2
        _b2 = _y - 1.106 * _i2 + 1.703 * _q2
        return torch.cat([_r2, _g2, _b2], dim=1).clamp(0.0, 1.0)

    def _gpu_augment_batch(_p1, _p2):
        _b, _, _h, _w = _p1.shape
        _dev = _p1.device
        # Identical horizontal flip
        _fmask = torch.rand(_b, 1, 1, 1, device=_dev) < 0.5
        _p1 = torch.where(_fmask, torch.flip(_p1, dims=[-1]), _p1)
        _p2 = torch.where(_fmask, torch.flip(_p2, dims=[-1]), _p2)
        # Identical small rotation (-10..+10 deg)
        _ang = (torch.rand(_b, device=_dev) * 20.0 - 10.0) * (_math.pi / 180.0)
        _ca, _sa = torch.cos(_ang), torch.sin(_ang)
        _zeros = torch.zeros_like(_ang)
        _theta = torch.stack([
            torch.stack([_ca, -_sa, _zeros], dim=-1),
            torch.stack([_sa, _ca, _zeros], dim=-1),
        ], dim=1)
        _grid = _F.affine_grid(_theta, _p1.size(), align_corners=False)
        _p1 = _F.grid_sample(_p1, _grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        _p2 = _F.grid_sample(_p2, _grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        # Independent ColorJitter
        _p1 = _gpu_color_jitter(_p1)
        _p2 = _gpu_color_jitter(_p2)
        return _p1, _p2

    def compute_class_weights_tensor(df, label_col="vlm_label"):
        _counts = df[label_col].value_counts()
        _w = np.array([1.0 / max(_counts.get(_c, 1), 1) for _c in CLASS_NAMES], dtype=np.float32)
        _w = _w / _w.mean()
        return torch.tensor(_w, dtype=torch.float32, device="cuda")

    def run_5fold_training(
        model_tag,
        model_cls,
        df_folds,
        epochs=30,
        batch_size=32,
        lr=3e-4,
        wd=1e-4,
        warmup_epochs=5,
        patience=7,
        init_ckpt_prefix=None,
        use_sample_weights=False,
        out_suffix="",
    ):
        _t_start = _time.time()
        _art = Path(ARTIFACT_DIR)
        _cw = compute_class_weights_tensor(df_folds)
        _pos_all = torch.tensor([_ID_TO_POS[_sid] for _sid in df_folds["id"]], dtype=torch.long, device="cuda")
        _y_all = torch.tensor(df_folds["vlm_label"].map(LABEL2IDX).values, dtype=torch.long, device="cuda")
        _w_all = torch.tensor(df_folds["sample_weight"].values, dtype=torch.float32, device="cuda") if use_sample_weights else None

        _oof_probs = np.zeros((len(df_folds), len(CLASS_NAMES)), dtype=np.float32)
        _y_true_np = df_folds["vlm_label"].map(LABEL2IDX).values
        _fold_rows = []

        for _fold in range(5):
            _tf_s = _time.time()
            _tr_mask = (df_folds["fold"].values != _fold)
            _va_idx_np = np.where(df_folds["fold"].values == _fold)[0]
            _tr_idx_t = torch.tensor(np.where(_tr_mask)[0], dtype=torch.long, device="cuda")
            _va_idx_t = torch.tensor(_va_idx_np, dtype=torch.long, device="cuda")

            _model = model_cls(num_classes=len(CLASS_NAMES), pretrained=(init_ckpt_prefix is None)).cuda()
            if init_ckpt_prefix is not None:
                _init_p = _art / f"ckpt_{init_ckpt_prefix}_fold{_fold}.pth"
                _model.load_state_dict(torch.load(_init_p, map_location="cuda", weights_only=True))

            _opt = torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=wd)
            _wu = torch.optim.lr_scheduler.LinearLR(_opt, start_factor=0.1, total_iters=max(warmup_epochs, 1))
            _cos = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
            _sched = torch.optim.lr_scheduler.SequentialLR(_opt, schedulers=[_wu, _cos], milestones=[warmup_epochs])
            _scaler = torch.amp.GradScaler("cuda", enabled=False)
            _crit = nn.CrossEntropyLoss(weight=_cw, label_smoothing=0.1, reduction="none" if use_sample_weights else "mean")

            _best_f1 = -1.0
            _best_probs = None
            _best_ep = 0
            _wait = 0
            _ckpt_path = _art / f"ckpt_{model_tag}{out_suffix}_fold{_fold}.pth"
            _n_tr = len(_tr_idx_t)

            for _ep in range(1, epochs + 1):
                _model.train()
                _perm = _tr_idx_t[torch.randperm(_n_tr, device="cuda")]
                for _s in range(0, _n_tr, batch_size):
                    _bi = _perm[_s:_s + batch_size]
                    _pos = _pos_all[_bi]
                    _yb = _y_all[_bi]
                    _p1 = GPU_PRE_U8[_pos].float().div_(255.0)
                    _p2 = GPU_POST_U8[_pos].float().div_(255.0)
                    _p1, _p2 = _gpu_augment_batch(_p1, _p2)
                    _p1 = (_p1 - _GPU_MEAN) / _GPU_STD
                    _p2 = (_p2 - _GPU_MEAN) / _GPU_STD

                    _opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        _logits = _model(_p1, _p2)
                        _loss = _crit(_logits, _yb)
                        if use_sample_weights:
                            _loss = (_loss * _w_all[_bi]).mean()
                    _scaler.scale(_loss).backward()
                    _scaler.step(_opt)
                    _scaler.update()
                _sched.step()

                _model.eval()
                _vp_list = []
                with torch.no_grad():
                    for _s in range(0, len(_va_idx_t), batch_size * 2):
                        _vi = _va_idx_t[_s:_s + batch_size * 2]
                        _pos = _pos_all[_vi]
                        _p1 = (GPU_PRE_U8[_pos].float().div_(255.0) - _GPU_MEAN) / _GPU_STD
                        _p2 = (GPU_POST_U8[_pos].float().div_(255.0) - _GPU_MEAN) / _GPU_STD
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            _pr = _F.softmax(_model(_p1, _p2).float(), dim=-1)
                        _vp_list.append(_pr.cpu().numpy())
                _vp = np.vstack(_vp_list)
                _vy = _y_true_np[_va_idx_np]
                _f1 = float(f1_score(_vy, _vp.argmax(axis=1), average="macro", zero_division=0))

                if _f1 > _best_f1:
                    _best_f1 = _f1
                    _best_probs = _vp
                    _best_ep = _ep
                    _wait = 0
                    torch.save(_model.state_dict(), _ckpt_path)
                else:
                    _wait += 1
                print(f'[{model_tag}{out_suffix} F{_fold}] ep={_ep:02d}/{epochs} val_f1={_f1:.4f} best={_best_f1:.4f}', flush=True)
                if _wait >= patience:
                    break

            _oof_probs[_va_idx_np] = _best_probs
            _fold_sec = round(_time.time() - _tf_s, 1)
            _fold_rows.append({"fold": _fold, "best_epoch": _best_ep, "val_macro_f1": round(_best_f1, 4), "sec": _fold_sec})
            print(f"[{model_tag}{out_suffix}] Fold {_fold} selesai | best_ep={_best_ep} | val_macro_f1={_best_f1:.4f} ({_fold_sec}s)", flush=True)
            del _model, _opt, _sched
            _gc.collect()
            torch.cuda.empty_cache()

        _oof_path = _art / f"oof_{model_tag}{out_suffix}.npy"
        np.save(_oof_path, _oof_probs)
        _oof_pred = _oof_probs.argmax(axis=1)
        _overall_f1 = float(f1_score(_y_true_np, _oof_pred, average="macro", zero_division=0))
        _per_class_f1 = f1_score(_y_true_np, _oof_pred, labels=list(range(6)), average=None, zero_division=0)
        _cls_dict = {CLASS_NAMES[_i]: round(float(_per_class_f1[_i]), 4) for _i in range(6)}

        _fold_df = pd.DataFrame(_fold_rows)
        _log_txt = (
            f"\n## {model_tag}{out_suffix} (epochs={epochs}, lr={lr}, time={_time.time()-_t_start:.1f}s)\n"
            f"- Overall OOF Macro F1: **{_overall_f1:.4f}** | Mean Fold F1: **{_fold_df['val_macro_f1'].mean():.4f}**\n"
            f"- Per-Class OOF F1: `{_cls_dict}`\n"
            + _fold_df.to_string(index=False) + "\n"
        )
        with open(_art / "training_log.md", "a", encoding="utf-8") as _f:
            _f.write(_log_txt)

        print(f"=== {model_tag}{out_suffix} SELESAI ({_time.time()-_t_start:.1f}s) ===", flush=True)
        print(f"Overall OOF Macro F1: {_overall_f1:.4f} | Mean Fold F1: {_fold_df['val_macro_f1'].mean():.4f}", flush=True)
        print("Per-class OOF F1:", _cls_dict, flush=True)
        mo.status.toast(title=f"Training selesai: {model_tag}{out_suffix}", description=f"OOF Macro F1 = {_overall_f1:.4f}")
        return _oof_probs, _overall_f1, _cls_dict, _fold_df

    with open(Path(ARTIFACT_DIR) / "training_log.md", "w", encoding="utf-8") as _f:
        _f.write("# Training Log — Siamese Dual Fusion (Fenomena)\n")

    mo.status.toast(title="Training Dual Siamese Dimulai", description="Melatih EfficientNet-B4 (5-Fold)")
    oof_effb4, f1_effb4_iter1, cls_f1_effb4_iter1, fold_df_effb4 = run_5fold_training(
        model_tag="effb4",
        model_cls=SiameseDualFusion,
        df_folds=df_train_folds,
        epochs=30,
        batch_size=32,
        lr=3e-4,
        wd=1e-4,
        warmup_epochs=5,
        patience=7,
    )
    fold_df_effb4
    return


if __name__ == "__main__":
    app.run()
