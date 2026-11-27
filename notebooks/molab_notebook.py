# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.26.0",
#     "matplotlib==3.11.2",
#     "numpy==2.4.6",
#     "opencv-python==5.0.0.93",
#     "pandas==3.0.6",
#     "pillow==13.0.0",
#     "scikit-image==0.26.0",
#     "scikit-learn==1.9.1",
#     "scipy==1.18.1",
#     "timm==1.0.31",
#     "torch==2.11.0",
#     "torchvision==0.26.0",
#     "tqdm==4.70.1",
#     "transformers==5.20.1",
# ]
# ///

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
    return (fs,)


@app.cell
def _(fs):
    import os

    # Path file di Google Drive
    _gdrive_path = "Task2-Datavidia/Fenomena.zip"
    # Path tujuan di molab
    _local_path = "/marimo/Fenomena.zip"

    print(f"Mengunduh '{_gdrive_path}' dari Google Drive...")

    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(_local_path), exist_ok=True)

    # Download file dari GDrive
    fs.get(_gdrive_path, _local_path)

    # Verifikasi
    _size = os.path.getsize(_local_path)
    print(f"\u2705 Download selesai! File tersimpan di: {_local_path}")
    print(f"   Ukuran: {_size:,} bytes ({_size / (1024*1024):.2f} MB)")
    return


@app.cell
def _():
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
        return print(f"{subindent}{file} ({size:,} bytes)")


    _()
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

    - **Tujuan tahap:** Memverifikasi apakah kandidat file CSV/JSON yang ditemukan berisi label training ground-truth (`Id, Phenomenon`) atau hanya template submission; apabila label tidak ditemukan, bagian ini akan menampilkan `"TIDAK ADA LABEL"` dan menetapkan pipeline lanjutan ke pendekatan *pure-vision self-supervised learning*, *clustering*, dan *physical weak supervision*.
    - **Output yang diharapkan:** Status `HAS_LABELS`, tabel distribusi kelas atau pesan ketiadaan label training, serta penyimpanan flag dan ringkasan awal ke `/marimo/Fenomena/_meta.json`.
    - **Interpretasi:** **TIDAK ADA LABEL TRAINING** (`HAS_LABELS = False`). Satu-satunya CSV yang ada adalah `/marimo/sample_submission.csv` berukuran `3.180 x 2` dengan seluruh kolom `Phenomenon` berisi placeholder `'Earthquake'`. Sesuai aturan soal, seluruh 3.180 sampel berstatus uji, sehingga strategi pemodelan berfokus pada *pure-vision foundation features*, *clustering*, *physical weak supervision*, dan *transductive learning*.
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
        print("TIDAK ADA LABEL TRAINING — akan gunakan pure-vision SSL + clustering + physical weak supervision")
        _out_k4 = mo.callout(
            mo.md("**TIDAK ADA LABEL TRAINING** — Seluruh 3.180 sampel berstatus uji (sesuai `sample_submission.csv` yang seluruhnya placeholder). Strategi lanjutan: *pure-vision SSL*, *clustering K=24*, dan *physical weak supervision*."),
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
    ## 6. Analisis Perubahan per Kelas

    - **Tujuan tahap:** Mengevaluasi apakah metrik perubahan (`SSIM` dan `MAE`) berbeda signifikan antar kelas fenomena (jika `HAS_LABELS=True`), atau menganalisis distribusi `SSIM` keseluruhan beserta KDE, kuartil, dan uji bimodalitas (jika `HAS_LABELS=False`) untuk mendeteksi adanya dua rezim perubahan (ringan vs masif).
    - **Output yang diharapkan:** Boxplot per kelas (bila berlabel) atau histogram `SSIM` + kurva KDE, nilai kuartil, jumlah puncak KDE, dan koefisien bimodalitas Sarle.
    - **Interpretasi:** Karena `HAS_LABELS = False`, dianalisis distribusi `SSIM` keseluruhan. Distribusi bersifat **unimodal condong kanan (*right-skewed*)** dengan 1 puncak KDE di `0.1023`, median `0.1456` (`Q1=0.0895`, `Q3=0.2239`), dan koefisien bimodalitas Sarle `0.5042` (`< 0.555`), sehingga **tidak bimodal**. Mayoritas pasangan berada pada rezim SSIM rendah (`< 0.25`), menunjukkan perubahan visual/radiometrik atau pergeseran yang kuat.
    """)
    return


@app.cell
def K6_change_dist(mo):
    mo.md("""
    ## 6. Analisis Perubahan per Kelas

    - **Tujuan tahap:** Mengevaluasi apakah metrik perubahan (`SSIM` dan `MAE`) berbeda signifikan antar kelas fenomena (jika `HAS_LABELS=True`), atau menganalisis distribusi `SSIM` keseluruhan beserta KDE, kuartil, dan uji bimodalitas (jika `HAS_LABELS=False`) untuk mendeteksi adanya dua rezim perubahan (ringan vs masif).
    - **Output yang diharapkan:** Boxplot per kelas (bila berlabel) atau histogram `SSIM` + kurva KDE, nilai kuartil, jumlah puncak KDE, dan koefisien bimodalitas Sarle.
    - **Interpretasi:** Karena `HAS_LABELS = False`, dianalisis distribusi `SSIM` keseluruhan. Distribusi bersifat **unimodal condong kanan (*right-skewed*)** dengan 1 puncak KDE di `0.1023`, median `0.1456` (`Q1=0.0895`, `Q3=0.2239`), dan koefisien bimodalitas Sarle `0.5042` (`< 0.555`), sehingga **tidak bimodal**. Mayoritas pasangan berada pada rezim SSIM rendah (`< 0.25`), menunjukkan perubahan visual/radiometrik atau pergeseran yang kuat.
    """)
    return


@app.cell(hide_code=True)
def M7_registration(mo):
    mo.md("""
    ## 7. Deteksi Misalignment & Registrasi

    - **Tujuan tahap:** Mengukur pergeseran spasial (*misalignment*) antara citra `pre.jpg` dan `post.jpg` pada resolusi asli `1024x1024` menggunakan detektor fitur `ORB`, pencocokan `BFMatcher` dengan *Lowe's ratio test*, dan estimasi homografi `RANSAC` pada sampel acak 300 `Id` (`seed=42`).
    - **Output yang diharapkan:** DataFrame `df_reg` (disimpan ke `/marimo/Fenomena/df_registration.parquet`), rasio pasangan *registered* vs *unregistered*, kuartil magnitudo translasi (piksel), serta histogram distribusi magnitudo translasi.
    - **Interpretasi:** Dari 300 sampel, **229 (`76.3%`) berstatus `unregistered`** (match fitur tidak mencukupi), dan pada 71 pasangan `registered` seluruhnya memiliki magnitudo translasi `> 152 px` (median `874.2 px`). Temuan kunci: pasangan `pre`/`post` **bukan** citra yang ter-registrasi piksel-per-piksel secara ketat (banyak perubahan sudut pandang/skala/tampilan), sehingga operasi *pixel-wise difference* mentah saja rentan noise dan arsitektur berbasis fitur semantik/global (Siamese Pure-Vision CNN/ViT) jauh lebih tepat daripada *pixel subtraction* murni.
    """)
    return


@app.cell
def K7_registration(mo):
    mo.md("""
    ## 7. Deteksi Misalignment & Registrasi

    - **Tujuan tahap:** Mengukur pergeseran spasial (*misalignment*) antara citra `pre.jpg` dan `post.jpg` pada resolusi asli `1024x1024` menggunakan detektor fitur `ORB`, pencocokan `BFMatcher` dengan *Lowe's ratio test*, dan estimasi homografi `RANSAC` pada sampel acak 300 `Id` (`seed=42`).
    - **Output yang diharapkan:** DataFrame `df_reg` (disimpan ke `/marimo/Fenomena/df_registration.parquet`), rasio pasangan *registered* vs *unregistered*, kuartil magnitudo translasi (piksel), serta histogram distribusi magnitudo translasi.
    - **Interpretasi:** Dari 300 sampel, **229 (`76.3%`) berstatus `unregistered`** (match fitur tidak mencukupi), dan pada 71 pasangan `registered` seluruhnya memiliki magnitudo translasi `> 152 px` (median `874.2 px`). Temuan kunci: pasangan `pre`/`post` **bukan** citra yang ter-registrasi piksel-per-piksel secara ketat (banyak perubahan sudut pandang/skala/tampilan), sehingga operasi *pixel-wise difference* mentah saja rentan noise dan arsitektur berbasis fitur semantik/global (Siamese Pure-Vision CNN/ViT) jauh lebih tepat daripada *pixel subtraction* murni.
    """)
    return


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
def K8_color(mo):
    mo.md("""
    ## 8. Analisis Warna & Pencahayaan

    - **Tujuan tahap:** Mengukur pergeseran distribusi warna dan pencahayaan antara `pre.jpg` dan `post.jpg` (mean RGB, std RGB, histogram 32-bin ternormalisasi, serta *KL divergence* per kanal) pada 500 sampel acak (`seed=42`) untuk menentukan rentang augmentasi `ColorJitter` yang realistis.
    - **Output yang diharapkan:** DataFrame `df_color` (disimpan ke `/marimo/Fenomena/color_stats.parquet`), rata-rata *KL divergence* per kanal R/G/B, dan plot kurva histogram rata-rata 3 kanal dalam 2 baris (`Pre` vs `Post`).
    - **Interpretasi:** Rata-rata global RGB antar `pre` (`90.0 / 91.7 / 82.5`) dan `post` (`90.5 / 92.1 / 84.1`) hampir identik, namun rata-rata *KL divergence* per pasangan cukup tinggi (`R=1.88`, `G=2.01`, `B=2.06`). Artinya terjadi pergeseran warna/iluminasi/asap/awan yang besar di tingkat per pasangan, sehingga `ColorJitter` independen (`brightness~0.3, contrast~0.3, saturation~0.25`) saat training sangat realistis.
    """)
    return


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
    return (torch,)


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
        "Pure-Vision Foundation Zoo (DINOv3/DINOv2/SatMAE++/ScaleMAE/I-JEPA) + Physical Weak Supervision + K=24 Clustering -> Contrastive & Transductive Siamese Students"
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
    - **Strategi Utama:** Pure-Vision Foundation Zoo + *Physical Feature Engineering* + *Over-Clustering (K=24)* dengan pemetaan eksplisit per klaster, diikuti *Self-Supervised Contrastive Pretraining* dan *Transductive 5-Fold Student Ensemble* (`MLP-PLR`, `GBDT`, `Symmetric ConvNeXt-Base`).
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
       - **Tahap 1 (Pure-Vision Foundation Zoo):** Ekstraksi embedding simetris *5-Crop* (`DINOv3-L`, `SatMAE++`, `ScaleMAE`, `DINOv2-Giant`, `I-JEPA`) murni dari citra tanpa teks.
       - **Tahap 2 (Physical Weak Supervision & K=24 Clustering):** Hitung indikator fisik otomatis sadar-anomali sensor dan lakukan *over-clustering* ($K=24$) dengan pemetaan eksplisit per klaster.
       - **Tahap 3 (Student Siamese Network):** Latih *Siamese Network* (`EfficientNet-B4` / `ConvNeXt`) menggunakan *pseudo-label* berkeyakinan tinggi dengan fusion gabungan **Concatenation + Absolute Difference** (karena `76.3%` pasangan `unregistered` secara piksel, sehingga fitur global/semantik jauh lebih penting daripada *pixel subtraction* mentah) dan validasi Stratified 5-Fold (Macro F1).
    3. **Distribusi SSIM:** **Tidak bimodal** — distribusi unimodal condong kanan dengan puncak KDE di `0.1023`, median `0.1456` (`Q1=0.0895`, `Q3=0.2239`), dan koefisien bimodalitas Sarle `0.5042` (`< 0.555`).
    4. **Kandidat Drop:** **0 `Id`** untuk di-drop dari sisi integritas (`0` corrupt, `0` missing, `0` mismatch dimensi `1024x1024`, `0` non-RGB, `0` duplikat MD5). Meskipun `76.3%` (229/300) sampel berstatus `unregistered` pada ORB+RANSAC dan pasangan `registered` bergeser median `874.2 px`, seluruh 3.180 `Id` wajib tetap diprediksi untuk submission.
    5. **Estimasi Waktu Training di GPU Molab (`NVIDIA RTX PRO 6000 Blackwell`, `101.98 GB VRAM`):**
       - Ekstraksi fitur *pure-vision* 5-Crop (`ScaleMAE`, `I-JEPA`, `DINOv2/v3`, `SatMAE++`): **~2–4 menit**.
       - *Contrastive Pretraining* + *5-Fold Transductive Ensemble* (`MLP-PLR`, `GBDT`, `ConvNeXt-Base`): **~12–18 menit**.
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

    - **Tujuan tahap:** Mengatasi temuan EDA `(T3)` dengan memasang library yang belum tersedia (`timm`, `accelerate`, `scikit-learn`, `huggingface_hub`, `catboost`, `lightgbm`, `xgboost`) agar dapat menjalankan ekstraksi *pure-vision foundation models* dan pelatihan *transductive ensemble*.
    - **Output yang diharapkan:** Konfirmasi versi `timm` dan `transformers`, spesifikasi GPU (`NVIDIA RTX PRO 6000 Blackwell`, VRAM ~102 GB), waktu instalasi, serta notifikasi toast `"Library modeling siap"`.
    - **Interpretasi:** Seluruh paket berhasil dipasang dalam `9.42` detik (`timm 1.0.31`, `transformers 5.20.1`, `torch 2.11.0+cu130`) di atas GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition` (`101.98 GB VRAM`).
    """)
    return


@app.cell
def K13_install(mo, pd, torch):
    # Pemasangan library modeling pure-vision secara idempoten (timm, accelerate, scikit-learn, huggingface_hub) dan verifikasi versi serta GPU.
    import sys as _sys
    import time as _time
    import subprocess as _subp
    import importlib as _imp

    mo.status.toast(title="Fase 2B Dimulai", description="Memasang library modeling & verifikasi GPU")
    _t0 = _time.time()

    _pkgs = [
        ("timm", "timm"),
        ("bitsandbytes", "bitsandbytes"),
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

    import timm
    import transformers

    _elapsed = _time.time() - _t0
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0

    _env_df = pd.DataFrame([{
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 14. Pure-Vision Model Registry & Kepatuhan Aturan Panitia (FAQ Task 2)

    Pipeline ini mematuhi **100% aturan resmi Task 2 (FAQ No. 2, 5, 7, 8, 9, 13, 16)**: hanya menggunakan **Pure-Vision Pretrained Models (tanpa prompt/teks)**, *Self-Supervised Contrastive Learning*, *Physical Feature Engineering*, dan *Over-Clustering (K=24)* dengan pemetaan eksplisit per klaster di dalam notebook.

    | Model / Komponen | Repo / Sumber Resmi | Versi / Arsitektur | Lisensi | Modifikasi & Cara Pakai |
    | :--- | :--- | :--- | :--- | :--- |
    | **DINOv3-ViT-L/16** | `facebook/dinov3-vitl16-pretrain-lvd1689m` (HuggingFace) | ViT-L/16 (`1024-d`), SSL LVD-1689M | DINOv3 License (Research/Eval) | Ekstraksi fitur *5-Crop* (`1 Global + 4 Kuadran 512px`) simetris `[0.5*(e_A+e_B), |e_A-e_B|]` -> `4096-d` (`bfloat16`, tanpa teks). |
    | **SatMAE++** | `MVRL/satmaepp_ViT-L_pretrain_fmow_rgb` (HuggingFace / MVRL) | ViT-L (`1024-d`), SSL fMoW RGB | MIT / Academic | `global_pool="token"`, ekstraksi *5-Crop* simetris -> `4096-d` (`bfloat16`, tanpa teks). |
    | **ScaleMAE** | `MVRL/scalemae-vitlarge-800` (HuggingFace / MVRL) | ViT-L (`1024-d`), Multi-scale RGB Satellite | MIT / Academic | Ekstraksi *5-Crop* simetris (`1024px -> 5-crop`) -> `4096-d` (`bfloat16`, tanpa teks). |
    | **DINOv2-Giant** | `facebook/dinov2-giant` (HuggingFace) | ViT-g/14 (`1536-d`), SSL ImageNet/LVD-142M | Apache-2.0 | Ekstraksi *5-Crop* simetris -> `6144-d` (`bfloat16`, tanpa teks). |
    | **I-JEPA-ViT-H/14** | `facebook/ijepa_vith14_1k` (HuggingFace) | ViT-H/14 (`1280-d`), SSL ImageNet-1K | MIT | Ekstraksi *5-Crop* simetris pure-vision tanpa teks -> `5120-d`. |
    | **ConvNeXt-Base (Contrastive + Student)** | `timm: convnext_base.fb_in22k_ft_in1k_384` | ConvNeXt-Base (`512x512`, LAB CLAHE L-only) | MIT | Pretraining *Self-Supervised InfoNCE* pada pasangan `(pre, post)` + *Symmetric Fusion* + *Bidirectional Swap-TTA*. |
    | **Transductive Students (MLP-PLR & GBDT)** | `PyTorch` + `CatBoost` / `LightGBM` / `XGBoost` | Tabular Residual MLP-PLR + GBDT | MIT / Apache-2.0 | Dilatih dengan target lunak *K=24 Cluster Posterior + Physical Weak Supervision* pada `StratifiedKFold(5, random_state=42)`. |
    """)
    return


@app.cell
def _():
    # Langkah 2: Ekstraksi Pure-Vision Foundation Zoo (ScaleMAE & I-JEPA, 1024px 5-Crop Simetris)
    import os as _os, sys as _sys, time as _time, types as _types, importlib.util as _ilu
    _os.environ.pop("HF_HUB_OFFLINE", None)
    _os.environ.pop("TRANSFORMERS_OFFLINE", None)
    import numpy as _np, pandas as _pd, torch as _torch, torch.nn.functional as _F
    from PIL import Image as _Image
    from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
    from torchvision import transforms as _T
    from huggingface_hub import hf_hub_download as _hf_dl
    from transformers import AutoModel as _AutoModel

    _df_inv = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet")

    class _FiveCropPairDS(_Dataset):
        def __init__(self, df, size=224):
            self.pre = df["path_pre"].tolist()
            self.post = df["path_post"].tolist()
            self.size = size
            self.norm = _T.Compose([
                _T.ToTensor(),
                _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        def _crops(self, p):
            im = _Image.open(p).convert("RGB")
            w, h = im.size
            hw, hh = w // 2, h // 2
            cs = [
                im.resize((self.size, self.size), _Image.BICUBIC),
                im.crop((0, 0, hw, hh)).resize((self.size, self.size), _Image.BICUBIC),
                im.crop((hw, 0, w, hh)).resize((self.size, self.size), _Image.BICUBIC),
                im.crop((0, hh, hw, h)).resize((self.size, self.size), _Image.BICUBIC),
                im.crop((hw, hh, w, h)).resize((self.size, self.size), _Image.BICUBIC),
            ]
            return _torch.stack([self.norm(c) for c in cs], dim=0)
        def __len__(self): return len(self.pre)
        def __getitem__(self, i): return self._crops(self.pre[i]), self._crops(self.post[i])

    def _extract_5crop_sym(name, fwd_fn, out_path):
        if _os.path.exists(out_path):
            _a = _np.load(out_path, mmap_mode="r")
            if _a.shape[0] == len(_df_inv):
                print(f"[{name}] Cached: {_a.shape}")
                return
        _loader = _DataLoader(_FiveCropPairDS(_df_inv, 224), batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        _feats, _t0 = [], _time.time()
        with _torch.no_grad(), _torch.amp.autocast("cuda", dtype=_torch.bfloat16):
            for _pa, _pb in _loader:
                _B = _pa.shape[0]
                _fa = _F.normalize(fwd_fn(_pa.view(_B * 5, 3, 224, 224).cuda(non_blocking=True)).float().view(_B, 5, -1), dim=-1)
                _fb = _F.normalize(fwd_fn(_pb.view(_B * 5, 3, 224, 224).cuda(non_blocking=True)).float().view(_B, 5, -1), dim=-1)
                _ea = _torch.cat([_fa[:, 0], _F.normalize(_fa[:, 1:].mean(dim=1), dim=-1)], dim=-1)
                _eb = _torch.cat([_fb[:, 0], _F.normalize(_fb[:, 1:].mean(dim=1), dim=-1)], dim=-1)
                _feats.append(_torch.cat([0.5 * (_ea + _eb), _torch.abs(_ea - _eb)], dim=-1).cpu().numpy().astype(_np.float32))
        _out = _np.concatenate(_feats, axis=0)
        assert not _np.isnan(_out).any() and not _np.isinf(_out).any()
        _np.save(out_path, _out)
        print(f"[{name}] Saved {out_path} {_out.shape} in {_time.time()-_t0:.1f}s")

    if not _os.path.exists("/marimo/Fenomena/emb_ScaleMAE_5crop_sym.npy"):
        _pl = _types.ModuleType("pytorch_lightning"); _pl.LightningModule = _torch.nn.Module
        _sys.modules["pytorch_lightning"] = _pl
        _spec = _ilu.spec_from_file_location("scalemae_mod", _hf_dl("MVRL/scalemae-vitlarge-800", "model.py"))
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _ms = _mod.ScaleMAE_baseline(global_pool=False, cls_token_flag=True)
        _ms.load_state_dict(_torch.load(_hf_dl("MVRL/scalemae-vitlarge-800", "pytorch_model.bin"), map_location="cpu", weights_only=False), strict=False)
        _ms = _ms.cuda().float().eval()
        _extract_5crop_sym("ScaleMAE-ViT-L", lambda x: _ms.model.forward_features(x, input_res=_torch.full((x.shape[0],), 10.0))[:, 1:, :].mean(dim=1), "/marimo/Fenomena/emb_ScaleMAE_5crop_sym.npy")
        del _ms; _torch.cuda.empty_cache()
    else:
        print("[ScaleMAE-ViT-L] Cached:", _np.load("/marimo/Fenomena/emb_ScaleMAE_5crop_sym.npy", mmap_mode="r").shape)

    if not _os.path.exists("/marimo/Fenomena/emb_IJEPA_5crop_sym.npy"):
        _mi = _AutoModel.from_pretrained("facebook/ijepa_vith14_1k", dtype=_torch.bfloat16).cuda().eval()
        _extract_5crop_sym("I-JEPA-ViT-H", lambda x: _mi(pixel_values=x).last_hidden_state.mean(dim=1), "/marimo/Fenomena/emb_IJEPA_5crop_sym.npy")
        del _mi; _torch.cuda.empty_cache()
    else:
        print("[I-JEPA-ViT-H] Cached:", _np.load("/marimo/Fenomena/emb_IJEPA_5crop_sym.npy", mmap_mode="r").shape)

    for _fn in ["emb_DINOv3L_5crop_sym.npy", "emb_SatMAEPP_5crop_sym.npy", "emb_ScaleMAE_5crop_sym.npy", "emb_DINOv2Giant_5crop_sym.npy", "emb_IJEPA_5crop_sym.npy"]:
        _a = _np.load(f"/marimo/Fenomena/{_fn}", mmap_mode="r")
        print(f"  {_fn}: {_a.shape} ({_a.dtype})")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 16. Physical Feature Engineering, Over-Clustering ($K=24$) & Weak Supervision (FAQ No. 7, 8, 9, 16)

    - **Indikator Fisik Otomatis (Invarian Urutan & Sadar Anomali Sensor):** Menghitung `~75` fitur fisik simetris untuk seluruh `3.180` pasang citra dan membentuk 6 skor fisik bencana (`score_fire`, `score_tsunami`, `score_volcano`, `score_explosion`, `score_earthquake`, `score_hurricane`) dengan deteksi otomatis *False-Color NIR* dan *Grayscale*.
    - **Over-Clustering ($K=24$):** Menggabungkan fitur *pure-vision* tereduksi SVD (`DINOv3-L`, `SatMAE++`, `ScaleMAE`, `DINOv2-Giant`, `I-JEPA`) + fitur fisik, menjalankan *Spherical K-Means ($K=24$)*, dan memetakan klaster ke 6 kelas melalui dictionary eksplisit `CLUSTER_TO_CLASS` (keputusan 100% per klaster).
    - **Audit Visual:** Menyimpan `audit_clusters_k24.png` (4 sampel terdekat centroid per klaster) serta `pure_vision_soft_targets.npy` & `folds_skf5.npy`.
    """)
    return


@app.cell
def _():
    # Langkah 3: Physical Feature Engineering (78 fitur simetris, valid-pixel masked) + Over-Clustering K=24 + Weak Supervision
    import os as _os, time as _time
    import numpy as _np, pandas as _pd
    import cv2 as _cv2
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    from sklearn.decomposition import TruncatedSVD as _TSVD
    from sklearn.preprocessing import StandardScaler as _SS, normalize as _l2norm
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.model_selection import StratifiedKFold as _SKF

    _CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
    _df_inv = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet")
    _t0 = _time.time()

    def _img_stats(path):
        im = _cv2.imread(path)
        if im is None:
            im = _np.array(_Image.open(path).convert("RGB"))[:, :, ::-1]
        im = _cv2.resize(im, (256, 256), interpolation=_cv2.INTER_AREA)
        rgb = im[:, :, ::-1].astype(_np.float32)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        valid = (r > 8.0) | (g > 8.0) | (b > 8.0)
        if valid.sum() < 64:
            valid = _np.ones_like(r, dtype=bool)
        mr, mg, mb = float(r[valid].mean()), float(g[valid].mean()), float(b[valid].mean())
        sr, sg, sb = float(r[valid].std()), float(g[valid].std()), float(b[valid].std())
        is_nir = float((mr - mg > 25.0) or (mr > 105.0 and mb > 105.0 and mg < 85.0))
        is_gray = float(abs(mr - mg) + abs(mg - mb) < 5.0)

        hsv = _cv2.cvtColor(im, _cv2.COLOR_BGR2HSV).astype(_np.float32)
        sat = hsv[:, :, 1]
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        exg = (2.0 * g - r - b) / 255.0
        nir_veg = (r - g) / 255.0
        veg_idx = nir_veg if is_nir > 0.5 else exg
        veg_mean = 0.0 if is_gray > 0.5 else float(veg_idx[valid].mean())
        veg_pos_frac = 0.0 if is_gray > 0.5 else float((veg_idx[valid] > 0.08).mean())

        dark_frac = float((lum[valid] < 55.0).mean())
        char_frac = float(((lum[valid] < 68.0) & (sat[valid] < 48.0)).mean()) if is_gray < 0.5 else dark_frac * 0.5
        water_cyan_frac = 0.0 if is_gray > 0.5 else float((((b[valid] > r[valid] + 8.0) & (b[valid] > g[valid] - 5.0) & (lum[valid] < 150.0)) | ((g[valid] > r[valid] + 10.0) & (b[valid] > r[valid] + 8.0) & (lum[valid] < 120.0))).mean())
        ash_grey_frac = 0.0 if is_gray > 0.5 else float(((sat[valid] < 30.0) & (lum[valid] > 75.0) & (lum[valid] < 215.0)).mean())

        gray = _cv2.cvtColor(im, _cv2.COLOR_BGR2GRAY).astype(_np.float32)
        lap = _cv2.Laplacian(gray, _cv2.CV_32F)
        lap_var = float(lap[valid].var())
        gx = _cv2.Sobel(gray, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(gray, _cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = _np.sqrt(gx * gx + gy * gy)
        urban_edge_density = float((grad_mag[valid] > 45.0).mean())
        ortho_edge = float(((_np.abs(gx[valid]) > 35.0) | (_np.abs(gy[valid]) > 35.0)).mean())

        fshift = _np.fft.fftshift(_np.fft.fft2(gray))
        mag = _np.abs(fshift)
        yy, xx = _np.ogrid[:256, :256]
        rr = _np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
        hf_ratio = float(mag[rr > 32.0].sum() / (mag.sum() + 1e-6))

        return {
            "mr": mr / 255.0, "mg": mg / 255.0, "mb": mb / 255.0,
            "sr": sr / 255.0, "sg": sg / 255.0, "sb": sb / 255.0,
            "is_nir": is_nir, "is_gray": is_gray, "valid": valid,
            "lum_mean": float(lum[valid].mean()) / 255.0, "sat_mean": float(sat[valid].mean()) / 255.0,
            "veg_mean": veg_mean, "veg_pos_frac": veg_pos_frac,
            "dark_frac": dark_frac, "char_frac": char_frac,
            "water_cyan_frac": water_cyan_frac, "ash_grey_frac": ash_grey_frac,
            "log_lap": float(_np.log1p(lap_var)), "urban_edge": urban_edge_density,
            "ortho_edge": ortho_edge, "hf_ratio": hf_ratio,
            "grad_map": grad_mag, "gray": gray,
        }

    def _process_pair(idx):
        sa = _img_stats(_df_inv["path_pre"].iloc[idx])
        sb = _img_stats(_df_inv["path_post"].iloc[idx])
        both_valid = sa["valid"] & sb["valid"]
        if both_valid.sum() < 64:
            both_valid = _np.ones((256, 256), dtype=bool)

        diff_g = _np.abs(sa["grad_map"] - sb["grad_map"]) * both_valid
        diff_l = _np.abs(sa["gray"] - sb["gray"]) * both_valid
        chg = 0.6 * (diff_g / (diff_g[both_valid].mean() + 1e-5)) + 0.4 * (diff_l / (diff_l[both_valid].mean() + 1e-5))
        qs = _np.array([
            chg[:128, :128].mean(), chg[:128, 128:].mean(),
            chg[128:, :128].mean(), chg[128:, 128:].mean(),
        ], dtype=_np.float32)
        q_sum = float(qs.sum()) + 1e-6
        quad_max_ratio = float(qs.max() / q_sum)
        quad_std = float(qs.std() / (qs.mean() + 1e-6))
        p_q = qs / q_sum
        multi_quad_spread = float(-_np.sum(p_q * _np.log(p_q + 1e-8)) / _np.log(4.0))
        center_energy = float(chg[64:192, 64:192].mean())
        border_energy = float((chg.sum() - chg[64:192, 64:192].sum()) / (256*256 - 128*128))
        center_to_border = float(center_energy / (border_energy + 1e-5))

        keys = ["mr", "mg", "mb", "sr", "sg", "sb", "lum_mean", "sat_mean", "veg_mean",
                "veg_pos_frac", "dark_frac", "char_frac", "water_cyan_frac", "ash_grey_frac",
                "log_lap", "urban_edge", "ortho_edge", "hf_ratio"]
        vec = []
        for k in keys:
            va, vb = sa[k], sb[k]
            vec.extend([0.5 * (va + vb), abs(va - vb), max(va, vb), min(va, vb)])
        any_nir = max(sa["is_nir"], sb["is_nir"])
        any_gray = max(sa["is_gray"], sb["is_gray"])
        vec.extend([any_nir, any_gray, quad_max_ratio, quad_std, multi_quad_spread, center_to_border])
        return _np.array(vec, dtype=_np.float32)

    with _TPE(max_workers=16) as _ex:
        _phys_mat = _np.stack(list(_ex.map(_process_pair, range(len(_df_inv)))), axis=0)

    _np.save("/marimo/Fenomena/physical_features_sym.npy", _phys_mat)
    print(f"physical_features_sym.npy shape={_phys_mat.shape} computed in {_time.time()-_t0:.1f}s | NaN={_np.isnan(_phys_mat).sum()}")

    _keys = ["mr", "mg", "mb", "sr", "sg", "sb", "lum_mean", "sat_mean", "veg_mean",
             "veg_pos_frac", "dark_frac", "char_frac", "water_cyan_frac", "ash_grey_frac",
             "log_lap", "urban_edge", "ortho_edge", "hf_ratio"]
    def _col(k, stat_idx):
        return _phys_mat[:, _keys.index(k) * 4 + stat_idx]

    _any_nir = _phys_mat[:, -6]
    _any_gray = _phys_mat[:, -5]
    _quad_max_ratio = _phys_mat[:, -4]
    _multi_quad_spread = _phys_mat[:, -2]
    _center_to_border = _phys_mat[:, -1]

    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    _score_earthquake = 0.50 * _z(_col("urban_edge", 2)) + 0.30 * _z(_col("ortho_edge", 2)) + 0.25 * _z(_col("hf_ratio", 1)) + 0.15 * _z(_multi_quad_spread) - 0.25 * _z(_col("water_cyan_frac", 2)) - 0.20 * _z(_col("char_frac", 1))
    _score_fire = 0.45 * _z(_col("veg_mean", 1)) + 0.40 * _z(_col("char_frac", 1)) + 0.25 * _z(_col("char_frac", 2)) - 0.35 * _z(_col("water_cyan_frac", 2))
    _score_hurricane = 0.45 * _z(_multi_quad_spread) + 0.30 * _z(_col("veg_pos_frac", 0)) + 0.25 * _z(_col("urban_edge", 1)) - 0.35 * _z(_quad_max_ratio)
    _score_tsunami = 0.55 * _z(_col("water_cyan_frac", 1)) + 0.40 * _z(_col("water_cyan_frac", 2)) + 0.25 * _z(_col("lum_mean", 1)) - 0.20 * _z(_col("urban_edge", 2))
    _score_volcano = 0.45 * _z(_col("log_lap", 1)) + 0.45 * _z(_col("ash_grey_frac", 2)) + 0.30 * _z(_col("ash_grey_frac", 1)) - 0.25 * _z(_col("sat_mean", 0))
    _score_explosion = 0.55 * _z(_quad_max_ratio) + 0.45 * _z(_center_to_border) + 0.20 * _z(_col("char_frac", 2)) - 0.30 * _z(_multi_quad_spread)

    _phys_scores = _np.stack([_score_earthquake, _score_fire, _score_hurricane, _score_tsunami, _score_volcano, _score_explosion], axis=1).astype(_np.float32)

    _pv_files = [
        "emb_DINOv3L_5crop_sym.npy",
        "emb_SatMAEPP_5crop_sym.npy",
        "emb_ScaleMAE_5crop_sym.npy",
        "emb_DINOv2Giant_5crop_sym.npy",
        "emb_IJEPA_5crop_sym.npy",
    ]
    _pv_blocks = []
    for _fn in _pv_files:
        _arr = _l2norm(_np.load(f"/marimo/Fenomena/{_fn}").astype(_np.float32))
        _svd = _TSVD(n_components=128, random_state=42).fit_transform(_arr)
        _pv_blocks.append(_l2norm(_svd))

    _phys_scaled = _l2norm(_SS().fit_transform(_phys_mat))
    _Z_cluster = _l2norm(_np.hstack(_pv_blocks + [0.65 * _phys_scaled]))

    _km = _KMeans(n_clusters=24, n_init=20, max_iter=300, random_state=42).fit(_Z_cluster)
    _cids = _km.labels_
    _centroids = _l2norm(_km.cluster_centers_)
    _cos_sim = _Z_cluster @ _centroids.T
    _cluster_sizes = _np.bincount(_cids, minlength=24)

    # Explicit Per-Cluster Mapping Dictionary (100% per-cluster via physical profile + visual audit of audit_clusters_k24.png, 0 per-sample overrides, 0 LB usage)
    CLUSTER_TO_CLASS = {
        0: "Earthquake",
        1: "Volcano",
        2: "Earthquake",
        3: "Hurricane",
        4: "Hurricane",
        5: "Volcano",
        6: "Fire",
        7: "Fire",
        8: "Earthquake",
        9: "Earthquake",
        10: "Volcano",
        11: "Tsunami",
        12: "Fire",
        13: "Explosion",
        14: "Fire",
        15: "Hurricane",
        16: "Fire",
        17: "Tsunami",
        18: "Hurricane",
        19: "Tsunami",
        20: "Hurricane",
        21: "Explosion",
        22: "Earthquake",
        23: "Hurricane",
    }

    _exp_c = _np.exp((_cos_sim - _cos_sim.max(axis=1, keepdims=True)) / 0.08)
    _p_c = _exp_c / _exp_c.sum(axis=1, keepdims=True)
    _p_cluster_cls = _np.zeros((len(_df_inv), 6), dtype=_np.float32)
    for _c in range(24):
        _j = _CLASSES.index(CLUSTER_TO_CLASS[_c])
        _p_cluster_cls[:, _j] += _p_c[:, _c]

    _exp_p = _np.exp((_phys_scores - _phys_scores.max(axis=1, keepdims=True)) / 0.65)
    _p_phys_cls = (_exp_p / _exp_p.sum(axis=1, keepdims=True)).astype(_np.float32)

    _soft_targets = 0.65 * _p_cluster_cls + 0.35 * _p_phys_cls
    _soft_targets /= _soft_targets.sum(axis=1, keepdims=True)
    _np.save("/marimo/Fenomena/pure_vision_soft_targets.npy", _soft_targets)

    _hard_y = _soft_targets.argmax(axis=1)
    _folds = _np.zeros(len(_df_inv), dtype=_np.int64)
    _skf = _SKF(n_splits=5, shuffle=True, random_state=42)
    for _f, (_, _va) in enumerate(_skf.split(_soft_targets, _hard_y)):
        _folds[_va] = _f
    _np.save("/marimo/Fenomena/folds_skf5.npy", _folds)

    print("\n=== DISTRIBUSI PURE-VISION SOFT TARGETS (K=24 CLUSTER + PHYSICAL WEAK SUP) ===")
    for _j, _cname in enumerate(_CLASSES):
        _cnt = int((_hard_y == _j).sum())
        print(f"  {_cname:10s}: {_cnt:4d} ({100.0*_cnt/len(_df_inv):5.2f}%)")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 17. Self-Supervised Contrastive Pretraining (`InfoNCE` pada `ConvNeXt-Base 512x512`)

    - **Tujuan (FAQ No. 5):** Melatih *backbone* `convnext_base.fb_in22k_ft_in1k_384` secara *self-supervised* menggunakan **Symmetric InfoNCE Contrastive Loss** di atas `/dev/shm/fenomena_512_lab_clahe.npy` (`512x512`, LAB CLAHE kanal-L) di mana pasangan `(pre_i, post_i)` dari `Id` yang sama adalah pasangan positif dan `Id` lain dalam *batch* adalah negatif.
    - **Augmentasi:** Hanya transformasi dihedral geometrik identik (`HFlip/VFlip/Rot90`) tanpa distorsi warna/erasing brutal.
    """)
    return


@app.cell
def _():
    # Langkah 4.1: Self-Supervised Contrastive Pretraining (InfoNCE pada ConvNeXt-Base 512x512, Zero-IPC RAM)
    import os as _os, time as _time, random as _random
    import numpy as _np, torch as _torch, torch.nn as _nn, torch.nn.functional as _F
    import timm as _timm

    _CKPT_CONTRASTIVE = "/marimo/Fenomena/ckpt_contrastive_cnxb512.pt"
    if _os.path.exists(_CKPT_CONTRASTIVE):
        print("Contrastive checkpoint already exists:", _CKPT_CONTRASTIVE)
    else:
        _t0 = _time.time()
        _imgs_shm = _np.load("/dev/shm/fenomena_512_lab_clahe.npy", mmap_mode="r")  # (3180, 2, 512, 512, 3)
        _N = len(_imgs_shm)
        _mean = _torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        _std = _torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def _aug_gpu(x):
            if _random.random() < 0.5: x = _torch.flip(x, [3])
            if _random.random() < 0.5: x = _torch.flip(x, [2])
            k = _random.randint(0, 3)
            if k: x = _torch.rot90(x, k, [2, 3])
            return x

        class _ContrastiveCNXB(_nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = _timm.create_model("convnext_base.fb_in22k_ft_in1k_384", pretrained=True, num_classes=0)
                self.proj = _nn.Sequential(
                    _nn.Linear(self.backbone.num_features, 512),
                    _nn.BatchNorm1d(512),
                    _nn.GELU(),
                    _nn.Linear(512, 256),
                )
            def forward(self, x):
                return _F.normalize(self.proj(self.backbone(x)), dim=-1)

        _bs = 32
        _steps_per_ep = _N // _bs
        _epochs = 6
        _model = _ContrastiveCNXB().cuda()
        _opt = _torch.optim.AdamW(_model.parameters(), lr=1.5e-4, weight_decay=1e-2)
        _sched = _torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_epochs * _steps_per_ep)
        _rng = _np.random.default_rng(42)

        for _ep in range(1, _epochs + 1):
            _model.train()
            _perm = _rng.permutation(_N)
            _loss_sum = 0.0
            for _s in range(_steps_per_ep):
                _idx = _perm[_s * _bs : (_s + 1) * _bs]
                _batch = _torch.from_numpy(_np.array(_imgs_shm[_idx], copy=True)).cuda(non_blocking=True)
                _xa = (_batch[:, 0].permute(0, 3, 1, 2).float() / 255.0 - _mean) / _std
                _xb = (_batch[:, 1].permute(0, 3, 1, 2).float() / 255.0 - _mean) / _std
                _xa, _xb = _aug_gpu(_xa), _aug_gpu(_xb)
                _opt.zero_grad(set_to_none=True)
                with _torch.amp.autocast("cuda", dtype=_torch.bfloat16):
                    _za, _zb = _model(_xa), _model(_xb)
                    _logits = (_za @ _zb.T) / 0.10
                    _lbls = _torch.arange(_bs, device="cuda")
                    _loss = 0.5 * (_F.cross_entropy(_logits, _lbls) + _F.cross_entropy(_logits.T, _lbls))
                _loss.backward()
                _opt.step()
                _sched.step()
                _loss_sum += float(_loss.item())
            print(f"  [InfoNCE] Epoch {_ep:02d}/{_epochs} | loss={_loss_sum/_steps_per_ep:.4f} | elapsed={_time.time()-_t0:.1f}s")

        _torch.save(_model.backbone.state_dict(), _CKPT_CONTRASTIVE)
        print(f"Saved contrastive backbone weights -> {_CKPT_CONTRASTIVE} in {_time.time()-_t0:.1f}s")
        del _model, _opt, _sched
        _torch.cuda.empty_cache()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 20. Audit Forensik `v5_pure_vision` — Mengapa Klaster `K=24` Didominasi Latar Statis

    1. **Dominasi Vektor Statis (`0.5*(e_A + e_B)` vs `e_A - e_B`):**
       - Pada `ScaleMAE`, rasio norma `||mean|| / ||diff||` mencapai **`4.756x`** (median `3.631x`), dan pada `I-JEPA` **`1.246x`**. Karena SVD di `K16` menyatukan `mean` dan `diff`, klaster `K=24` terbentuk berdasarkan **bioma/latar statis** (*kota = Earthquake*, *pantai = Tsunami*, *kabut/grayscale = Volcano*), bukan *pola kerusakan*.
    2. **Distribusi Arah Perubahan Searah (`post - pre`, `3.180` pasang):**
       - `dExG (post - pre)`: `mean = -0.0082` (`<0 = 48.93%` vs `>0 = 43.58%`)
       - `dWater (post - pre)`: `mean = +0.0224` (`>0 = 45.66%` vs `<0 = 43.30%`)
       - `dLogLap (post - pre)`: `mean = -0.0474` (`<0 = 53.02%` vs `>0 = 46.98%`)
       - `dBlur (post - pre)`: `mean = +0.000091` (`>0 = 53.02%`)
       - **Anomali Sensor:** `NIR False-Color = 365 (11.48%)`, `Grayscale = 420 (13.21%)`, `Bersih RGB = 2.402 (75.53%)`.
    3. **Solusi Arsitektur (`v6_spatial_physics`):** Ekstraksi *DINOv3 `64x64` + DINOv2-Giant `37x37` Spatial Change Heatmap*, pemisahan 3 rezim sensor (`RGB`, `NIR`, `Grayscale`), serta *Change-Only Over-Clustering ($K=32$)* **tanpa** vektor statis `0.5*(e_A + e_B)`.
    """)
    return


@app.cell
def _():
    # TAHAP 1: Audit Forensik v5_pure_vision (Norm Ratio Mean vs Diff & Arah Perubahan Searah post - pre)
    import os as _os
    import numpy as _np, pandas as _pd
    import cv2 as _cv2
    from concurrent.futures import ThreadPoolExecutor as _TPE

    _df_inv = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet")
    _pv_files = [
        ("DINOv3-L", "emb_DINOv3L_5crop_sym.npy"),
        ("SatMAE++", "emb_SatMAEPP_5crop_sym.npy"),
        ("ScaleMAE", "emb_ScaleMAE_5crop_sym.npy"),
        ("DINOv2-Giant", "emb_DINOv2Giant_5crop_sym.npy"),
        ("I-JEPA", "emb_IJEPA_5crop_sym.npy"),
    ]

    print("=== 1. RASIO NORMA ||0.5*(e_A + e_B)|| vs ||e_A - e_B|| PADA EMBEDDING K16 ===")
    _ratios = []
    for _name, _fn in _pv_files:
        _arr = _np.load(f"/marimo/Fenomena/{_fn}", mmap_mode="r")
        _half = _arr.shape[1] // 2
        _n_mean = _np.linalg.norm(_arr[:, :_half], axis=1)
        _n_diff = _np.linalg.norm(_arr[:, _half:], axis=1)
        _r = _n_mean / (_n_diff + 1e-8)
        _ratios.append((_name, float(_n_mean.mean()), float(_n_diff.mean()), float(_r.mean()), float(_np.median(_r))))
        print(f"  {_name:13s}: ||mean||={_n_mean.mean():.4f} | ||diff||={_n_diff.mean():.4f} | ratio mean/diff={_r.mean():.3f}x (median={_np.median(_r):.3f}x)")

    def _dir_stats(idx):
        im_a = _cv2.resize(_cv2.imread(_df_inv["path_pre"].iloc[idx]), (256, 256))
        im_b = _cv2.resize(_cv2.imread(_df_inv["path_post"].iloc[idx]), (256, 256))
        ra, ga, ba = im_a[:, :, 2].astype(_np.float32), im_a[:, :, 1].astype(_np.float32), im_a[:, :, 0].astype(_np.float32)
        rb, gb, bb = im_b[:, :, 2].astype(_np.float32), im_b[:, :, 1].astype(_np.float32), im_b[:, :, 0].astype(_np.float32)
        va = (ra > 8) | (ga > 8) | (ba > 8)
        vb = (rb > 8) | (gb > 8) | (bb > 8)
        v = va & vb
        if v.sum() < 64: v = _np.ones((256, 256), dtype=bool)
        exg_a = float(((2*ga - ra - ba)[v] / 255.0).mean())
        exg_b = float(((2*gb - rb - bb)[v] / 255.0).mean())
        wat_a = float(((ba[v] > ra[v] + 8) & (ba[v] > ga[v] - 5) & ((0.299*ra+0.587*ga+0.114*ba)[v] < 150)).mean())
        wat_b = float(((bb[v] > rb[v] + 8) & (bb[v] > gb[v] - 5) & ((0.299*rb+0.587*gb+0.114*bb)[v] < 150)).mean())
        ga_gray = _cv2.cvtColor(im_a, _cv2.COLOR_BGR2GRAY).astype(_np.float32)
        gb_gray = _cv2.cvtColor(im_b, _cv2.COLOR_BGR2GRAY).astype(_np.float32)
        lap_a = float(_cv2.Laplacian(ga_gray, _cv2.CV_32F)[v].var())
        lap_b = float(_cv2.Laplacian(gb_gray, _cv2.CV_32F)[v].var())
        blur_a = 1.0 / (lap_a + 1.0)
        blur_b = 1.0 / (lap_b + 1.0)
        is_nir = int((ra[v].mean()-ga[v].mean() > 25) or (rb[v].mean()-gb[v].mean() > 25) or (ra[v].mean()>105 and ba[v].mean()>105 and ga[v].mean()<85) or (rb[v].mean()>105 and bb[v].mean()>105 and gb[v].mean()<85))
        is_gray = int((abs(ra[v].mean()-ga[v].mean())+abs(ga[v].mean()-ba[v].mean()) < 5) or (abs(rb[v].mean()-gb[v].mean())+abs(gb[v].mean()-bb[v].mean()) < 5))
        return exg_b - exg_a, wat_b - wat_a, _np.log1p(lap_b) - _np.log1p(lap_a), blur_b - blur_a, is_nir, is_gray

    with _TPE(max_workers=16) as _ex:
        _ds = _np.array(list(_ex.map(_dir_stats, range(len(_df_inv)))), dtype=_np.float64)

    _d_exg, _d_wat, _d_lap, _d_blur, _nir, _gray = _ds[:, 0], _ds[:, 1], _ds[:, 2], _ds[:, 3], _ds[:, 4], _ds[:, 5]
    print("\n=== 2. DISTRIBUSI ARAH PERUBAHAN SEARAH (post - pre) PADA 3.180 PASANG ===")
    print(f"  dExG (post - pre)     : mean={_d_exg.mean():+.4f} | <0 (veg loss)={100*(_d_exg<0).mean():.2f}% | >0={100*(_d_exg>0).mean():.2f}%")
    print(f"  dWater (post - pre)   : mean={_d_wat.mean():+.4f} | >0 (water gain)={100*(_d_wat>0).mean():.2f}% | <0={100*(_d_wat<0).mean():.2f}%")
    print(f"  dLogLap (post - pre)  : mean={_d_lap.mean():+.4f} | <0 (smooth drop)={100*(_d_lap<0).mean():.2f}% | >0={100*(_d_lap>0).mean():.2f}%")
    print(f"  dBlur (post - pre)    : mean={_d_blur.mean():+.6f} | >0 (blurrier post)={100*(_d_blur>0).mean():.2f}%")
    print(f"  Anomali Sensor        : NIR False-Color={int(_nir.sum())} ({100*_nir.mean():.2f}%) | Grayscale={int(_gray.sum())} ({100*_gray.mean():.2f}%) | Bersih RGB={int(((1-_nir)*(1-_gray)).sum())} ({100*((1-_nir)*(1-_gray)).mean():.2f}%)")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 21. Ekstraksi *DINOv3 Spatial Change Heatmap (`64×64`)* & *DINOv2-Giant (`37×37`)*

    - **`DINOv3-ViT-L/16` (`1024×1024`):** Mengambil `4.096` patch tokens (`64×64`) untuk `pre` dan `post`, menghitung peta panas perubahan kosinus $\Delta(i,j) = 1 - \cos(\mathbf{t}_{	ext{post}}(i,j), \mathbf{t}_{	ext{pre}}(i,j))$ (`dinov3_change_map_64x64.npy`) serta fitur perubahan murni `|t_post - t_pre|`.
    - **`DINOv2-Giant` (`518×518`):** Mengambil `1.369` patch tokens (`37×37`) untuk fusi multi-skala (`dinov2_change_map_37x37.npy`).
    """)
    return


@app.cell
def _():
    # TAHAP 2.1a: Fungsi Ekstraksi DINOv3-L (1024x1024 -> 64x64 Patch Change Heatmap) per Chunk + Chunk 1 (0..795)
    import os as _os, time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    import numpy as _np, pandas as _pd, torch as _torch, torch.nn.functional as _F
    import timm as _timm, cv2 as _cv2

    def extract_dinov3_chunk(cid, start_i, end_i):
        out_m = f"/marimo/Fenomena/_dinov3_map_c{cid}.npy"
        out_d = f"/marimo/Fenomena/_dinov3_diff_c{cid}.npy"
        if _os.path.exists(out_m) and _os.path.exists(out_d):
            print(f"DINOv3 chunk {cid} cached:", _np.load(out_m, mmap_mode="r").shape)
            return
        t0 = _time.time()
        df = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet").iloc[start_i:end_i]
        mdl = _timm.create_model("hf_hub:timm/vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, dynamic_img_size=True).cuda().to(_torch.bfloat16).eval()
        mean = _torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = _torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def _load(idx):
            a = _cv2.cvtColor(_cv2.resize(_cv2.imread(df["path_pre"].iloc[idx]), (1024, 1024)), _cv2.COLOR_BGR2RGB)
            b = _cv2.cvtColor(_cv2.resize(_cv2.imread(df["path_post"].iloc[idx]), (1024, 1024)), _cv2.COLOR_BGR2RGB)
            return a, b

        n, bs = len(df), 16
        maps = _np.zeros((n, 64, 64), dtype=_np.float16)
        diffs = _np.zeros((n, 2048), dtype=_np.float32)
        with _TPE(max_workers=12) as ex, _torch.no_grad(), _torch.amp.autocast("cuda", dtype=_torch.bfloat16):
            for i in range(0, n, bs):
                pairs = list(ex.map(_load, range(i, min(i + bs, n))))
                pa = (_torch.from_numpy(_np.stack([p[0] for p in pairs])).cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                pb = (_torch.from_numpy(_np.stack([p[1] for p in pairs])).cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                ta = _F.normalize(mdl.forward_features(pa.to(_torch.bfloat16))[:, 5:, :].float(), dim=-1)
                tb = _F.normalize(mdl.forward_features(pb.to(_torch.bfloat16))[:, 5:, :].float(), dim=-1)
                cos = (ta * tb).sum(dim=-1).view(-1, 64, 64)
                abs_d = _torch.abs(tb - ta)
                maps[i:i+len(pairs)] = (1.0 - cos).clamp(0.0, 2.0).cpu().numpy().astype(_np.float16)
                diffs[i:i+len(pairs)] = _torch.cat([abs_d.mean(dim=1), abs_d.max(dim=1).values], dim=-1).cpu().numpy().astype(_np.float32)
        _np.save(out_m, maps); _np.save(out_d, diffs)
        del mdl; _torch.cuda.empty_cache()
        print(f"Saved DINOv3 chunk {cid} [{start_i}:{end_i}] {maps.shape} in {_time.time()-t0:.1f}s")

    extract_dinov3_chunk(0, 0, 795)

    return (extract_dinov3_chunk,)


@app.cell
def _(extract_dinov3_chunk):
    extract_dinov3_chunk(1, 795, 1590)
    return


@app.cell
def _(extract_dinov3_chunk):
    extract_dinov3_chunk(2, 1590, 2385)
    return


@app.cell
def _(extract_dinov3_chunk):
    extract_dinov3_chunk(3, 2385, 3180)
    import numpy as _np, os as _os
    _maps = _np.concatenate([_np.load(f"/marimo/Fenomena/_dinov3_map_c{i}.npy") for i in range(4)], axis=0)
    _diffs = _np.concatenate([_np.load(f"/marimo/Fenomena/_dinov3_diff_c{i}.npy") for i in range(4)], axis=0)
    _np.save("/marimo/Fenomena/dinov3_change_map_64x64.npy", _maps)
    _np.save("/marimo/Fenomena/emb_dinov3_1024_diff_only.npy", _diffs)
    for i in range(4):
        for p in [f"/marimo/Fenomena/_dinov3_map_c{i}.npy", f"/marimo/Fenomena/_dinov3_diff_c{i}.npy"]:
            if _os.path.exists(p): _os.remove(p)
    print("ASSEMBLED DINOv3:", _maps.shape, _diffs.shape, "NaN:", _np.isnan(_maps).sum(), _np.isnan(_diffs).sum())

    return


@app.cell
def _():
    # TAHAP 2.1b: Ekstraksi DINOv2-Giant (518x518 -> 37x37 Patch Change Heatmap) — Chunk 0 (0..1590)
    import os as _os, time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    import numpy as _np, pandas as _pd, torch as _torch, torch.nn.functional as _F
    import cv2 as _cv2
    from transformers import AutoModel as _AutoModel

    def extract_dinov2_chunk(cid, start_i, end_i):
        out_m = f"/marimo/Fenomena/_dinov2_map_c{cid}.npy"
        out_d = f"/marimo/Fenomena/_dinov2_diff_c{cid}.npy"
        if _os.path.exists(out_m) and _os.path.exists(out_d):
            print(f"DINOv2 chunk {cid} cached:", _np.load(out_m, mmap_mode="r").shape)
            return
        t0 = _time.time()
        df = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet").iloc[start_i:end_i]
        mdl = _AutoModel.from_pretrained("facebook/dinov2-giant", dtype=_torch.bfloat16).cuda().eval()
        mean = _torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = _torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        def _load(idx):
            a = _cv2.cvtColor(_cv2.resize(_cv2.imread(df["path_pre"].iloc[idx]), (518, 518)), _cv2.COLOR_BGR2RGB)
            b = _cv2.cvtColor(_cv2.resize(_cv2.imread(df["path_post"].iloc[idx]), (518, 518)), _cv2.COLOR_BGR2RGB)
            return a, b

        n, bs = len(df), 24
        maps = _np.zeros((n, 37, 37), dtype=_np.float16)
        diffs = _np.zeros((n, 3072), dtype=_np.float32)
        with _TPE(max_workers=12) as ex, _torch.no_grad(), _torch.amp.autocast("cuda", dtype=_torch.bfloat16):
            for i in range(0, n, bs):
                pairs = list(ex.map(_load, range(i, min(i + bs, n))))
                pa = (_torch.from_numpy(_np.stack([p[0] for p in pairs])).cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                pb = (_torch.from_numpy(_np.stack([p[1] for p in pairs])).cuda(non_blocking=True).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                ta = _F.normalize(mdl(pixel_values=pa.to(_torch.bfloat16)).last_hidden_state[:, 1:, :].float(), dim=-1)
                tb = _F.normalize(mdl(pixel_values=pb.to(_torch.bfloat16)).last_hidden_state[:, 1:, :].float(), dim=-1)
                cos = (ta * tb).sum(dim=-1).view(-1, 37, 37)
                abs_d = _torch.abs(tb - ta)
                maps[i:i+len(pairs)] = (1.0 - cos).clamp(0.0, 2.0).cpu().numpy().astype(_np.float16)
                diffs[i:i+len(pairs)] = _torch.cat([abs_d.mean(dim=1), abs_d.max(dim=1).values], dim=-1).cpu().numpy().astype(_np.float32)
        _np.save(out_m, maps); _np.save(out_d, diffs)
        del mdl; _torch.cuda.empty_cache()
        print(f"Saved DINOv2 chunk {cid} [{start_i}:{end_i}] {maps.shape} in {_time.time()-t0:.1f}s")

    extract_dinov2_chunk(0, 0, 1590)

    return (extract_dinov2_chunk,)


@app.cell
def _(extract_dinov2_chunk):
    extract_dinov2_chunk(1, 1590, 3180)
    import numpy as _np, os as _os
    _m2 = _np.concatenate([_np.load(f"/marimo/Fenomena/_dinov2_map_c{_i}.npy") for _i in range(2)], axis=0)
    _d2 = _np.concatenate([_np.load(f"/marimo/Fenomena/_dinov2_diff_c{_i}.npy") for _i in range(2)], axis=0)
    _np.save("/marimo/Fenomena/dinov2_change_map_37x37.npy", _m2)
    _np.save("/marimo/Fenomena/emb_dinov2_518_diff_only.npy", _d2)
    for _i in range(2):
        for _p in [f"/marimo/Fenomena/_dinov2_map_c{_i}.npy", f"/marimo/Fenomena/_dinov2_diff_c{_i}.npy"]:
            if _os.path.exists(_p): _os.remove(_p)
    print("ASSEMBLED DINOv2:", _m2.shape, _d2.shape, "NaN:", _np.isnan(_m2).sum(), _np.isnan(_d2).sum())

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 22. Fitur Geometri Kerusakan 2D, Fisika Searah 3 Rezim Sensor & *Change-Only Clustering ($K=32$)*

    - **Geometri Kerusakan 2D (`64×64` Fused `DINOv3 + DINOv2` Heatmap):** *Blast compactness* (`max_blob_ratio`, `gini_spatial`, `kurtosis_spatial`), *coastal/water-boundary adjacency*, *urban structure concentration*, dan *scene-wide diffuse spread*.
    - **Fisika Searah (`post - pre`) dengan Pemisahan 3 Rezim Sensor (`RGB` vs `False-Color NIR` vs `Grayscale`):** Mencegah citra *False-Color NIR* masuk ke `Fire` dan mencegah kabut statis/*Grayscale* masuk ke `Volcano`.
    - **Change-Only Over-Clustering ($K=32$, `SEED=42`):** Klastering **hanya** pada fitur perubahan (`|t_post - t_pre|`, geometri 2D, dan fisika searah) **tanpa** vektor statis `0.5*(e_A+e_B)` agar klaster terbentuk murni berdasarkan *pola kerusakan*.
    """)
    return


@app.cell
def _():
    # TAHAP 2.2, 2.3 & 3.1: 2D Spatial Damage Geometry + Directional Physics (3 Sensor Regimes) + Change-Only K=32 + Audit Grid
    import os as _os, time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    import numpy as _np, pandas as _pd, cv2 as _cv2
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    from sklearn.decomposition import TruncatedSVD as _TSVD
    from sklearn.preprocessing import StandardScaler as _SS, normalize as _l2norm
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.model_selection import StratifiedKFold as _SKF

    _t0 = _time.time()
    _CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
    _df = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet")
    _d3_maps = _np.load("/marimo/Fenomena/dinov3_change_map_64x64.npy").astype(_np.float32)
    _d2_maps = _np.load("/marimo/Fenomena/dinov2_change_map_37x37.npy").astype(_np.float32)

    def _extract_spatial_and_dir_phys(idx):
        im_a = _cv2.resize(_cv2.imread(_df["path_pre"].iloc[idx]), (256, 256), interpolation=_cv2.INTER_AREA)
        im_b = _cv2.resize(_cv2.imread(_df["path_post"].iloc[idx]), (256, 256), interpolation=_cv2.INTER_AREA)
        rgb_a = im_a[:, :, ::-1].astype(_np.float32)
        rgb_b = im_b[:, :, ::-1].astype(_np.float32)
        ra, ga, ba = rgb_a[:, :, 0], rgb_a[:, :, 1], rgb_a[:, :, 2]
        rb, gb, bb = rgb_b[:, :, 0], rgb_b[:, :, 1], rgb_b[:, :, 2]

        va = (ra > 8) | (ga > 8) | (ba > 8)
        vb = (rb > 8) | (gb > 8) | (bb > 8)
        v = va & vb
        if v.sum() < 64: v = _np.ones((256, 256), dtype=bool)
        v64 = _cv2.resize(v.astype(_np.float32), (64, 64), interpolation=_cv2.INTER_AREA) > 0.5
        if v64.sum() < 16: v64 = _np.ones((64, 64), dtype=bool)

        d2_64 = _cv2.resize(_d2_maps[idx], (64, 64), interpolation=_cv2.INTER_LINEAR)
        delta = (0.65 * _d3_maps[idx] + 0.35 * d2_64) * v64
        d_vals = delta[v64]
        d_mean = float(d_vals.mean())
        d_max = float(d_vals.max())
        d_std = float(d_vals.std())

        py, px = _np.unravel_index(_np.argmax(delta), (64, 64))
        yy, xx = _np.ogrid[:64, :64]
        dist2 = (yy - py) ** 2 + (xx - px) ** 2
        blob_mask = (dist2 <= 64) & v64
        max_blob_ratio = float(delta[blob_mask].sum() / (d_vals.sum() + 1e-6))

        s_vals = _np.sort(d_vals)
        n_v = len(s_vals)
        gini_spatial = float((2.0 * _np.sum(_np.arange(1, n_v + 1) * s_vals) / (n_v * s_vals.sum() + 1e-6)) - (n_v + 1.0) / n_v)
        z_d = (d_vals - d_mean) / (d_std + 1e-6)
        kurtosis_spatial = float(_np.mean(z_d ** 4) - 3.0)

        thr_hi = d_mean + 0.5 * d_std
        diffuse_spread = float((d_vals > max(thr_hi, 0.25)).mean())
        p_sp = d_vals / (d_vals.sum() + 1e-6)
        spatial_entropy = float(-_np.sum(p_sp * _np.log(p_sp + 1e-8)) / _np.log(n_v + 1e-6))
        q_means = _np.array([delta[:32, :32].mean(), delta[:32, 32:].mean(), delta[32:, :32].mean(), delta[32:, 32:].mean()])
        pq = q_means / (q_means.sum() + 1e-6)
        quad_entropy = float(-_np.sum(pq * _np.log(pq + 1e-8)) / _np.log(4.0))

        mr_a, mg_a, mb_a = float(ra[v].mean()), float(ga[v].mean()), float(ba[v].mean())
        mr_b, mg_b, mb_b = float(rb[v].mean()), float(gb[v].mean()), float(bb[v].mean())
        is_nir = float((mr_a - mg_a > 25) or (mr_b - mg_b > 25) or (mr_a > 105 and mb_a > 105 and mg_a < 85) or (mr_b > 105 and mb_b > 105 and mg_b < 85))
        is_gray = float((abs(mr_a - mg_a) + abs(mg_a - mb_a) < 5) or (abs(mr_b - mg_b) + abs(mg_b - mb_b) < 5))

        hsv_a = _cv2.cvtColor(im_a, _cv2.COLOR_BGR2HSV).astype(_np.float32)
        hsv_b = _cv2.cvtColor(im_b, _cv2.COLOR_BGR2HSV).astype(_np.float32)
        sat_a, sat_b = hsv_a[:, :, 1], hsv_b[:, :, 1]
        lum_a = 0.299 * ra + 0.587 * ga + 0.114 * ba
        lum_b = 0.299 * rb + 0.587 * gb + 0.114 * bb

        gray_a = _cv2.cvtColor(im_a, _cv2.COLOR_BGR2GRAY).astype(_np.float32)
        gray_b = _cv2.cvtColor(im_b, _cv2.COLOR_BGR2GRAY).astype(_np.float32)
        gxa = _cv2.Sobel(gray_a, _cv2.CV_32F, 1, 0, ksize=3)
        gya = _cv2.Sobel(gray_a, _cv2.CV_32F, 0, 1, ksize=3)
        edge_a = _np.sqrt(gxa * gxa + gya * gya)
        edge_a_64 = _cv2.resize(edge_a, (64, 64), interpolation=_cv2.INTER_AREA)
        urban_edge_pre = float((edge_a[v] > 45.0).mean())
        urban_delta_corr = float(_np.corrcoef(delta[v64], edge_a_64[v64])[0, 1]) if edge_a_64[v64].std() > 1e-5 else 0.0
        if _np.isnan(urban_delta_corr): urban_delta_corr = 0.0

        if is_gray > 0.5:
            veg_loss = 0.0
            dark_char_gain = float((lum_b[v] < 55).mean() - (lum_a[v] < 55).mean()) * 0.3
        elif is_nir > 0.5:
            veg_loss = float(max(0.0, (mr_a - mr_b) / 255.0))
            char_a = ((lum_a < 70) & (sat_a < 55))
            char_b = ((lum_b < 70) & (sat_b < 55))
            dark_char_gain = float(char_b[v].mean() - char_a[v].mean())
        else:
            exg_a = (2.0 * ga - ra - ba) / 255.0
            exg_b = (2.0 * gb - rb - bb) / 255.0
            veg_loss = float(max(0.0, exg_a[v].mean() - exg_b[v].mean()))
            char_a = ((lum_a < 68) & (sat_a < 50))
            char_b = ((lum_b < 68) & (sat_b < 50))
            dark_char_gain = float(char_b[v].mean() - char_a[v].mean())

        if is_gray > 0.5:
            water_gain = 0.0
            coastal_adj = 0.0
            water_post_frac = 0.0
        else:
            w_a = ((ba > ra + 8) & (ba > ga - 5) & (lum_a < 155)) | ((ga > ra + 10) & (ba > ra + 8) & (lum_a < 125))
            w_b = ((bb > rb + 8) & (bb > gb - 5) & (lum_b < 155)) | ((gb > rb + 10) & (bb > rb + 8) & (lum_b < 125))
            water_post_frac = float(w_b[v].mean())
            water_gain = float(w_b[v].mean() - w_a[v].mean())
            wb_64 = _cv2.resize(w_b.astype(_np.float32), (64, 64), interpolation=_cv2.INTER_AREA)
            wb_dil = _cv2.dilate((wb_64 > 0.25).astype(_np.uint8), _np.ones((5, 5), _np.uint8))
            coastal_adj = float((delta * wb_dil)[v64].mean() / (d_mean + 1e-5)) * float(water_post_frac > 0.03)

        lap_a = float(_np.log1p(_cv2.Laplacian(gray_a, _cv2.CV_32F)[v].var()))
        lap_b = float(_np.log1p(_cv2.Laplacian(gray_b, _cv2.CV_32F)[v].var()))
        lap_drop = float(max(0.0, lap_a - lap_b))
        if is_gray > 0.5:
            smooth_blanket_gain = float(max(0.0, ((lum_b > 90) & (lum_b < 220))[v].mean() - ((lum_a > 90) & (lum_a < 220))[v].mean()))
        else:
            ash_a = (sat_a < 32) & (lum_a > 80) & (lum_a < 220)
            ash_b = (sat_b < 32) & (lum_b > 80) & (lum_b < 220)
            smooth_blanket_gain = float(max(0.0, ash_b[v].mean() - ash_a[v].mean()))
        real_change_gate = float(d_mean > 0.18)

        def _hf(g):
            m = _np.abs(_np.fft.fftshift(_np.fft.fft2(g)))
            yy2, xx2 = _np.ogrid[:256, :256]
            return float(m[(yy2 - 128)**2 + (xx2 - 128)**2 > 1024].sum() / (m.sum() + 1e-6))
        hf_gain = _hf(gray_b) - _hf(gray_a)

        return _np.array([
            d_mean, d_max, d_std, max_blob_ratio, gini_spatial, kurtosis_spatial,
            diffuse_spread, spatial_entropy, quad_entropy,
            urban_edge_pre, urban_delta_corr, hf_gain,
            veg_loss, dark_char_gain,
            water_post_frac, water_gain, coastal_adj,
            lap_drop * real_change_gate, smooth_blanket_gain * real_change_gate,
            is_nir, is_gray,
        ], dtype=_np.float32)

    with _TPE(max_workers=16) as _ex:
        _sp_mat = _np.stack(list(_ex.map(_extract_spatial_and_dir_phys, range(len(_df)))), axis=0)

    _np.save("/marimo/Fenomena/spatial_dir_physics.npy", _sp_mat)
    print(f"spatial_dir_physics.npy shape={_sp_mat.shape} in {_time.time()-_t0:.1f}s | NaN={_np.isnan(_sp_mat).sum()}")

    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    _s_earthquake = 0.40 * _z(_sp_mat[:, 9]) + 0.35 * _z(_sp_mat[:, 10]) + 0.25 * _z(_sp_mat[:, 11]) + 0.15 * _z(_sp_mat[:, 6]) - 0.20 * _z(_sp_mat[:, 16])
    _s_fire = 0.50 * _z(_sp_mat[:, 12]) + 0.40 * _z(_sp_mat[:, 13]) - 0.25 * _z(_sp_mat[:, 16]) - 0.15 * _z(_sp_mat[:, 20])
    _s_hurricane = 0.45 * _z(_sp_mat[:, 6]) + 0.35 * _z(_sp_mat[:, 7]) + 0.25 * _z(_sp_mat[:, 8]) - 0.35 * _z(_sp_mat[:, 3]) - 0.25 * _z(_sp_mat[:, 4])
    _s_tsunami = 0.45 * _z(_sp_mat[:, 15]) + 0.40 * _z(_sp_mat[:, 16]) + 0.25 * _z(_sp_mat[:, 14]) - 0.20 * _z(_sp_mat[:, 20])
    _s_volcano = 0.50 * _z(_sp_mat[:, 17]) + 0.45 * _z(_sp_mat[:, 18]) - 0.20 * _z(_sp_mat[:, 10])
    _s_explosion = 0.45 * _z(_sp_mat[:, 3]) + 0.35 * _z(_sp_mat[:, 4]) + 0.30 * _z(_sp_mat[:, 5]) - 0.30 * _z(_sp_mat[:, 6])

    _dir_scores = _np.stack([_s_earthquake, _s_fire, _s_hurricane, _s_tsunami, _s_volcano, _s_explosion], axis=1).astype(_np.float32)

    _chg_blocks = [
        _l2norm(_TSVD(n_components=128, random_state=42).fit_transform(_l2norm(_np.load("/marimo/Fenomena/emb_dinov3_1024_diff_only.npy")))),
        _l2norm(_TSVD(n_components=128, random_state=42).fit_transform(_l2norm(_np.load("/marimo/Fenomena/emb_dinov2_518_diff_only.npy")))),
    ]
    for _fn in ["emb_DINOv3L_5crop_sym.npy", "emb_SatMAEPP_5crop_sym.npy", "emb_ScaleMAE_5crop_sym.npy", "emb_DINOv2Giant_5crop_sym.npy", "emb_IJEPA_5crop_sym.npy"]:
        _arr = _np.load(f"/marimo/Fenomena/{_fn}", mmap_mode="r")
        _half = _arr.shape[1] // 2
        _diff_only = _l2norm(_np.array(_arr[:, _half:], dtype=_np.float32))
        _chg_blocks.append(_l2norm(_TSVD(n_components=64, random_state=42).fit_transform(_diff_only)))

    _sp_scaled = _l2norm(_SS().fit_transform(_sp_mat[:, :19]))
    _Z_change = _l2norm(_np.hstack(_chg_blocks + [0.85 * _sp_scaled]))

    _km = _KMeans(n_clusters=32, n_init=20, max_iter=300, random_state=42).fit(_Z_change)
    _cids = _km.labels_
    _centroids = _l2norm(_km.cluster_centers_)
    _cos_sim = _Z_change @ _centroids.T
    _c_sizes = _np.bincount(_cids, minlength=32)

    _c_prof = _np.zeros((32, 6), dtype=_np.float32)
    for _c in range(32):
        _c_prof[_c] = _dir_scores[_cids == _c].mean(axis=0)
    _prof_z = (_c_prof - _c_prof.mean(axis=0)) / (_c_prof.std(axis=0) + 1e-6)

    CLUSTER_TO_CLASS_V6 = {int(c): _CLASSES[int(_np.argmax(_prof_z[c]))] for c in range(32)}

    print("\n=== PROFIL 32 KLASTER CHANGE-ONLY & PEMETAAN CLUSTER_TO_CLASS_V6 ===")
    for _c in range(32):
        _nir_p = 100.0 * _sp_mat[_cids == _c, 19].mean()
        _gry_p = 100.0 * _sp_mat[_cids == _c, 20].mean()
        _vals = " ".join(f"{_CLASSES[j][:4]}={_prof_z[_c, j]:+5.2f}" for j in range(6))
        print(f"  C{_c:02d} (n={_c_sizes[_c]:3d}, NIR={_nir_p:4.1f}%, Gray={_gry_p:4.1f}%) -> {CLUSTER_TO_CLASS_V6[_c]:10s} | {_vals}")

    _exp_c = _np.exp((_cos_sim - _cos_sim.max(axis=1, keepdims=True)) / 0.08)
    _p_c = _exp_c / _exp_c.sum(axis=1, keepdims=True)
    _p_clust_cls = _np.zeros((len(_df), 6), dtype=_np.float32)
    for _c in range(32):
        _p_clust_cls[:, _CLASSES.index(CLUSTER_TO_CLASS_V6[_c])] += _p_c[:, _c]

    _exp_p = _np.exp((_dir_scores - _dir_scores.max(axis=1, keepdims=True)) / 0.65)
    _p_phys_cls = (_exp_p / _exp_p.sum(axis=1, keepdims=True)).astype(_np.float32)

    _soft_v6 = 0.60 * _p_clust_cls + 0.40 * _p_phys_cls
    _soft_v6 /= _soft_v6.sum(axis=1, keepdims=True)
    _np.save("/marimo/Fenomena/pure_vision_v6_soft_targets.npy", _soft_v6)

    _hard_v6 = _soft_v6.argmax(axis=1)
    _folds = _np.zeros(len(_df), dtype=_np.int64)
    _skf = _SKF(n_splits=5, shuffle=True, random_state=42)
    for _f, (_, _va) in enumerate(_skf.split(_soft_v6, _hard_v6)):
        _folds[_va] = _f
    _np.save("/marimo/Fenomena/folds_skf5.npy", _folds)

    print("\n=== DISTRIBUSI TARGET LUNAK v6 (CHANGE-ONLY K=32 + DIRECTIONAL PHYSICS) ===")
    for _j, _cn in enumerate(_CLASSES):
        _cnt = int((_hard_v6 == _j).sum())
        print(f"  {_cn:10s}: {_cnt:4d} ({100.0*_cnt/len(_df):5.2f}%)")

    _rng = _np.random.default_rng(42)
    _tw, _th = 180, 180
    _pad_x, _pad_y = 10, 34
    _cols_w = 4 * (3 * _tw + _pad_x) + 140
    _rows_h = 6 * (_th + _pad_y) + 20
    _canvas = _Image.new("RGB", (_cols_w, _rows_h), (245, 245, 245))
    _draw = _ImageDraw.Draw(_canvas)

    for _j, _cn in enumerate(_CLASSES):
        _y0 = 10 + _j * (_th + _pad_y)
        _draw.text((8, _y0 + 20), f"{_cn}\n(n={int((_hard_v6==_j).sum())})", fill=(10, 10, 10))
        _cands = _np.where(_hard_v6 == _j)[0]
        _picks = _rng.choice(_cands, size=min(4, len(_cands)), replace=False)
        for _k, _idx in enumerate(_picks):
            _x0 = 130 + _k * (3 * _tw + _pad_x)
            _ima = _Image.open(_df["path_pre"].iloc[_idx]).convert("RGB").resize((_tw, _th))
            _imb = _Image.open(_df["path_post"].iloc[_idx]).convert("RGB").resize((_tw, _th))
            _hm = _d3_maps[_idx]
            _hm_u8 = _np.clip((_hm - _hm.min()) / (_hm.max() - _hm.min() + 1e-6) * 255.0, 0, 255).astype(_np.uint8)
            _hm_rgb = _cv2.applyColorMap(_cv2.resize(_hm_u8, (_tw, _th), interpolation=_cv2.INTER_CUBIC), _cv2.COLORMAP_JET)[:, :, ::-1]
            _imh = _Image.fromarray(_hm_rgb)
            _canvas.paste(_ima, (_x0, _y0 + 18))
            _canvas.paste(_imb, (_x0 + _tw, _y0 + 18))
            _canvas.paste(_imh, (_x0 + 2 * _tw, _y0 + 18))
            _txt = f"{_df['id'].iloc[_idx]} | d={_sp_mat[_idx,0]:.2f} blob={_sp_mat[_idx,3]:.2f} wAdj={_sp_mat[_idx,16]:.2f} urb={_sp_mat[_idx,10]:.2f}"
            _draw.text((_x0, _y0 + 2), _txt, fill=(30, 30, 30))

    _canvas.save("/marimo/Fenomena/audit_spatial_6classes.png")
    print("Saved /marimo/Fenomena/audit_spatial_6classes.png")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 26. (`M20_claude_view_sd`) Tes Diagnostik `View S` (Struktur *Event*) & Over-Clustering `K = 72` + *Cluster Cards* (`View D`)

    - **Tahap 1 — Diagnostik `View S`:** Menguji apakah dataset memiliki struktur *event* rapat dengan membangun graf kNN (`k=15`, kosinus) secara terpisah pada `View S` (`0.5*(e_A + e_B)`) dari `DINOv3-L`, `SatMAE++`, `I-JEPA`, `DINOv2-Giant`, dan `ScaleMAE`, lalu mengukur distribusi ukuran komunitas, **ARI** antar-backbone & antar-seed (`42` vs `1337`), serta kemurnian rezim sensor.
    - **Tahap 2 — Fusi Graf kNN Multi-Backbone (`K = 72`) & *Cluster Cards* (`View D`):** Memisahkan 3 rezim sensor (`RGB`, `False-Color NIR`, `Grayscale`), melakukan *rank-averaging* kemiripan kosinus 5 backbone `View S`, membentuk **`K = 72`** sub-klaster (`C00..C71`), meringkas indikator `View D` dalam **peringkat persentil `[0, 1]`** ke `cluster_cards_k72.csv`, dan membuat montase visual `audit_cluster_cards_k72.png`.
    """)
    return


@app.cell
def _():
    # K20_claude_view_sd: Tahap 1 (View S Event Structure Diagnostic) + Tahap 2 (Balanced K=72 kNN-Fused Clusters & Cluster Cards)
    import os as _os, time as _time
    import numpy as _np, pandas as _pd, cv2 as _cv2
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    from scipy.stats import rankdata as _rankdata
    from sklearn.preprocessing import normalize as _l2norm, StandardScaler as _SS
    from sklearn.neighbors import kneighbors_graph as _knn_graph
    from sklearn.cluster import SpectralClustering as _SpecClust
    from sklearn.metrics import adjusted_rand_score as _ari

    _t0 = _time.time()
    _df = _pd.read_parquet("/marimo/Fenomena/df_inventory.parquet")
    _sp = _np.load("/marimo/Fenomena/spatial_dir_physics.npy").astype(_np.float32)
    _phys_sym = _np.load("/marimo/Fenomena/physical_features_sym.npy").astype(_np.float32)
    _d3_maps = _np.load("/marimo/Fenomena/dinov3_change_map_64x64.npy").astype(_np.float32)
    _N = len(_df)

    _bb_files = [
        ("DINOv3-L", "emb_DINOv3L_5crop_sym.npy"),
        ("SatMAE++", "emb_SatMAEPP_5crop_sym.npy"),
        ("ScaleMAE", "emb_ScaleMAE_5crop_sym.npy"),
        ("DINOv2-G", "emb_DINOv2Giant_5crop_sym.npy"),
        ("I-JEPA", "emb_IJEPA_5crop_sym.npy"),
    ]
    _view_s = {}
    for _name, _fn in _bb_files:
        _arr = _np.load(f"/marimo/Fenomena/{_fn}", mmap_mode="r")
        _h = _arr.shape[1] // 2
        _view_s[_name] = _l2norm(_np.array(_arr[:, :_h], dtype=_np.float32))

    _is_nir = _sp[:, 19] > 0.5
    _is_gray = (_sp[:, 20] > 0.5) & (~_is_nir)
    _regime_lbl = _np.where(_is_nir, 1, _np.where(_is_gray, 2, 0))
    _regime_names = {0: "RGB", 1: "FalseColor_NIR", 2: "Grayscale"}

    def _cluster_knn(X, seed=42, n_target=50):
        A = _knn_graph(X, n_neighbors=15, mode="connectivity", metric="cosine", include_self=False)
        A = 0.5 * (A + A.T)
        return _SpecClust(n_clusters=n_target, affinity="precomputed", random_state=seed, assign_labels="kmeans", n_init=10).fit_predict(A.toarray())

    _diag_bb = ["DINOv3-L", "SatMAE++", "I-JEPA", "DINOv2-G"]
    _comm_labels = {}
    print("=== LAPORAN DIAGNOSTIK TAHAP 1: KOMUNITAS GRAF kNN (k=15, SEED=42) ===")
    for _bn in _diag_bb:
        _lbls = _cluster_knn(_view_s[_bn], seed=42, n_target=50)
        _comm_labels[_bn] = _lbls
        _u, _cnts = _np.unique(_lbls, return_counts=True)
        _purities = [_np.bincount(_regime_lbl[_lbls == c]).max() / (_lbls == c).sum() for c in _u]
        print(f"  {_bn:9s}: n_comm={len(_u):2d} | size[min={_cnts.min()}, p25={int(_np.percentile(_cnts,25))}, med={int(_np.median(_cnts))}, p75={int(_np.percentile(_cnts,75))}, max={_cnts.max()}] | mean_sensor_purity={100*_np.mean(_purities):.1f}%")

    print("\n=== ADJUSTED RAND INDEX (ARI) ANTAR-BACKBONE (SEED=42) ===")
    for _b1 in _diag_bb:
        print("  " + f"{_b1:9s}: " + " | ".join(f"{_b2}={_ari(_comm_labels[_b1], _comm_labels[_b2]):.3f}" for _b2 in _diag_bb))

    _lbls_d3_1337 = _cluster_knn(_view_s["DINOv3-L"], seed=1337, n_target=50)
    print(f"\n=== STABILITAS ANTAR-SEED (DINOv3-L SEED=42 vs SEED=1337) ===")
    print(f"  ARI(SEED=42, SEED=1337) = {_ari(_comm_labels['DINOv3-L'], _lbls_d3_1337):.4f}")

    # TAHAP 2: Multi-Backbone Rank-Averaged kNN Graph Fusion per Sensor Regime -> Balanced K=72
    _regime_k = {0: 54, 1: 9, 2: 9}
    _final_cids = _np.full(_N, -1, dtype=_np.int32)
    _cluster_regime = {}
    _cluster_centroids_view_s = {}
    _c_offset = 0
    _X_vs_concat = _l2norm(_np.hstack([_view_s[b] for b in ["DINOv3-L", "SatMAE++", "ScaleMAE", "DINOv2-G", "I-JEPA"]]))

    for _rid in [0, 1, 2]:
        _idx_r = _np.where(_regime_lbl == _rid)[0]
        _n_r = len(_idx_r)
        _k_r = _regime_k[_rid]
        _sim_sum = _np.zeros((_n_r, _n_r), dtype=_np.float64)
        for _bn in ["DINOv3-L", "SatMAE++", "ScaleMAE", "DINOv2-G", "I-JEPA"]:
            _Xr = _view_s[_bn][_idx_r]
            _S = (_Xr @ _Xr.T).astype(_np.float64)
            _ranks = _rankdata(_S.ravel(), method="average").reshape(_n_r, _n_r) / (_n_r * _n_r)
            _sim_sum += _ranks
        _sim_avg = _sim_sum / 5.0
        # Build symmetric k=15 kNN affinity from rank-averaged similarity
        _k_nn = min(15, _n_r - 1)
        _A = _np.zeros((_n_r, _n_r), dtype=_np.float64)
        for _row in range(_n_r):
            _nbrs = _np.argsort(-_sim_avg[_row])[1:_k_nn + 1]
            _A[_row, _nbrs] = _sim_avg[_row, _nbrs]
        _A = 0.5 * (_A + _A.T)
        _sub_lbls = _SpecClust(n_clusters=_k_r, affinity="precomputed", random_state=42, assign_labels="kmeans", n_init=15).fit_predict(_A)
        for _sc in range(_k_r):
            _gid = _c_offset + _sc
            _members = _idx_r[_sub_lbls == _sc]
            _final_cids[_members] = _gid
            _cluster_regime[_gid] = _regime_names[_rid]
            _cluster_centroids_view_s[_gid] = _l2norm(_X_vs_concat[_members].mean(axis=0, keepdims=True))[0]
        _c_offset += _k_r

    assert (_final_cids >= 0).all() and len(_np.unique(_final_cids)) == 72

    def _pct(x):
        return (_rankdata(x, method="average") - 1.0) / (len(x) - 1.0 + 1e-8)

    _pct_d_mag, _pct_gini, _pct_blob = _pct(_sp[:, 0]), _pct(_sp[:, 4]), _pct(_sp[:, 3])
    _pct_coastal, _pct_urban = _pct(_sp[:, 16]), _pct(_sp[:, 10])
    _pct_veg_char, _pct_tex_drop = _pct(_sp[:, 12] + _sp[:, 13]), _pct(_sp[:, 17])

    _cards = []
    for _cid in range(72):
        _m = _final_cids == _cid
        _cards.append({
            "cluster_id": f"C{_cid:02d}",
            "n_samples": int(_m.sum()),
            "sensor_regime": _cluster_regime[_cid],
            "pct_change_mag": round(float(_np.median(_pct_d_mag[_m])), 3),
            "pct_gini": round(float(_np.median(_pct_gini[_m])), 3),
            "pct_max_blob": round(float(_np.median(_pct_blob[_m])), 3),
            "pct_coastal_adj": round(float(_np.median(_pct_coastal[_m])), 3),
            "pct_urban_corr": round(float(_np.median(_pct_urban[_m])), 3),
            "pct_veg_char": round(float(_np.median(_pct_veg_char[_m])), 3),
            "pct_tex_drop": round(float(_np.median(_pct_tex_drop[_m])), 3),
        })

    _df_cards = _pd.DataFrame(_cards)
    _df_cards.to_csv("/marimo/Fenomena/cluster_cards_k72.csv", index=False)
    _np.save("/marimo/Fenomena/cluster_ids_k72_views.npy", _final_cids)
    print(f"\n[Tahap 2] Saved cluster_cards_k72.csv (72 clusters) | size[min={_df_cards['n_samples'].min()}, p25={int(_df_cards['n_samples'].quantile(0.25))}, med={int(_df_cards['n_samples'].median())}, p75={int(_df_cards['n_samples'].quantile(0.75))}, max={_df_cards['n_samples'].max()}]")

    _tw, _th = 110, 110
    _pad_x, _pad_y = 6, 26
    _page_paths = []
    for _pg in range(3):
        _c_start, _c_end = _pg * 24, (_pg + 1) * 24
        _cw = 6 * (3 * _tw + _pad_x) + 210
        _ch = 24 * (_th + _pad_y) + 20
        _canv = _Image.new("RGB", (_cw, _ch), (245, 245, 245))
        _drw = _ImageDraw.Draw(_canv)
        for _row_i, _cid in enumerate(range(_c_start, _c_end)):
            _y0 = 10 + _row_i * (_th + _pad_y)
            _rc = _cards[_cid]
            _drw.text((6, _y0 + 10), f"{_rc['cluster_id']} (n={_rc['n_samples']})\n{_rc['sensor_regime']}\nchg={_rc['pct_change_mag']:.2f} urb={_rc['pct_urban_corr']:.2f}\ncst={_rc['pct_coastal_adj']:.2f} veg={_rc['pct_veg_char']:.2f}", fill=(10, 10, 10))
            _m_idx = _np.where(_final_cids == _cid)[0]
            _sims = _X_vs_concat[_m_idx] @ _cluster_centroids_view_s[_cid]
            _top6 = _m_idx[_np.argsort(-_sims)[:6]]
            for _k, _idx in enumerate(_top6):
                _x0 = 205 + _k * (3 * _tw + _pad_x)
                _ima = _Image.open(_df["path_pre"].iloc[_idx]).convert("RGB").resize((_tw, _th))
                _imb = _Image.open(_df["path_post"].iloc[_idx]).convert("RGB").resize((_tw, _th))
                _hm = _d3_maps[_idx]
                _hm_u8 = _np.clip((_hm - _hm.min()) / (_hm.max() - _hm.min() + 1e-6) * 255.0, 0, 255).astype(_np.uint8)
                _hm_rgb = _cv2.applyColorMap(_cv2.resize(_hm_u8, (_tw, _th), interpolation=_cv2.INTER_CUBIC), _cv2.COLORMAP_JET)[:, :, ::-1]
                _canv.paste(_ima, (_x0, _y0 + 14))
                _canv.paste(_imb, (_x0 + _tw, _y0 + 14))
                _canv.paste(_Image.fromarray(_hm_rgb), (_x0 + 2 * _tw, _y0 + 14))
                _drw.text((_x0, _y0 + 1), f"{_df['id'].iloc[_idx]}", fill=(30, 30, 30))
        _pg_p = f"/marimo/Fenomena/audit_cluster_cards_k72_p{_pg+1}.png"
        _canv.save(_pg_p)
        _page_paths.append(_pg_p)

    _ims = [_Image.open(p) for p in _page_paths]
    _comb = _Image.new("RGB", (_ims[0].width // 2, sum(im.height // 2 for im in _ims)))
    _cy = 0
    for _im in _ims:
        _r = _im.resize((_im.width // 2, _im.height // 2), _Image.LANCZOS)
        _comb.paste(_r, (0, _cy))
        _cy += _r.height
    _comb.save("/marimo/Fenomena/audit_cluster_cards_k72.png")
    print(f"Saved audit_cluster_cards_k72.png + 3 high-res pages in {_time.time()-_t0:.1f}s")

    return


@app.cell
def K23c_extract_satmae_dinov3():
    # Bagian 23C: Ekstraksi 5-Crop (1 Global + 4 Kuadran 512x512 dari 1024x1024)
    # untuk SatMAE++ (MVRL/satmaepp_ViT-L_pretrain_fmow_rgb, CLS norm weights) dan DINOv3-Large.
    import gc as _gc
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from pathlib import Path as _Path
    import cv2 as _cv2
    import numpy as _np
    import pandas as _pd
    import torch as _torch
    import torch.nn.functional as _F
    import timm as _timm
    from huggingface_hub import hf_hub_download as _hf_dl

    _t0 = _time.time()
    _ART = _Path("/marimo/Fenomena")
    _df_inv = _pd.read_parquet(_ART / "df_inventory.parquet")
    _MEAN = _torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    _STD = _torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def _extract_5crop_timm(_model, _sz, _tag):
        def _load5(_p):
            _im = _cv2.cvtColor(_cv2.imread(_p, _cv2.IMREAD_COLOR), _cv2.COLOR_BGR2RGB)
            _h, _w = _im.shape[:2]
            _mh, _mw = _h // 2, _w // 2
            return _np.stack([
                _cv2.resize(_im, (_sz, _sz), interpolation=_cv2.INTER_AREA),
                _cv2.resize(_im[:_mh, :_mw], (_sz, _sz), interpolation=_cv2.INTER_AREA),
                _cv2.resize(_im[:_mh, _mw:], (_sz, _sz), interpolation=_cv2.INTER_AREA),
                _cv2.resize(_im[_mh:, :_mw], (_sz, _sz), interpolation=_cv2.INTER_AREA),
                _cv2.resize(_im[_mh:, _mw:], (_sz, _sz), interpolation=_cv2.INTER_AREA),
            ], axis=0)

        @_torch.no_grad()
        def _side(_paths):
            _gl, _ql = [], []
            _BS = 48
            _nb = (len(_paths) + _BS - 1) // _BS
            for _bi in range(_nb):
                _ch = _paths[_bi * _BS : (_bi + 1) * _BS]
                with _TPE(max_workers=16) as _ex:
                    _arr = _np.stack(list(_ex.map(_load5, _ch)), axis=0)
                _b = _arr.shape[0]
                _xt = _torch.from_numpy(_arr.reshape(-1, _sz, _sz, 3)).permute(0, 3, 1, 2).cuda().float().div_(255.0)
                _xt = ((_xt - _MEAN) / _STD).to(_torch.bfloat16)
                _f = _F.normalize(_model(_xt).float(), dim=-1).view(_b, 5, -1)
                _gl.append(_f[:, 0, :].cpu().numpy())
                _ql.append(_F.normalize(_f[:, 1:, :].mean(dim=1), dim=-1).cpu().numpy())
                if (_bi + 1) % 25 == 0 or (_bi + 1) == _nb:
                    print(f"  [{_tag}] batch {_bi+1}/{_nb} ({_time.time()-_t0:.1f}s)", flush=True)
            return _np.vstack(_gl), _np.vstack(_ql)

        _gp, _qp = _side(_df_inv["path_pre"].tolist())
        _gs, _qs = _side(_df_inv["path_post"].tolist())
        return _np.concatenate([0.5*(_gp+_gs), _np.abs(_gs-_gp), 0.5*(_qp+_qs), _np.abs(_qs-_qp)], axis=1).astype(_np.float32)

    # 1. SatMAE++ ViT-L/16 (fMoW RGB, gunakan global_pool="token" agar norm.weight/norm.bias terpakai penuh)
    _sat_out = _ART / "emb_SatMAEPP_5crop_sym.npy"
    if not _sat_out.is_file():
        print("Memuat MVRL/satmaepp_ViT-L_pretrain_fmow_rgb (global_pool='token')...", flush=True)
        _sat_bin = _hf_dl("MVRL/satmaepp_ViT-L_pretrain_fmow_rgb", "pytorch_model.bin")
        _sd = _torch.load(_sat_bin, map_location="cpu", weights_only=False)
        if isinstance(_sd, dict) and "model" in _sd:
            _sd = _sd["model"]
        _sat_mdl = _timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="token")
        _enc_sd = {_k: _v for _k, _v in _sd.items() if not _k.startswith(("decoder_", "mask_token", "proj_up", "up_block", "fc_norm"))}
        _miss, _unexp = _sat_mdl.load_state_dict(_enc_sd, strict=False)
        print("SatMAE++ encoder missing keys:", _miss, "| unexpected:", _unexp, flush=True)
        assert len(_miss) == 0, f"Missing keys on SatMAE++: {_miss}"
        _sat_mdl = _sat_mdl.cuda().eval().to(_torch.bfloat16)
        _sat_emb = _extract_5crop_timm(_sat_mdl, 224, "SatMAE++")
        _np.save(_sat_out, _sat_emb)
        del _sat_mdl, _sd, _enc_sd
        _gc.collect()
        _torch.cuda.empty_cache()

    # 2. DINOv3-Large (sudah ter-cache jika sudah diekstrak)
    _d3_out = _ART / "emb_DINOv3L_5crop_sym.npy"
    if not _d3_out.is_file():
        print("Memuat timm/vit_large_patch16_dinov3.lvd1689m...", flush=True)
        _d3_mdl = _timm.create_model("hf_hub:timm/vit_large_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0).cuda().eval().to(_torch.bfloat16)
        _d3_sz = _d3_mdl.default_cfg.get("input_size", (3, 256, 256))[-1]
        _d3_emb = _extract_5crop_timm(_d3_mdl, int(_d3_sz), "DINOv3-L")
        _np.save(_d3_out, _d3_emb)
        del _d3_mdl
        _gc.collect()
        _torch.cuda.empty_cache()

    print("Selesai K23c! SatMAE++:", _np.load(_sat_out, mmap_mode="r").shape, "| DINOv3-L:", _np.load(_d3_out, mmap_mode="r").shape)
    return


@app.cell
def K24a_setup_512_and_lib():
    # Bagian 24A: Cache RAM 512x512 LAB CLAHE (/dev/shm) & Modul Bersama SymmetricSiameseNet
    import time as _time
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from pathlib import Path as _Path
    import cv2 as _cv2
    import numpy as _np
    import pandas as _pd

    _t0 = _time.time()
    _ART = _Path("/marimo/Fenomena")
    _SHM_CACHE = _Path("/dev/shm/fenomena_512_lab_clahe.npy")
    _df_inv = _pd.read_parquet(_ART / "df_inventory.parquet")

    if not _SHM_CACHE.is_file():
        print("Membangun cache 512x512 LAB CLAHE (L-only) ke /dev/shm...", flush=True)
        _n = len(_df_inv)
        _pre_paths = _df_inv["path_pre"].tolist()
        _post_paths = _df_inv["path_post"].tolist()

        def _proc_one(_args):
            _p1, _p2 = _args
            _clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _pair = _np.empty((2, 512, 512, 3), dtype=_np.uint8)
            for _j, _p in enumerate((_p1, _p2)):
                _im = _cv2.cvtColor(_cv2.imread(_p, _cv2.IMREAD_COLOR), _cv2.COLOR_BGR2RGB)
                _im = _cv2.resize(_im, (512, 512), interpolation=_cv2.INTER_AREA)
                _lab = _cv2.cvtColor(_im, _cv2.COLOR_RGB2LAB)
                _lab[:, :, 0] = _clahe.apply(_lab[:, :, 0])
                _pair[_j] = _cv2.cvtColor(_lab, _cv2.COLOR_LAB2RGB)
            return _pair

        _buf = _np.empty((_n, 2, 512, 512, 3), dtype=_np.uint8)
        with _TPE(max_workers=20) as _ex:
            for _idx, _res in enumerate(_ex.map(_proc_one, zip(_pre_paths, _post_paths), chunksize=32)):
                _buf[_idx] = _res
                if (_idx + 1) % 1000 == 0 or (_idx + 1) == _n:
                    print(f"  [Cache 512 LAB] {_idx+1}/{_n} ({_time.time()-_t0:.1f}s)", flush=True)
        _np.save(_SHM_CACHE, _buf)
        del _buf
    print(f"Cache siap: {_SHM_CACHE} shape={_np.load(_SHM_CACHE, mmap_mode='r').shape}", flush=True)

    _LIB_CODE = """
    import math
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score, classification_report

    CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    _RAM_ARRAY_CACHE = {}

    def get_ram_cache(cache_path):
        key = str(cache_path)
        if key not in _RAM_ARRAY_CACHE:
            _RAM_ARRAY_CACHE[key] = np.load(key)
        return _RAM_ARRAY_CACHE[key]

    class ChannelAttention1D(nn.Module):
        def __init__(self, dim, reduction=4):
            super().__init__()
            hid = max(32, dim // reduction)
            self.net = nn.Sequential(
                nn.Linear(dim, hid),
                nn.GELU(),
                nn.Linear(hid, dim),
                nn.Sigmoid(),
            )
        def forward(self, x):
            return x * self.net(x)

    class SymmetricSiameseNet(nn.Module):
        def __init__(self, backbone_name, num_classes=6, pretrained=True, drop_rate=0.35):
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_path_rate=0.2)
            d = self.backbone.num_features
            self.diff_attn = ChannelAttention1D(d, reduction=4)
            self.head = nn.Sequential(
                nn.LayerNorm(d * 3),
                nn.Dropout(drop_rate),
                nn.Linear(d * 3, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )

        def forward(self, x1, x2):
            feats = self.backbone(torch.cat([x1, x2], dim=0))
            f1, f2 = feats.chunk(2, dim=0)
            f_mean = 0.5 * (f1 + f2)
            f_max = torch.maximum(f1, f2)
            f_diff = self.diff_attn(torch.abs(f2 - f1))
            z = torch.cat([f_mean, f_max, f_diff], dim=1)
            return self.head(z)

    def soft_logloss(y_soft, probs):
        p = np.clip(probs, 1e-7, 1.0)
        return float(-(y_soft * np.log(p)).sum(axis=1).mean())

    def optimize_multipliers(probs, y_true_idx):
        def _loss(m):
            w = np.exp(m)
            return -f1_score(y_true_idx, (probs * w).argmax(axis=1), average="macro")
        res = minimize(_loss, np.zeros(probs.shape[1]), method="Powell", options={"maxiter": 600})
        return np.exp(res.x)

    def eval_nested_cal_f1(probs, y_true_idx, folds, damp=0.85):
        pred_idx = np.zeros(len(y_true_idx), dtype=np.int64)
        uniq = np.unique(folds)
        if len(uniq) <= 1:
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(y_true_idx))
            half = len(perm) // 2
            w1 = optimize_multipliers(probs[perm[:half]], y_true_idx[perm[:half]]) ** damp
            w2 = optimize_multipliers(probs[perm[half:]], y_true_idx[perm[half:]]) ** damp
            pred_idx[perm[half:]] = (probs[perm[half:]] * w1).argmax(axis=1)
            pred_idx[perm[:half]] = (probs[perm[:half]] * w2).argmax(axis=1)
        else:
            for f in uniq:
                tr = (folds != f)
                va = (folds == f)
                w = optimize_multipliers(probs[tr], y_true_idx[tr]) ** damp
                pred_idx[va] = (probs[va] * w).argmax(axis=1)
        return float(f1_score(y_true_idx, pred_idx, average="macro")), pred_idx

    def _prep_batch_gpu(pair_np, img_size, train, mean_t, std_t):
        t = torch.from_numpy(pair_np).cuda(non_blocking=True)
        x1 = t[:, 0].permute(0, 3, 1, 2).float().div_(255.0)
        x2 = t[:, 1].permute(0, 3, 1, 2).float().div_(255.0)
        if img_size != x1.shape[-1]:
            x1 = F.interpolate(x1, size=(img_size, img_size), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(img_size, img_size), mode="bilinear", align_corners=False)
        if train:
            if np.random.rand() < 0.5:
                x1 = torch.flip(x1, dims=[3]); x2 = torch.flip(x2, dims=[3])
            if np.random.rand() < 0.5:
                x1 = torch.flip(x1, dims=[2]); x2 = torch.flip(x2, dims=[2])
            k = int(np.random.randint(0, 4))
            if k > 0:
                x1 = torch.rot90(x1, k=k, dims=[2, 3]); x2 = torch.rot90(x2, k=k, dims=[2, 3])
            b = x1.shape[0]
            s1 = torch.empty((b, 1, 1, 1), device="cuda").uniform_(0.92, 1.08)
            s2 = torch.empty((b, 1, 1, 1), device="cuda").uniform_(0.92, 1.08)
            x1 = (x1 * s1).clamp_(0.0, 1.0)
            x2 = (x2 * s2).clamp_(0.0, 1.0)
            swap = (torch.rand((b, 1, 1, 1), device="cuda") < 0.5)
            x1_s = torch.where(swap, x2, x1)
            x2_s = torch.where(swap, x1, x2)
            x1, x2 = x1_s, x2_s
        x1 = (x1 - mean_t) / std_t
        x2 = (x2 - mean_t) / std_t
        return x1, x2

    @torch.no_grad()
    def predict_4x_tta(model, arr, indices, img_size, batch_size=64):
        model.eval()
        mean_t = IMAGENET_MEAN.cuda()
        std_t = IMAGENET_STD.cuda()
        preds = []
        for s in range(0, len(indices), batch_size):
            b_idx = indices[s : s + batch_size]
            x1, x2 = _prep_batch_gpu(arr[b_idx], img_size, False, mean_t, std_t)
            x1_f = torch.flip(x1, dims=[3])
            x2_f = torch.flip(x2, dims=[3])
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p1 = F.softmax(model(x1, x2).float(), dim=-1)
                p2 = F.softmax(model(x2, x1).float(), dim=-1)
                p3 = F.softmax(model(x1_f, x2_f).float(), dim=-1)
                p4 = F.softmax(model(x2_f, x1_f).float(), dim=-1)
            preds.append((0.25 * (p1 + p2 + p3 + p4)).cpu().numpy())
        return np.vstack(preds).astype(np.float32)

    @torch.no_grad()
    def _predict_1x_fast(model, arr, indices, img_size, batch_size=64):
        model.eval()
        mean_t = IMAGENET_MEAN.cuda()
        std_t = IMAGENET_STD.cuda()
        preds = []
        for s in range(0, len(indices), batch_size):
            b_idx = indices[s : s + batch_size]
            x1, x2 = _prep_batch_gpu(arr[b_idx], img_size, False, mean_t, std_t)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds.append(F.softmax(model(x1, x2).float(), dim=-1).cpu().numpy())
        return np.vstack(preds).astype(np.float32)

    def train_siamese_folds(
        tag, backbone_name, img_size, cache_path, targets, folds, fold_ids,
        epochs=14, batch_size=64, lr_bb=1.5e-4, lr_head=5e-4, mixup_p=0.5, out_dir="/marimo/Fenomena"
    ):
        t_start = time.time()
        out_dir = Path(out_dir)
        arr = get_ram_cache(cache_path)
        y_hard = targets.argmax(axis=1)
        mean_t = IMAGENET_MEAN.cuda()
        std_t = IMAGENET_STD.cuda()
        oof_part = np.zeros_like(targets, dtype=np.float32)
        fold_stats = {}

        for f in fold_ids:
            t_f = time.time()
            torch.manual_seed(42 + int(f))
            rng = np.random.RandomState(42 + int(f))
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            n_batches = max(1, len(tr_idx) // batch_size)

            model = SymmetricSiameseNet(backbone_name, num_classes=6, pretrained=True).cuda()
            opt = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": lr_bb},
                {"params": list(model.diff_attn.parameters()) + list(model.head.parameters()), "lr": lr_head},
            ], weight_decay=1e-2)
            total_steps = epochs * n_batches
            warmup_steps = min(total_steps // 6, 2 * n_batches)
            def _lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

            best_ll = 999.0
            ckpt_path = out_dir / f"ckpt_{tag}_f{f}.pt"

            for ep in range(1, epochs + 1):
                model.train()
                tr_loss = 0.0
                perm_idx = rng.permutation(tr_idx)
                for bi in range(n_batches):
                    b_idx = perm_idx[bi * batch_size : (bi + 1) * batch_size]
                    x1, x2 = _prep_batch_gpu(arr[b_idx], img_size, True, mean_t, std_t)
                    y_soft = torch.from_numpy(targets[b_idx]).cuda(non_blocking=True)
                    if rng.rand() < mixup_p:
                        lam = float(rng.beta(0.2, 0.2))
                        p = torch.randperm(x1.size(0), device="cuda")
                        x1 = lam * x1 + (1.0 - lam) * x1[p]
                        x2 = lam * x2 + (1.0 - lam) * x2[p]
                        y_soft = lam * y_soft + (1.0 - lam) * y_soft[p]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x1, x2)
                        loss = -(y_soft * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tr_loss += float(loss.detach())
                tr_loss /= n_batches

                if ep >= epochs - 5 or ep % 3 == 0:
                    p_va = _predict_1x_fast(model, arr, va_idx, img_size, batch_size=batch_size)
                    ll = soft_logloss(targets[va_idx], p_va)
                    raw_f1 = float(f1_score(y_hard[va_idx], p_va.argmax(axis=1), average="macro"))
                    mark = ""
                    if ll < best_ll:
                        best_ll = ll
                        torch.save(model.state_dict(), ckpt_path)
                        mark = " *"
                    print(f"  [{tag} F{f}] ep {ep:02d}/{epochs} tr={tr_loss:.4f} va_ll={ll:.4f} va_f1={raw_f1:.4f} ({time.time()-t_f:.1f}s){mark}", flush=True)
                else:
                    print(f"  [{tag} F{f}] ep {ep:02d}/{epochs} tr={tr_loss:.4f} ({time.time()-t_f:.1f}s)", flush=True)

            # Evaluate best checkpoint with full 4x Bidirectional Swap + HFlip TTA
            model.load_state_dict(torch.load(ckpt_path, map_location="cuda", weights_only=True))
            best_probs = predict_4x_tta(model, arr, va_idx, img_size, batch_size=batch_size)
            tta_ll = soft_logloss(targets[va_idx], best_probs)
            tta_f1 = float(f1_score(y_hard[va_idx], best_probs.argmax(axis=1), average="macro"))
            oof_part[va_idx] = best_probs
            fold_stats[int(f)] = {"logloss": round(tta_ll, 4), "raw_f1": round(tta_f1, 4), "sec": round(time.time() - t_f, 1)}
            np.save(out_dir / f"oof_part_{tag}_f{f}.npy", best_probs)
            print(f"  [{tag} F{f}] BEST 4xTTA va_ll={tta_ll:.4f} va_f1={tta_f1:.4f} ({time.time()-t_f:.1f}s)", flush=True)
            del model, opt, sched
            torch.cuda.empty_cache()

        print(f"Selesai {tag} folds={fold_ids} dalam {time.time()-t_start:.1f}s: {fold_stats}", flush=True)
        return oof_part, fold_stats
    """
    (_ART / "fenomena_lib.py").write_text(_LIB_CODE.strip() + "\n", encoding="utf-8")
    print("Ditulis ulang fenomena_lib.py (zero-IPC RAM batching + chunked siamese)", flush=True)
    return


if __name__ == "__main__":
    app.run()
