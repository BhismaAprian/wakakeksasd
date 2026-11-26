# Fast Logit-Level Qwen2.5-VL-32B-Instruct Teacher (448x448 per tile, 896x448 side-by-side) + 2x HFlip TTA + Prior De-Biasing
import json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import transformers
from sklearn.model_selection import StratifiedKFold

ART = Path("/marimo/Fenomena")
CLASS_NAMES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

RUBRIC_PROMPT = (
    "You are a disaster remote-sensing expert analyzing a side-by-side satellite image pair "
    "(LEFT = pre-disaster, RIGHT = post-disaster) of the exact same location.\n"
    "Classify the disaster into EXACTLY ONE of: Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion.\n"
    "Use this contrastive visual rubric strictly (do NOT confuse dry seasonal grass, shadows, or gray rubble with Fire):\n"
    "- Fire: blackened/charred vegetation burn scar or active wildfire smoke plumes while roads/buildings remain structurally intact.\n"
    "- Explosion: localized blast crater or industrial/building destruction radiating radially from a single epicenter.\n"
    "- Earthquake: widespread collapsed buildings, gray concrete rubble piles, cracked urban blocks without wide burn scars or coastal wave sweep.\n"
    "- Hurricane: widespread wind damage (stripped roofs, scattered uprooted trees, inland rainwater puddles/flooding).\n"
    "- Tsunami: coastal zone with eroded/receded shoreline, beachfront buildings swept completely bare, and muddy seawater deposits inland.\n"
    "- Volcano: thick gray/white volcanic ash blanket, pyroclastic deposits, or dark lava streams burying land/settlements.\n"
    "Answer with the exact single class name ONLY (one of: Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion)."
)

def make_sbs(row, tile_sz=448, hflip=False):
    with Image.open(row["path_pre"]) as im1, Image.open(row["path_post"]) as im2:
        r1 = im1.convert("RGB").resize((tile_sz, tile_sz), Image.Resampling.BILINEAR)
        r2 = im2.convert("RGB").resize((tile_sz, tile_sz), Image.Resampling.BILINEAR)
    if hflip:
        r1 = r1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        r2 = r2.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    canvas = Image.new("RGB", (tile_sz * 2, tile_sz))
    canvas.paste(r1, (0, 0))
    canvas.paste(r2, (tile_sz, 0))
    return canvas

t0 = time.time()
df_inv = pd.read_parquet(ART / "df_inventory.parquet")
cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) or getattr(transformers, "AutoModelForImageTextToText", None)
processor = transformers.AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True,
    min_pixels=448 * 448, max_pixels=896 * 448 + 4096,
)
processor.tokenizer.padding_side = "left"

# Exact first token IDs (both unspaced & spaced) for each of the 6 classes
tok = processor.tokenizer
CLASS_TOK_PAIRS = [
    (tok.encode(c, add_special_tokens=False)[0], tok.encode(" " + c, add_special_tokens=False)[0])
    for c in CLASS_NAMES
]
print("Class token ID pairs:", dict(zip(CLASS_NAMES, CLASS_TOK_PAIRS)), flush=True)

model = cls.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
).cuda().eval()
print(f"[{time.strftime('%H:%M:%S')}] Loaded 32B model in {time.time()-t0:.1f}s", flush=True)

PARTIAL_NPY = ART / "_vlm32b_logits_partial.npz"
logits_orig = np.zeros((len(df_inv), 6), dtype=np.float32)
logits_flip = np.zeros((len(df_inv), 6), dtype=np.float32)
done_count = 0
if PARTIAL_NPY.is_file():
    z = np.load(PARTIAL_NPY)
    logits_orig = z["logits_orig"]
    logits_flip = z["logits_flip"]
    done_count = int(z["done_count"])
print(f"Resuming from {done_count}/{len(df_inv)}", flush=True)

BATCH = 16
rows_all = df_inv.to_dict("records")

@torch.no_grad()
def forward_logits(rows, hflip):
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(lambda r: make_sbs(r, 448, hflip=hflip), rows))
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": RUBRIC_PROMPT}]}] for im in imgs]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
    out = model(**inp, use_cache=False)
    last_logits = out.logits[:, -1, :].float()  # (B, vocab)
    # LogSumExp over unspaced & spaced first-token IDs for each class
    cls_scores = []
    for t_ns, t_sp in CLASS_TOK_PAIRS:
        pair_l = torch.stack([last_logits[:, t_ns], last_logits[:, t_sp]], dim=-1)
        cls_scores.append(torch.logsumexp(pair_l, dim=-1))
    return torch.stack(cls_scores, dim=-1).cpu().numpy()  # (B, 6)

t_inf = time.time()
n_batches = (len(rows_all) + BATCH - 1) // BATCH
start_bi = done_count // BATCH
for bi in range(start_bi, n_batches):
    tb = time.time()
    s_idx = bi * BATCH
    e_idx = min(s_idx + BATCH, len(rows_all))
    chunk = rows_all[s_idx:e_idx]
    logits_orig[s_idx:e_idx] = forward_logits(chunk, hflip=False)
    logits_flip[s_idx:e_idx] = forward_logits(chunk, hflip=True)
    done_count = e_idx
    if (bi + 1) % 10 == 0 or bi == start_bi or (bi + 1) == n_batches:
        np.savez(PARTIAL_NPY, logits_orig=logits_orig, logits_flip=logits_flip, done_count=np.array(done_count))
        el = time.time() - t_inf
        done_now = bi - start_bi + 1
        eta = (el / done_now) * (n_batches - bi - 1)
        raw_top = CLASS_NAMES[int((logits_orig[s_idx] + logits_flip[s_idx]).argmax())]
        print(f"[{time.strftime('%H:%M:%S')}] Batch {bi+1}/{n_batches} ({done_count}/3180) | dt={time.time()-tb:.2f}s | ETA={eta/60:.1f}m | sample0={raw_top}", flush=True)

np.save(ART / "vlm32b_raw_logits_orig.npy", logits_orig)
np.save(ART / "vlm32b_raw_logits_flip.npy", logits_flip)

# Average TTA logits, then apply temperature & class-prior de-biasing (centering)
avg_logits = 0.5 * (logits_orig + logits_flip)
raw_probs = F.softmax(torch.tensor(avg_logits) / 0.75, dim=-1).numpy()
raw_vc = pd.Series([CLASS_NAMES[i] for i in raw_probs.argmax(axis=1)]).value_counts()
print("\nRaw VLM-32B Logits Argmax (before de-biasing):")
for c in CLASS_NAMES:
    n = int(raw_vc.get(c, 0))
    print(f"  {c:12s}: {n:4d} ({n/3180*100:5.2f}%)")

# De-biasing: center each class logit column by subtracting damped column mean (alpha=0.85)
# Preserves per-sample relative visual ranking while eliminating VLM "Fire" prior bias!
centered_logits = avg_logits - 0.85 * avg_logits.mean(axis=0, keepdims=True)
# Normalize scale & apply T=0.75 temperature calibration
centered_logits = centered_logits / (centered_logits.std() + 1e-8) * 1.5
p_vlm32b = F.softmax(torch.tensor(centered_logits) / 0.75, dim=-1).numpy().astype(np.float32)
np.save(ART / "vlm32b_teacher_probs.npy", p_vlm32b)

p_clip = np.load(ART / "clip_teacher_probs.npy")
consensus_soft = (0.72 * p_vlm32b + 0.28 * p_clip).astype(np.float32)
consensus_soft /= consensus_soft.sum(axis=1, keepdims=True)

vlm_argmax = p_vlm32b.argmax(axis=1)
clip_argmax = p_clip.argmax(axis=1)
tta_agree = (logits_orig.argmax(axis=1) == logits_flip.argmax(axis=1))
agree = (vlm_argmax == clip_argmax).astype(np.float32)
sample_weight = (0.65 + 0.35 * agree).astype(np.float32)

np.save(ART / "consensus_soft_v1.npy", consensus_soft)
cons_lbl_idx = consensus_soft.argmax(axis=1)
df_v2 = pd.DataFrame({
    "id": df_inv["id"],
    "vlm32b_label": [CLASS_NAMES[i] for i in vlm_argmax],
    "vlm32b_conf": p_vlm32b.max(axis=1),
    "vlm32b_tta_agree": tta_agree,
    "clip_label": [CLASS_NAMES[i] for i in clip_argmax],
    "clip_conf": p_clip.max(axis=1),
    "consensus_label": [CLASS_NAMES[i] for i in cons_lbl_idx],
    "consensus_conf": consensus_soft.max(axis=1),
    "teacher_agree": agree.astype(bool),
    "sample_weight": sample_weight,
})
df_v2.to_parquet(ART / "pseudo_labels_v2.parquet", index=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(df_v2), dtype=np.int64)
for f_idx, (_, va_idx) in enumerate(skf.split(df_v2, df_v2["consensus_label"])):
    folds[va_idx] = f_idx
np.save(ART / "folds_skf5.npy", folds)

print("\n=== FASE 1 DE-BIASED SELESAI ===", flush=True)
print("TTA agree %:", round(float(tta_agree.mean() * 100), 2), "| VLM-CLIP agree %:", round(float(agree.mean() * 100), 2))
for c in CLASS_NAMES:
    nv = int((df_v2["vlm32b_label"] == c).sum())
    nc = int((df_v2["clip_label"] == c).sum())
    nk = int((df_v2["consensus_label"] == c).sum())
    print(f"  {c:12s} | VLM32B_deb: {nv:4d} ({nv/3180*100:5.2f}%) | CLIP: {nc:4d} ({nc/3180*100:5.2f}%) | Consensus: {nk:4d} ({nk/3180*100:5.2f}%)")
