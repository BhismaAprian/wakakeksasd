# High-Res Qwen2.5-VL-32B-Instruct Teacher (448x448 per tile, 896x448 side-by-side) + 2x HFlip TTA
import gc, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import torch
import transformers

ART = Path("/marimo/Fenomena")
CLASS_NAMES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
CANON = {c.lower(): c for c in CLASS_NAMES}
PARTIAL_JSONL = ART / "_vlm32b_partial.jsonl"
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
    "Respond ONLY with valid JSON: {\"label\": \"<one of the six>\", \"confidence\": <0.0-1.0>, \"reason\": \"<short>\"}."
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

def parse_vlm(txt):
    clean = re.sub(r"```(?:json)?|```", "", str(txt)).strip()
    m = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            raw_lbl = str(d.get("label", "")).strip().lower()
            lbl = CANON.get(raw_lbl)
            if lbl is None:
                for k, v in CANON.items():
                    if k in raw_lbl:
                        lbl = v
                        break
            conf = float(np.clip(float(d.get("confidence", 0.5)), 0.0, 1.0))
            rsn = str(d.get("reason", ""))[:180]
            if lbl is not None:
                return lbl, conf, rsn, True
        except Exception:
            pass
    lbl = None
    for k, v in CANON.items():
        if re.search(rf"\b{k}\b", clean, flags=re.IGNORECASE):
            lbl = v
            break
    mc = re.search(r"\"?confidence\"?\s*:\s*([0-9]*\.?[0-9]+)", clean)
    conf = float(np.clip(float(mc.group(1)), 0.0, 1.0)) if mc else (0.55 if lbl else 0.0)
    if lbl is not None:
        return lbl, conf, clean[:150], True
    return "Earthquake", 0.2, clean[:150], False

def label_conf_to_dist(lbl, conf):
    idx = CLASS_NAMES.index(lbl)
    conf = float(np.clip(conf, 0.20, 0.98))
    p = np.full(6, (1.0 - conf) / 5.0, dtype=np.float32)
    p[idx] = conf
    return p

def temp_calibrate(p, T=0.75):
    logits = np.log(np.clip(p, 1e-8, 1.0)) / T
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

t0 = time.time()
df_inv = pd.read_parquet(ART / "df_inventory.parquet")
print(f"[{time.strftime('%H:%M:%S')}] Loading {MODEL_ID} in bfloat16...", flush=True)
cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) or getattr(transformers, "AutoModelForImageTextToText", None)
processor = transformers.AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True,
    min_pixels=448 * 448, max_pixels=896 * 448 + 4096,
)
if hasattr(processor, "tokenizer"):
    processor.tokenizer.padding_side = "left"

model = cls.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
).cuda().eval()
print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time()-t0:.1f}s!", flush=True)

done_by_id = {}
if PARTIAL_JSONL.is_file():
    with open(PARTIAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                done_by_id[r["id"]] = r
print(f"Resuming: {len(done_by_id)} / {len(df_inv)} already done.", flush=True)

todo = [r for r in df_inv.to_dict("records") if r["id"] not in done_by_id]
BATCH = 16  # 16 pairs * 2 TTA = 32 images or run orig & hflip sequentially per chunk

def infer_batch(rows, hflip):
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(lambda r: make_sbs(r, 448, hflip=hflip), rows))
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": RUBRIC_PROMPT}]}] for im in imgs]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    inp = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
    gens = processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return [parse_vlm(g) for g in gens]

t_inf = time.time()
n_batches = (len(todo) + BATCH - 1) // BATCH
with open(PARTIAL_JSONL, "a", encoding="utf-8") as flog:
    for bi in range(n_batches):
        tb = time.time()
        chunk = todo[bi * BATCH : (bi + 1) * BATCH]
        res_orig = infer_batch(chunk, hflip=False)
        res_flip = infer_batch(chunk, hflip=True)
        for row, (l0, c0, r0, ok0), (l1, c1, r1, ok1) in zip(chunk, res_orig, res_flip):
            p_avg = 0.5 * (label_conf_to_dist(l0, c0) + label_conf_to_dist(l1, c1))
            p_cal = temp_calibrate(p_avg[None, :], T=0.75)[0]
            rec = {
                "id": row["id"],
                "orig_label": l0, "orig_conf": c0, "orig_ok": ok0,
                "flip_label": l1, "flip_conf": c1, "flip_ok": ok1,
                "agree_tta": (l0 == l1),
                "reason": r0,
                "probs": [round(float(x), 6) for x in p_cal],
            }
            done_by_id[row["id"]] = rec
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
        flog.flush()
        if (bi + 1) % 5 == 0 or bi == 0 or (bi + 1) == n_batches:
            el = time.time() - t_inf
            eta = (el / (bi + 1)) * (n_batches - bi - 1)
            print(f"[{time.strftime('%H:%M:%S')}] Batch {bi+1}/{n_batches} ({len(done_by_id)}/3180) | dt={time.time()-tb:.1f}s | ETA={eta/60:.1f}m", flush=True)

# Build P_vlm32b and fuse with clip_teacher_probs
ordered = [done_by_id[sid] for sid in df_inv["id"]]
p_vlm32b = np.array([r["probs"] for r in ordered], dtype=np.float32)
np.save(ART / "vlm32b_teacher_probs.npy", p_vlm32b)

p_clip = np.load(ART / "clip_teacher_probs.npy")
consensus_soft = (0.72 * p_vlm32b + 0.28 * p_clip).astype(np.float32)
consensus_soft /= consensus_soft.sum(axis=1, keepdims=True)

vlm_argmax = p_vlm32b.argmax(axis=1)
clip_argmax = p_clip.argmax(axis=1)
agree = (vlm_argmax == clip_argmax).astype(np.float32)
sample_weight = (0.65 + 0.35 * agree).astype(np.float32)

np.save(ART / "consensus_soft_v1.npy", consensus_soft)

cons_lbl_idx = consensus_soft.argmax(axis=1)
cons_conf = consensus_soft.max(axis=1)
df_v2 = pd.DataFrame({
    "id": df_inv["id"],
    "vlm32b_label": [CLASS_NAMES[i] for i in vlm_argmax],
    "vlm32b_conf": p_vlm32b.max(axis=1),
    "vlm32b_tta_agree": [r["agree_tta"] for r in ordered],
    "clip_label": [CLASS_NAMES[i] for i in clip_argmax],
    "clip_conf": p_clip.max(axis=1),
    "consensus_label": [CLASS_NAMES[i] for i in cons_lbl_idx],
    "consensus_conf": cons_conf,
    "teacher_agree": agree.astype(bool),
    "sample_weight": sample_weight,
    "parse_ok": [r["orig_ok"] or r["flip_ok"] for r in ordered],
})
df_v2.to_parquet(ART / "pseudo_labels_v2.parquet", index=False)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = np.zeros(len(df_v2), dtype=np.int64)
for f_idx, (_, va_idx) in enumerate(skf.split(df_v2, df_v2["consensus_label"])):
    folds[va_idx] = f_idx
np.save(ART / "folds_skf5.npy", folds)

print("\n=== FASE 1 SELESAI ===", flush=True)
print("VLM32B parse_ok %:", float(df_v2["parse_ok"].mean() * 100), "| TTA agree %:", float(df_v2["vlm32b_tta_agree"].mean() * 100), "| VLM-CLIP agree %:", float(agree.mean() * 100))
print("\nDistribusi Kelas (VLM-32B vs CLIP vs Consensus v2):")
for c in CLASS_NAMES:
    nv = int((df_v2["vlm32b_label"] == c).sum())
    nc = int((df_v2["clip_label"] == c).sum())
    nk = int((df_v2["consensus_label"] == c).sum())
    print(f"  {c:12s} | VLM32B: {nv:4d} ({nv/3180*100:5.2f}%) | CLIP: {nc:4d} ({nc/3180*100:5.2f}%) | Consensus: {nk:4d} ({nk/3180*100:5.2f}%)")
