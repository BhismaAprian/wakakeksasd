# Multi-prompt z-score de-biased CLIP teacher (RemoteCLIP + SkyCLIP)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from huggingface_hub import hf_hub_download

ART = Path("/marimo/Fenomena")
CLASS_NAMES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]

PROMPTS = {
    "Earthquake": [
        "aerial satellite view of collapsed buildings and gray concrete rubble after an earthquake",
        "urban residential blocks shattered and crumbled into debris piles from earthquake shaking",
        "earthquake structural devastation with fractured roads and demolished multi-story buildings without burn scars",
        "post-earthquake aerial imagery showing widespread roof collapses and gray dust over city streets",
        "satellite change detection showing intact neighborhoods turned into earthquake rubble fields",
    ],
    "Fire": [
        "aerial satellite image of charred black vegetation burn scar from wildfire",
        "active forest fire and brushfire smoke plumes over scorched dark brown terrain",
        "post-wildfire satellite view with blackened burnt trees while paved roads remain intact",
        "burnt farmland and forest canopy turned black and dark ash by wildfire",
        "aerial change detection of green vegetation transformed into scorched wildfire burn scar",
    ],
    "Hurricane": [
        "aerial satellite image of hurricane wind damage with stripped roofs and scattered debris",
        "widespread tropical cyclone damage showing uprooted trees and flooded inland neighborhoods",
        "post-hurricane aerial view with torn blue tarps on damaged roofs and storm surge puddles",
        "wind-damaged suburban houses and fallen forest timber caused by severe hurricane",
        "satellite change detection showing hurricane wind destruction and inland rainwater flooding",
    ],
    "Tsunami": [
        "aerial satellite image of tsunami coastal flooding with shoreline buildings swept completely bare",
        "eroded receding coastline and muddy seawater inundation extending inland after a tsunami",
        "coastal settlement washed away to bare mud and debris lines by tsunami ocean waves",
        "post-tsunami satellite view showing flooded harbor, washed-out coastal roads, and stranded boats",
        "aerial change detection of coastal town obliterated by tsunami wave surge",
    ],
    "Volcano": [
        "aerial satellite image of thick gray or white volcanic ash blanket covering land and roofs",
        "volcanic eruption aftermath with dark basaltic lava flows and pyroclastic deposits burying terrain",
        "satellite view of volcanic crater rim, ash fallout plumes, and ash-covered vegetation",
        "post-volcano aerial imagery showing smooth gray tephra deposits and hardened lava channels",
        "aerial change detection of green valley buried under volcanic ash and lava flow",
    ],
    "Explosion": [
        "aerial satellite image of a localized circular explosion crater and radial blast damage",
        "industrial or port blast epicenter showing circular shockwave destruction radiating outward",
        "concentrated explosion site with scorched blast crater and shattered nearby warehouses",
        "post-explosion aerial view showing single-point blast crater surrounded by radially blown roofs",
        "aerial change detection of intact industrial facility replaced by a localized blast crater",
    ],
}

def encode_prompts(repo_id, filename):
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
    model = model.cuda().eval()
    cls_embs = []
    with torch.no_grad():
        for cname in CLASS_NAMES:
            t = tok(PROMPTS[cname]).cuda()
            tf = F.normalize(model.encode_text(t).float(), dim=-1)
            tf_mean = F.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
            cls_embs.append(tf_mean.cpu().numpy()[0])
    del model
    torch.cuda.empty_cache()
    return np.stack(cls_embs, axis=0)  # (6, 768)

def zscore_softmax(sim_mat, temp=0.5):
    z = (sim_mat - sim_mat.mean(axis=0, keepdims=True)) / (sim_mat.std(axis=0, keepdims=True) + 1e-8)
    z = z / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

print("Encoding RemoteCLIP prompts...")
rc_tf = encode_prompts("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt")
rc_diff = 0.5 * (np.load(ART / "emb_RemoteCLIP_diff.npy") + np.load(ART / "emb_RemoteCLIP_diff_hflip.npy"))
rc_diff = rc_diff / (np.linalg.norm(rc_diff, axis=1, keepdims=True) + 1e-8)
rc_post = np.load(ART / "emb_RemoteCLIP_post.npy")
rc_sim = 0.65 * (rc_diff @ rc_tf.T) + 0.35 * (rc_post @ rc_tf.T)
p_rc = zscore_softmax(rc_sim, temp=0.5)

print("Encoding SkyCLIP prompts...")
sc_tf = encode_prompts("furiousteabag/SkyCLIP", "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS_epoch_20.pt")
sc_diff = np.load(ART / "emb_SkyCLIP_diff.npy")
sc_post = np.load(ART / "emb_SkyCLIP_post.npy")
sc_sim = 0.65 * (sc_diff @ sc_tf.T) + 0.35 * (sc_post @ sc_tf.T)
p_sc = zscore_softmax(sc_sim, temp=0.5)

clip_probs = (0.55 * p_rc + 0.45 * p_sc).astype(np.float32)
np.save(ART / "clip_teacher_probs.npy", clip_probs)

preds = clip_probs.argmax(axis=1)
vc = pd.Series([CLASS_NAMES[i] for i in preds]).value_counts()
print("CLIP Teacher argmax distribution:")
for c in CLASS_NAMES:
    n = int(vc.get(c, 0))
    print(f"  {c:12s}: {n:4d} ({n/len(preds)*100:5.2f}%)")
print("Mean max prob:", float(clip_probs.max(axis=1).mean()))
