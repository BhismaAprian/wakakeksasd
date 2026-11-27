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
