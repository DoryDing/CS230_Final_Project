"""
Baselines for continuity-error detection on the pilot pairs (milestone).

    python baselines/run_baselines.py --data /content/drive/MyDrive/<...>/data --out results

Reads <data>/manifest.csv from make_pairs.py. Every method outputs an error score per pair
(higher = more likely an error). The decision threshold is chosen to maximize F1 on VAL,
then F1 / precision / recall / AUROC are reported on TEST (held-out videos).

Methods
  B0  color-hist      L1 distance between global Lab color histograms (no learning, no DL)
  B1  dino-global     1 - cosine(DINOv2 CLS(a), CLS(b))            (no training)
  B2  dino-logreg     logistic regression on [|ea-eb|, ea*eb] of DINOv2 CLS (+mean patch)
  R1  dino-mask (oracle) 1 - cosine of DINOv2 patch features pooled inside the GT edit mask.
      Uses the ground-truth mask, so it is a reference for "perfect localization", not a fair baseline.

DINOv2 features are cached in <out>/feats.npz so re-runs are fast.
"""
import argparse, csv, json, os
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rows(data):
    rows = list(csv.DictReader(open(os.path.join(data, "manifest.csv"))))
    for r in rows:
        r["label"] = int(r["label"]); r["delta_e"] = float(r["delta_e"])
    return rows


# ------------------------------------------------------------------ features
def prep(img, short=224, max_long=392, patch=14):
    """Keep aspect ratio: short side 224, long side a multiple of 14 (<= 392)."""
    w, h = img.size
    if w <= h:
        nw, nh = short, min(max_long, round(h * short / w / patch) * patch)
    else:
        nh, nw = short, min(max_long, round(w * short / h / patch) * patch)
    x = (np.asarray(img.convert("RGB").resize((nw, nh), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1), (nh // patch, nw // patch)


def mask_to_grid(mask_path, grid):
    m = Image.open(mask_path).convert("L").resize((grid[1], grid[0]), Image.BOX)
    m = np.asarray(m, np.float32) / 255
    sel = m > 0.5
    return sel if sel.any() else m > 0


def extract(rows, data, out, model_name, device):
    cache = os.path.join(out, "feats.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["cls"].item(), z["meanp"].item(), z["maskp"].item(), z["hist"].item()
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    cls, meanp, maskp, hist = {}, {}, {}, {}
    frames = sorted({r["frame_a"] for r in rows} | {r["frame_b"] for r in rows})
    masks_for = {}
    for r in rows:  # GT mask of the pair applies to both frames (same shot, small motion)
        masks_for.setdefault(r["frame_a"], set()).add(r["mask_path"])
        masks_for.setdefault(r["frame_b"], set()).add(r["mask_path"])
    for i, f in enumerate(frames):
        img = Image.open(os.path.join(data, f))
        # B0: global Lab color histogram (8x8x8 bins)
        lab = np.asarray(img.convert("RGB").convert("LAB")).reshape(-1, 3)
        h, _ = np.histogramdd(lab, bins=(8, 8, 8), range=((0, 256),) * 3)
        hist[f] = (h / h.sum()).ravel().astype(np.float32)
        x, grid = prep(img)
        with torch.no_grad():
            o = model(pixel_values=torch.from_numpy(x)[None].to(device)).last_hidden_state[0]
        o = o.float().cpu().numpy()
        cls[f] = o[0]
        patches = o[-grid[0] * grid[1]:].reshape(grid[0], grid[1], -1)  # skip CLS/register tokens
        meanp[f] = patches.mean((0, 1))
        for mp in masks_for[f]:
            sel = mask_to_grid(os.path.join(data, mp), grid)
            maskp[(f, mp)] = patches[sel].mean(0)
        if i % 100 == 0:
            print(f"features {i}/{len(frames)}")
    np.savez(cache, cls=cls, meanp=meanp, maskp=maskp, hist=hist)
    return cls, meanp, maskp, hist


# ------------------------------------------------------------------ metrics
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def best_threshold(scores, labels):
    best_t, best_f1 = 0.0, -1.0
    for t in np.unique(scores):
        f1 = prf(scores >= t, labels)[2]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def prf(pred, labels):
    tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.data)
    split = np.array([r["split"] for r in rows])
    y = np.array([r["label"] for r in rows])
    print({s: int((split == s).sum()) for s in ["train", "val", "test"]}, "| errors:", int(y.sum()))
    cls, meanp, maskp, hist = extract(rows, args.data, args.out, args.model, device)

    scores = {
        "B0 color-hist": np.array([np.abs(hist[r["frame_a"]] - hist[r["frame_b"]]).sum() for r in rows]),
        "B1 dino-global (no training)": np.array([1 - cos(cls[r["frame_a"]], cls[r["frame_b"]]) for r in rows]),
        "R1 dino-mask (oracle mask)": np.array([1 - cos(maskp[(r["frame_a"], r["mask_path"])],
                                                        maskp[(r["frame_b"], r["mask_path"])]) for r in rows]),
    }
    # B2: logistic regression on pair features, trained on TRAIN, C tuned on VAL
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    def pair_feat(r):
        a = np.concatenate([cls[r["frame_a"]], meanp[r["frame_a"]]])
        b = np.concatenate([cls[r["frame_b"]], meanp[r["frame_b"]]])
        return np.concatenate([np.abs(a - b), a * b])
    X = np.stack([pair_feat(r) for r in rows])
    tr, va = split == "train", split == "val"
    sc = StandardScaler().fit(X[tr])
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        a = auroc(clf.predict_proba(sc.transform(X[va]))[:, 1], y[va])
        if best is None or a > best[0]:
            best = (a, C, clf)
    scores[f"B2 dino-logreg (C={best[1]})"] = best[2].predict_proba(sc.transform(X))[:, 1]

    results, te = {}, split == "test"
    for name, s in scores.items():
        t = best_threshold(s[va], y[va]) if va.any() else best_threshold(s[tr], y[tr])
        p, r, f1 = prf(s[te] >= t, y[te])
        # recall on test errors split by edit strength (delta E in Lab)
        de = np.array([row["delta_e"] for row in rows])
        bins = {"dE<40": (de < 40), "40-80": (de >= 40) & (de < 80), ">=80": (de >= 80)}
        rec_by = {k: round(float((s[te & m & (y == 1)] >= t).mean()), 3) if (te & m & (y == 1)).any() else None
                  for k, m in bins.items()}
        results[name] = {"test_F1": round(f1, 3), "precision": round(p, 3), "recall": round(r, 3),
                         "AUROC": round(auroc(s[te], y[te]), 3), "threshold": float(t),
                         "recall_by_dE": rec_by}
        print(f"{name:32s} F1={f1:.3f} P={p:.3f} R={r:.3f} AUROC={results[name]['AUROC']:.3f}  recall by dE {rec_by}")

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    with open(os.path.join(args.out, "results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\nMethod & F1 & Precision & Recall & AUROC \\\\\n\\midrule\n")
        for n, m in results.items():
            f.write(f"{n} & {m['test_F1']:.2f} & {m['precision']:.2f} & {m['recall']:.2f} & {m['AUROC']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # precision-recall curves on test
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(4.2, 3.2))
    for name, s in scores.items():
        pr, rc, _ = precision_recall_curve(y[te], s[te])
        plt.plot(rc, pr, label=name.split(" (")[0], linestyle="--" if name.startswith("R1") else "-")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=7); plt.title("Test PR curves", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "pr_curves.pdf")); plt.savefig(os.path.join(args.out, "pr_curves.png"), dpi=200)
    print("saved", os.path.join(args.out, "results.json"), "results_table.tex", "pr_curves.pdf")


if __name__ == "__main__":
    main()
