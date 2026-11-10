"""
Spot the Goof - pilot data pipeline (milestone version).

videos/  ->  shot split  ->  frame pairs within a shot  ->  Grounding DINO + SAM masks
         ->  synthetic continuity errors (color edits inside the mask)
         ->  manifest.csv (split 80/10/10 by source video) + contact sheet for spot-checking

Colab setup (Runtime > Change runtime type > T4 GPU):
    !pip install -q scenedetect[opencv] transformers accelerate
    !python make_pairs.py --videos /content/drive/MyDrive/goof/videos --out /content/drive/MyDrive/goof/data

Every source pair yields TWO examples sharing frame_a:
    label 0 (consistent): frame_b passed through the same blend/re-encode with a zero edit
    label 1 (error):      frame_b with the masked region recolored
Both go through identical processing so the model cannot win by spotting re-encoding.
"""
import argparse, csv, json, os, random
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Grounding DINO text prompt -> error_type category
REGIONS = {
    "shirt": "clothing", "jacket": "clothing", "dress": "clothing", "sweater": "clothing",
    "coat": "clothing", "pants": "clothing", "skirt": "clothing",
    "hair": "hair",
    "hat": "accessory", "glasses": "accessory", "bag": "accessory", "scarf": "accessory",
    "necklace": "accessory", "tie": "accessory",
}


# ---------------------------------------------------------------- shots & frames
def detect_shots(path, fps):
    """Return [(start_frame, end_frame)] using PySceneDetect (CPU). Falls back to one shot."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(path, ContentDetector(threshold=27.0))
        shots = [(s.get_frames(), e.get_frames()) for s, e in scenes]
        if shots:
            return shots
    except Exception as e:  # scenedetect missing or failed
        print(f"  shot detection failed ({e}); treating video as one shot")
    n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    return [(0, n)]


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def sample_pairs(path, n_pairs, gap_s, rng):
    """Sample (frame_a, frame_b) 2-5 s apart inside the same shot."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    shots = [s for s in detect_shots(path, fps) if (s[1] - s[0]) / fps >= gap_s[0] + 0.5]
    pairs = []
    for _ in range(n_pairs * 3):
        if not shots or len(pairs) >= n_pairs:
            break
        s, e = rng.choice(shots)
        gap = int(rng.uniform(*gap_s) * fps)
        if e - s <= gap + 1:
            gap = e - s - 2
        a = rng.randint(s, e - gap - 1)
        fa, fb = read_frame(cap, a), read_frame(cap, a + gap)
        if fa is not None and fb is not None:
            pairs.append((fa, fb))
    cap.release()
    return pairs


# ---------------------------------------------------------------- masks
class Segmenter:
    def __init__(self, device):
        import torch
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                                  SamModel, SamProcessor)
        self.torch, self.device = torch, device
        gid = "IDEA-Research/grounding-dino-tiny"
        self.gproc = AutoProcessor.from_pretrained(gid)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gid).to(device).eval()
        sid = "facebook/sam-vit-base"
        self.sproc = SamProcessor.from_pretrained(sid)
        self.sam = SamModel.from_pretrained(sid).to(device).eval()
        self.prompt = ". ".join(REGIONS) + "."

    def __call__(self, img, box_thr=0.35, min_area=0.01, max_area=0.5):
        """Return list of (region_name, bool mask HxW) for confident, reasonably sized regions."""
        torch = self.torch
        pil = Image.fromarray(img)
        H, W = img.shape[:2]
        inp = self.gproc(images=pil, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gdino(**inp)
        res = self.gproc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=box_thr, text_threshold=0.25, target_sizes=[(H, W)])[0]
        labels = res.get("text_labels", res.get("labels"))
        boxes, names = [], []
        for box, lab in zip(res["boxes"].tolist(), labels):
            name = next((r for r in REGIONS if r in str(lab)), None)
            if name:
                boxes.append(box); names.append(name)
        if not boxes:
            return []
        sin = self.sproc(pil, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sout = self.sam(**sin, multimask_output=False)
        masks = self.sproc.image_processor.post_process_masks(
            sout.pred_masks.cpu(), sin["original_sizes"].cpu(), sin["reshaped_input_sizes"].cpu())[0]
        found = []
        for name, m in zip(names, masks[:, 0].numpy().astype(bool)):
            frac = m.mean()
            if min_area <= frac <= max_area:
                found.append((name, m))
        return found


# ---------------------------------------------------------------- edits
def recolor(img, mask, rng, min_delta_e=20.0):
    """Recolor the masked region. Returns (edited, delta_e) or (None, delta_e) if too subtle."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=2.0)[..., None]  # feathered edge
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1][mask].mean()
    new = hsv.copy()
    if sat < 40:  # black/white/gray garment: hue shift is invisible, so tint it instead
        new[..., 0] = rng.uniform(0, 180)
        new[..., 1] = np.clip(new[..., 1] + rng.uniform(90, 160), 0, 255)
        new[..., 2] = np.clip(new[..., 2] * rng.uniform(0.9, 1.3) + 25, 0, 255)
    else:
        new[..., 0] = (new[..., 0] + rng.uniform(40, 140)) % 180  # 80-280 deg hue rotation
    edited_full = cv2.cvtColor(new.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = (soft * edited_full + (1 - soft) * img.astype(np.float32)).astype(np.uint8)
    lab0 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)[mask]
    lab1 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)[mask]
    de = float(np.linalg.norm(lab0.mean(0) - lab1.mean(0)))
    return (out if de >= min_delta_e else None), de


def null_edit(img, mask):
    """Same blending path with zero change, so consistent and error frames share processing."""
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=2.0)[..., None]
    return (soft * img.astype(np.float32) + (1 - soft) * img.astype(np.float32)).astype(np.uint8)


# ---------------------------------------------------------------- helpers
def resize_short(img, short=448):
    h, w = img.shape[:2]
    s = short / min(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def save_jpg(img, path):
    Image.fromarray(img).save(path, quality=92)


def contact_sheet(rows, out_path, n=24, thumb=200):
    rows = rows[:n]
    if not rows:
        return
    sheet = Image.new("RGB", (thumb * 3, thumb * len(rows)), "white")
    for i, (a, b, m, title) in enumerate(rows):
        overlay = b.copy()
        overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        for j, im in enumerate([a, b, overlay]):
            t = Image.fromarray(im); t.thumbnail((thumb, thumb))
            sheet.paste(t, (j * thumb, i * thumb))
        ImageDraw.Draw(sheet).text((4, i * thumb + 4), title, fill=(255, 255, 0))
    sheet.save(out_path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs_per_video", type=int, default=20)
    ap.add_argument("--gap", type=float, nargs=2, default=[2.0, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    for d in ["frames", "masks"]:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    seg = Segmenter(device)

    vids = sorted(f for f in os.listdir(args.videos) if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")))
    vid_ids = [os.path.splitext(v)[0] for v in vids]
    shuffled = vid_ids[:]; rng.shuffle(shuffled)
    n = len(shuffled); n_tr, n_va = round(0.8 * n), round(0.1 * n)
    split_of = {v: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                for i, v in enumerate(shuffled)}

    rows, sheet_rows, stats = [], [], {"pairs_sampled": 0, "no_region": 0, "too_subtle": 0}
    for vf, vid in zip(vids, vid_ids):
        pairs = sample_pairs(os.path.join(args.videos, vf), args.pairs_per_video, args.gap, rng)
        stats["pairs_sampled"] += len(pairs)
        print(f"{vid}: {len(pairs)} pairs ({split_of[vid]})")
        for k, (fa, fb) in enumerate(pairs):
            fa, fb = resize_short(fa), resize_short(fb)
            regions = seg(fb)
            if not regions:
                stats["no_region"] += 1
                continue
            name, mask = regions[rng.randrange(len(regions))]
            edited, de = recolor(fb, mask, nrng)
            if edited is None:
                stats["too_subtle"] += 1
                continue
            pid = f"{vid}_{k:03d}"
            pa = f"frames/{pid}_a.jpg"
            pb0, pb1, pm = f"frames/{pid}_b0.jpg", f"frames/{pid}_b1.jpg", f"masks/{pid}.png"
            save_jpg(fa, os.path.join(args.out, pa))
            save_jpg(null_edit(fb, mask), os.path.join(args.out, pb0))
            save_jpg(edited, os.path.join(args.out, pb1))
            Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(args.out, pm))
            common = dict(video_id=vid, split=split_of[vid], frame_a=pa, region=name, mask_path=pm)
            rows.append(dict(pair_id=pid + "_c", frame_b=pb0, label=0, error_type="none",
                             delta_e=0.0, **common))
            rows.append(dict(pair_id=pid + "_e", frame_b=pb1, label=1, error_type=REGIONS[name],
                             delta_e=round(de, 1), **common))
            if len(sheet_rows) < 24:
                sheet_rows.append((fa, edited, mask, f"{pid} {name} dE={de:.0f}"))

    fields = ["pair_id", "video_id", "split", "frame_a", "frame_b", "label", "error_type",
              "region", "mask_path", "delta_e"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    contact_sheet(sheet_rows, os.path.join(args.out, "contact_sheet.jpg"))

    summary = {"videos": len(vids), "examples": len(rows), **stats,
               "per_split": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
               "per_error_type": {t: sum(r["error_type"] == t for r in rows)
                                  for t in ["clothing", "hair", "accessory"]}}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(
            "# Pilot pairs (Spot the Goof)\n"
            "- manifest.csv: one row per example. frame_a = reference, frame_b = second frame.\n"
            "- label 1 = synthetic continuity error (masked region recolored), 0 = consistent.\n"
            "- Each source pair appears twice (_c and _e) with the same frame_a; both b-frames go\n"
            "  through identical blend + JPEG processing. mask_path = edited region (use for IoU).\n"
            "- split is by video_id (80/10/10): never mix videos across splits.\n"
            "- delta_e = mean Lab color change in the mask (edit strength). See summary.json.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
