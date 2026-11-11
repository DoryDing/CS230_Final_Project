"""
Example-pairs figure for the report: examples side by side, from different videos/error types.
Columns: reference frame | second frame | edit mask (red) on the unedited second frame.

    python baselines/make_figure.py --data $DATA --out $DATA/../results/examples.png
"""
import argparse, csv, os, random
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--h", type=int, default=260, help="height of each tile in px")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(args.data, "manifest.csv"))))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    picks, used_vid, used_type = [], set(), set()
    for r in rows:  # errors from different videos, covering different error types first
        if r["label"] == "1" and r["video_id"] not in used_vid and r["error_type"] not in used_type:
            picks.append(r); used_vid.add(r["video_id"]); used_type.add(r["error_type"])
    for r in rows:
        if len(picks) >= args.n - 1:
            break
        if r["label"] == "1" and r["video_id"] not in used_vid:
            picks.append(r); used_vid.add(r["video_id"])
    picks = picks[:args.n - 1]
    picks.append(next(r for r in rows if r["label"] == "0" and r["video_id"] not in used_vid))

    def load(p):
        im = Image.open(os.path.join(args.data, p)).convert("RGB")
        return im.resize((round(im.width * args.h / im.height), args.h))

    tiles = []
    for r in picks:
        a, b = load(r["frame_a"]), load(r["frame_b"])
        orig_b = load(r["frame_b"].replace("_b1.jpg", "_b0.jpg"))
        m = np.asarray(Image.open(os.path.join(args.data, r["mask_path"])).convert("L").resize(b.size)) > 127
        ov = np.asarray(orig_b).copy()
        ov[m] = (0.45 * ov[m] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        label = (f"error: {r['region']} ({r['error_type']}), dE={float(r['delta_e']):.0f}"
                 if r["label"] == "1" else "consistent (no error)")
        tiles.append(([a, b, Image.fromarray(ov)], label))

    # examples side by side in one row (fits the page width); 3 tiles per example
    gap, big_gap, cap = 3, 14, 16
    widths = [sum(t.width for t in ims) + 2 * gap for ims, _ in tiles]
    W = sum(widths) + big_gap * (len(tiles) - 1)
    sheet = Image.new("RGB", (W, args.h + cap), "white")
    d = ImageDraw.Draw(sheet)
    x0 = 0
    for (ims, label), w in zip(tiles, widths):
        x = x0
        for t in ims:
            sheet.paste(t, (x, 0)); x += t.width + gap
        d.text((x0 + 2, args.h + 2), label, fill=(0, 0, 0))
        x0 += w + big_gap
    sheet.save(args.out)
    print("saved", args.out, sheet.size, [p["pair_id"] for p in picks])


if __name__ == "__main__":
    main()
