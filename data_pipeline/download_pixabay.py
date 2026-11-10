"""
Download free people videos from Pixabay for the pilot dataset.

    python data_pipeline/download_pixabay.py --key YOUR_PIXABAY_KEY --out videos --n 40

Get a free key: sign up at pixabay.com, then open https://pixabay.com/api/docs/ (key shown there).
Saves px001.mp4, px002.mp4, ... plus sources.csv (id, page URL, query) for the dataset section.
"""
import argparse, csv, os, time, requests

QUERIES = ["woman talking", "man talking", "woman walking", "man walking street",
           "couple conversation", "business meeting", "woman portrait", "man portrait",
           "people office", "woman cafe", "man phone call", "friends talking"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", default="videos")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min_s", type=int, default=8, help="minimum clip length in seconds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seen, rows = set(), []
    per_q = max(3, args.n // len(QUERIES) + 1)
    for q in QUERIES:
        r = requests.get("https://pixabay.com/api/videos/", params={
            "key": args.key, "q": q, "video_type": "film", "per_page": 50, "safesearch": "true"})
        r.raise_for_status()
        taken = 0
        for hit in r.json().get("hits", []):
            if len(rows) >= args.n or taken >= per_q:
                break
            if hit["id"] in seen or hit["duration"] < args.min_s:
                continue
            v = hit["videos"].get("medium") or hit["videos"].get("small")  # ~720p / 540p
            if not v or not v.get("url"):
                continue
            seen.add(hit["id"])
            name = f"px{len(rows) + 1:03d}.mp4"
            with requests.get(v["url"], stream=True) as d, open(os.path.join(args.out, name), "wb") as f:
                for chunk in d.iter_content(1 << 20):
                    f.write(chunk)
            rows.append({"file": name, "pixabay_id": hit["id"], "page": hit["pageURL"],
                         "query": q, "duration_s": hit["duration"]})
            taken += 1
            print(f"{name}  {hit['duration']}s  {q}")
            time.sleep(0.3)
        if len(rows) >= args.n:
            break

    with open(os.path.join(args.out, "sources.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"done: {len(rows)} videos in {args.out}/")


if __name__ == "__main__":
    main()
