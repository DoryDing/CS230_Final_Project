# Spot the Goof: Detecting Continuity Errors in AI Short Dramas
CS230 project, Dory Ding and Haoran Gao.

## Layout
- `data_pipeline/make_pairs.py`: videos -> frame pairs within a shot -> Grounding DINO + SAM masks -> synthetic color-edit errors -> `manifest.csv` (80/10/10 split by video).
- `baselines/`: DINOv2 features + baselines (B1 masked cosine threshold, B2 whole-frame classifier). Coming next.
- `milestone/`: LaTeX for the milestone (`main.tex`, `references.bib`, `neurips_2026.sty`).

## Data
Videos and generated data are NOT in git. They live in the shared Google Drive folder `goof/`:
- `goof/videos/`: source clips (filename = video_id)
- `goof/data/`: output of make_pairs.py (frames/, masks/, manifest.csv, summary.json, contact_sheet.jpg, README.md)

## Running on Colab (T4 GPU)
```
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/DoryDing/CS230_Final_Project.git
%cd CS230_Final_Project
!pip install -q scenedetect[opencv] transformers accelerate
!python data_pipeline/make_pairs.py --videos /content/drive/MyDrive/goof/videos --out /content/drive/MyDrive/goof/data
```
