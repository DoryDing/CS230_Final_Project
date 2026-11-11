#!/usr/bin/env bash
# One-time setup on a fresh AWS Deep Learning AMI (PyTorch) instance.
# Usage: bash aws/setup_instance.sh <s3-bucket-name>
set -euo pipefail
BUCKET="${1:?usage: bash aws/setup_instance.sh <s3-bucket-name>}"
cd "$(dirname "$0")/.."

# The DLAMI ships a ready PyTorch env; activate it if present.
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi

python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=name,memory.total --format=csv

pip install -q -r requirements.txt
sudo apt-get -qq update && sudo apt-get -qq install -y tmux ffmpeg

# Cache HF models on the big disk.
mkdir -p ~/data ~/hf_cache
grep -q HF_HOME ~/.bashrc || echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc
grep -q opt/pytorch ~/.bashrc || echo '[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate' >> ~/.bashrc

# Pull data from S3 (needs the instance IAM role with S3 access).
aws s3 sync "s3://$BUCKET/videos" ~/data/videos --only-show-errors
echo "videos: $(ls ~/data/videos | wc -l) files in ~/data/videos"

cat <<EOF

Setup done. Typical commands:
  tmux new -s run
  python data_pipeline/make_pairs.py --videos ~/data/videos --out ~/data/pairs --pairs_per_video 40
  aws s3 sync ~/data/pairs s3://$BUCKET/pairs        # back up results
Stop the instance from the EC2 console when finished.
EOF
