# AWS GPU setup (Spot the Goof)

One-time setup takes ~30 min of clicking plus waiting for the GPU quota approval.
Use ONE region for everything (instance, S3 bucket, quota). Our quota request was made in **us-east-1 (N. Virginia)**: use that region for the bucket and instance too.

## 0. Apply the credits
Billing and Cost Management -> Credits -> Redeem credit -> paste the promo code.
Then Billing -> Budgets -> Create budget -> "Zero spend" or a monthly cost budget (e.g. $50) with an email alert,
so a forgotten instance can't drain the credits.

## 1. Request GPU quota (do this first: approval takes hours to ~2 days)
Service Quotas -> AWS services -> Amazon EC2 -> search **"Running On-Demand G and VT instances"**
-> Request increase at account level -> **8** (vCPUs; enough for one g5.2xlarge or two g5.xlarge).
Optional: also request **"All G and VT Spot Instance Requests" = 8** (Spot is ~60-70% cheaper, but can be interrupted).
Use case text: "Stanford CS230 course project: training a vision model (DINOv2-based) and running diffusion inpainting."

## 2. S3 bucket for data
S3 -> Create bucket -> name `goof-cs230-<yourname>` -> same region -> keep "Block all public access" ON.
Upload `goof/videos/` and later `goof/data/` here (or `aws s3 sync` from the instance).

## 3. Key pair + launch the instance (after quota is approved)
EC2 -> Key pairs -> Create -> `cs230-key`, type ED25519, .pem -> it downloads; then on your Mac:
`mv ~/Downloads/cs230-key.pem ~/.ssh/ && chmod 400 ~/.ssh/cs230-key.pem`

EC2 -> Launch instance:
- Name: `goof-gpu`
- AMI: search **"Deep Learning OSS Nvidia Driver AMI GPU PyTorch"** (Ubuntu) - has CUDA, drivers, PyTorch.
- Instance type: **g5.xlarge** (A10G 24 GB; good default) or **g4dn.xlarge** (T4 16 GB; cheaper, ~Colab speed).
- Key pair: `cs230-key`
- Network: allow SSH from **My IP** only.
- Storage: **150 GB gp3** (models + frames add up).
- Advanced -> IAM instance profile: create a role with `AmazonS3FullAccess` (lets the instance read/write your bucket without keys).

## 4. Connect and set up
```
ssh -i ~/.ssh/cs230-key.pem ubuntu@<public-ip>
git clone https://github.com/DoryDing/CS230_Final_Project.git && cd CS230_Final_Project
bash aws/setup_instance.sh goof-cs230-<yourname>
```
PyCharm Professional can use the instance as a remote interpreter (Settings -> Python Interpreter -> Add -> On SSH).
Otherwise edit locally, push, and `git pull` on the instance.

Run long jobs inside `tmux` so they survive a dropped SSH connection:
`tmux new -s run` -> start job -> `Ctrl-b d` to detach -> `tmux attach -t run` to come back.

## 5. Save money
- **Stop** the instance whenever you're not using it (EC2 -> Instance state -> Stop). Stopped = only disk is billed.
- **Terminate** at the end of the quarter (also delete the EBS volume and empty the bucket).
- Approx on-demand prices (check the EC2 pricing page for your region): g4dn.xlarge ~$0.53/h, g5.xlarge ~$1.0/h.
- Partner access: IAM -> Users -> create a user for Haoran with EC2 + S3 permissions; never share root login.
