import os
import subprocess
import sys

# ── 1. Install dependencies ───────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "wandb", "sacrebleu", "seaborn", "gdown", "datasets", "spacy"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

# ── 2. W&B login ─────────────────────────────────────────────────────────────
import wandb
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
wandb.login(key=secrets.get_secret("WANDB_API_KEY"))
print("W&B login successful")

# ── 3. Clone repo ─────────────────────────────────────────────────────────────
token = secrets.get_secret("GITHUB_TOKEN")
clone_url = f"https://{token}@github.com/Anurag9Dhiman/da6401_assignment_3.git"
subprocess.run(["git", "clone", clone_url, "/kaggle/working/project"], check=True)
os.chdir("/kaggle/working/project")
print("Repo cloned. Files:", os.listdir("."))

# ── 4. Verify GPU ─────────────────────────────────────────────────────────────
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 5. Train ──────────────────────────────────────────────────────────────────
from train import run_training_experiment
run_training_experiment()

# ── 6. Checkpoint info ────────────────────────────────────────────────────────
ckpt = "/kaggle/working/project/best_checkpoint.pt"
print(f"\nCheckpoint size : {os.path.getsize(ckpt) / 1e6:.1f} MB")
print("Download best_checkpoint.pt from Kaggle Output tab,")
print("upload to Google Drive, copy the file ID into model.py _GDRIVE_FILE_ID")
