# -------------------------------------------------------------
# This script prepares your OWN collected images
# (separate from the Kaggle dataset) by splitting them into:
# train / val / test folders
#
# Example output structure after running this:
# data/own_images/
# ├── train/
# │    ├── paper/
# │    ├── plastic/
# │    └── metal/
# ├── val/
# └── test/
# -------------------------------------------------------------

from pathlib import Path
import shutil, random

# ---- SOURCE & DESTINATION FOLDERS ----
SRC = Path("data/own_images")  # Folder where your raw images are currently stored
DST = Path("data/own_images")  # Output folder (same location, will create train/val/test inside)

# ---- SPLIT RATIOS ----
SPLIT = {"train":0.70, "val":0.15, "test":0.15}

# ---- SUPPORTED IMAGE EXTENSIONS ----
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

# ---- FIX RANDOMNESS (for reproducible splits) ----
random.seed(42)

# -------------------------------------------------------------
# FUNCTION: gather()
# -------------------------------------------------------------
def gather(p): return [f for f in p.rglob("*") if f.suffix.lower() in EXTS]

# -------------------------------------------------------------
# FUNCTION: split_copy()
# -------------------------------------------------------------
def split_copy(cls):

    # 1️⃣ Gather all image paths for this class
    files = gather(SRC/cls); random.shuffle(files) # Shuffle before splitting for randomness
   
    # 2️⃣ Compute how many go into each split
    n=len(files); n_tr=int(n*SPLIT["train"]); n_va=int(n*SPLIT["val"])
    
    # Slice list into 3 parts
    parts = {"train":files[:n_tr], "val":files[n_tr:n_tr+n_va], "test":files[n_tr+n_va:]}

    # 3️⃣ Copy files into their corresponding split folders
    for split, arr in parts.items():
        out = DST/split/cls; out.mkdir(parents=True, exist_ok=True) # Create directory if missing
        for src in arr: shutil.copy(src, out/src.name) # Copy image file

    # 4️⃣ Print summary for this class
    print(f"{cls}: {n} -> {len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")

# -------------------------------------------------------------
# MAIN EXECUTION: process all 3 recyclable material classes
# -------------------------------------------------------------
for cls in ["paper","plastic","metal"]:
    split_copy(cls)

print("Done", DST)
