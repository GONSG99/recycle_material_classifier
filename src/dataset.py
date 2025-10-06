# -------------------------------------------------------------
# This script:
# Downloads the Trash-Type Image Dataset from Kaggle
# Keeps only the classes: paper, plastic, metal (groups others)
# Splits data into train / val / test
# Copies them into the folder structure:
#   data/images/train/paper/
#   data/images/val/plastic/
#   data/images/test/metal/
# -------------------------------------------------------------

import os, shutil, random
from pathlib import Path
import kagglehub

# ---- CLASS FILTERING ----
CLASS_KEEP = {"paper", "plastic", "metal"}      # Classes we care about
TO_OTHERS  = {"glass", "cardboard", "trash"}    # Classes to group as "others"

# ---- TARGET DIRECTORY ----
TARGET = Path("data/images") # Where to save the new dataset

# ---- DATA SPLIT RATIOS ----
SPLITS = {"train":0.7,"val":0.15,"test":0.15}

# ---- IMAGE EXTENSIONS TO LOOK FOR ----
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

# ---- SET FIXED RANDOM SEED (for reproducibility) ----
random.seed(42)

# -------------------------------------------------------------
# STEP 1: INFER CLASS FROM FILE PATH
# -------------------------------------------------------------
def infer_class(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts): # Search backwards in path parts
        if p in CLASS_KEEP:
            return p
        if p in TO_OTHERS:
            return "others"
    return "others"  # Fallback if no match found

# -------------------------------------------------------------
# STEP 2: FIND ALL IMAGES IN THE DOWNLOADED DATASET
# -------------------------------------------------------------
def gather_images(root: Path):
    imgs = []
    for fp in root.rglob("*"): # Recursively walk through subfolders
        if fp.suffix.lower() in IMG_EXTS:
            cls = infer_class(fp.parent)
            imgs.append((fp, cls))
    return imgs

# -------------------------------------------------------------
# STEP 3: SPLIT INTO TRAIN/VAL/TEST AND COPY FILES
# -------------------------------------------------------------
def split_and_copy(items):

    by_cls = {}

    # Group all images by class
    for fp, cls in items:
        by_cls.setdefault(cls, []).append(fp)

    TARGET.mkdir(parents=True, exist_ok=True)

    # For each class, randomly split its images
    for cls, files in by_cls.items():

        random.shuffle(files)
        n = len(files); n_tr = int(SPLITS["train"]*n); n_va = int(SPLITS["val"]*n)

        # Split into train, val, test
        splits = {"train":files[:n_tr], "val":files[n_tr:n_tr+n_va], "test":files[n_tr+n_va:]}

        # Copy images to correct folders
        for split, fps in splits.items():
            outdir = TARGET / split / cls
            outdir.mkdir(parents=True, exist_ok=True)
            for src in fps:
                shutil.copy(src, outdir / src.name)

        # Print summary
        print(f"{cls}: {n} -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

# -------------------------------------------------------------
# STEP 4: MAIN FUNCTION – DOWNLOAD & PREPARE DATASET
# -------------------------------------------------------------
def main():

    # Download dataset (cached locally after first time)
    root = Path(kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset"))
    print("Dataset at:", root)

    # Collect all image paths + inferred labels
    items = gather_images(root)

    if not items:
        raise SystemExit("No images found. Check dataset layout.")
    print("Classes detected:", sorted({c for _, c in items}))

    # Split and copy into new structure
    split_and_copy(items)
    print("Flattened dataset ready in:", TARGET)

# -------------------------------------------------------------
# ENTRY POINT (only runs when executed directly)
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
