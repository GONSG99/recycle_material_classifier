# -------------------------------------------------------------
# This script merges the two datasets:
#   1️⃣ Kaggle dataset (data/images)
#   2️⃣ Your own collected images (data/own_images)
#
# It combines both into one new dataset folder:
# data/combined/
#
# The structure after merging looks like:
#   data/combined/
#       ├── train/
#       │    ├── paper/
#       │    ├── plastic/
#       │    └── metal/
#       ├── val/
#       └── test/
#
# So the model can train on all available images together.
# -------------------------------------------------------------

from pathlib import Path
import shutil

# ---- Define source and destination paths ----
SRC_K = Path("data/images")        # Kaggle-prepared dataset
SRC_O = Path("data/own_images")    # Your own collected images
DST   = Path("data/combined")      # Combined output dataset

# -------------------------------------------------------------
# FUNCTION: copy_all
# -------------------------------------------------------------
def copy_all(src_dir: Path, dst_dir: Path):
    
    if not src_dir.exists():
        return 0 # Skip if source folder doesn’t exist
    
    dst_dir.mkdir(parents=True, exist_ok=True) # Make sure target exists
    n = 0

    # Copy all files from source to destination
    for p in src_dir.glob("*"):
        if p.is_file(): # Ignore subfolders
            shutil.copy(p, dst_dir / p.name)
            n += 1
    return n # Return count of copied files

# -------------------------------------------------------------
# FUNCTION: classes_in
# -------------------------------------------------------------
def classes_in(root: Path):
    out = set()
    for split in ["train", "val", "test"]:
        d = root / split
        if d.exists():
            for c in d.iterdir():
                if c.is_dir():
                    out.add(c.name)
    return out

# -------------------------------------------------------------
# STEP 1: Identify all classes across Kaggle + own datasets
# -------------------------------------------------------------
classes = sorted(classes_in(SRC_K) | classes_in(SRC_O)) # Combine both sets

if not classes:
    raise SystemExit("No classes found in data/images or data/own_images. Build them first.")

print("Classes to combine:", classes)

# -------------------------------------------------------------
# STEP 2: Create folder structure for combined dataset
# -------------------------------------------------------------
for split in ["train", "val", "test"]:
    for cls in classes:
        (DST / split / cls).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# STEP 3: Combine TRAIN images (both Kaggle + Own)
# -------------------------------------------------------------
for cls in classes:
    n1 = copy_all(SRC_K / "train" / cls, DST / "train" / cls) # Copy from Kaggle
    n2 = copy_all(SRC_O / "train" / cls, DST / "train" / cls) # Copy from own
    print(f"train/{cls}: +{n1} (kaggle) +{n2} (own)")

# -------------------------------------------------------------
# STEP 4: Combine VALIDATION and TEST images
# -------------------------------------------------------------
for split in ["val", "test"]:

    for cls in classes:

        n = copy_all(SRC_K / split / cls, DST / split / cls) # Copy from Kaggle first

        # If none, fill from own images 
        if n == 0:  
            n = copy_all(SRC_O / split / cls, DST / split / cls)
            if n > 0:
                print(f"{split}/{cls}: filled from own ({n} files)")
        else:
            print(f"{split}/{cls}: +{n} (kaggle)")

# -------------------------------------------------------------
# STEP 5: Done — print where the merged dataset is saved
# -------------------------------------------------------------
print("Combined dataset at:", DST)
