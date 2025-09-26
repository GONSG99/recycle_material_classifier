from pathlib import Path
import shutil

SRC_K = Path("data/images")        
SRC_O = Path("data/own_images")    
DST   = Path("data/combined")

def copy_all(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.glob("*"):
        if p.is_file():
            shutil.copy(p, dst_dir / p.name)
            n += 1
    return n

def classes_in(root: Path):
    out = set()
    for split in ["train", "val", "test"]:
        d = root / split
        if d.exists():
            for c in d.iterdir():
                if c.is_dir():
                    out.add(c.name)
    return out


classes = sorted(classes_in(SRC_K) | classes_in(SRC_O))
if not classes:
    raise SystemExit("No classes found in data/images or data/own_images. Build them first.")

print("Classes to combine:", classes)


for split in ["train", "val", "test"]:
    for cls in classes:
        (DST / split / cls).mkdir(parents=True, exist_ok=True)


for cls in classes:
    n1 = copy_all(SRC_K / "train" / cls, DST / "train" / cls)
    n2 = copy_all(SRC_O / "train" / cls, DST / "train" / cls)
    print(f"train/{cls}: +{n1} (kaggle) +{n2} (own)")


for split in ["val", "test"]:
    for cls in classes:
        n = copy_all(SRC_K / split / cls, DST / split / cls)
        if n == 0:  
            n = copy_all(SRC_O / split / cls, DST / split / cls)
            if n > 0:
                print(f"{split}/{cls}: filled from own ({n} files)")
        else:
            print(f"{split}/{cls}: +{n} (kaggle)")

print("Combined dataset at:", DST)
