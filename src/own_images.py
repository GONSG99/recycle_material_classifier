from pathlib import Path
import shutil, random

SRC = Path("data/own_images") # paper/, plastic/, metal/
DST = Path("data/own_images") # output root
SPLIT = {"train":0.70, "val":0.15, "test":0.15}
EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

random.seed(42)

def gather(p): return [f for f in p.rglob("*") if f.suffix.lower() in EXTS]

def split_copy(cls):

    files = gather(SRC/cls); random.shuffle(files)
    n=len(files); n_tr=int(n*SPLIT["train"]); n_va=int(n*SPLIT["val"])
    parts = {"train":files[:n_tr], "val":files[n_tr:n_tr+n_va], "test":files[n_tr+n_va:]}

    for split, arr in parts.items():
        out = DST/split/cls; out.mkdir(parents=True, exist_ok=True)

        for src in arr: shutil.copy(src, out/src.name)

    print(f"{cls}: {n} -> {len(parts['train'])}/{len(parts['val'])}/{len(parts['test'])}")

for cls in ["paper","plastic","metal"]:
    split_copy(cls)

print("Done", DST)
