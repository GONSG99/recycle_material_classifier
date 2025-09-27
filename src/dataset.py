import os, shutil, random
from pathlib import Path
import kagglehub

CLASS_KEEP = {"paper", "plastic", "metal"}
TO_OTHERS  = {"glass", "cardboard", "trash"}

TARGET = Path("data/images")
SPLITS = {"train":0.7,"val":0.15,"test":0.15}
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")
random.seed(42)

# Infer class from path parts
def infer_class(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts):
        if p in CLASS_KEEP:
            return p
        if p in TO_OTHERS:
            return "others"
    return "others"  

# Gather all images under root
def gather_images(root: Path):
    imgs = []
    for fp in root.rglob("*"):
        if fp.suffix.lower() in IMG_EXTS:
            cls = infer_class(fp.parent)
            imgs.append((fp, cls))
    return imgs

# Split by class and copy to target
def split_and_copy(items):

    by_cls = {}
    for fp, cls in items:
        by_cls.setdefault(cls, []).append(fp)

    TARGET.mkdir(parents=True, exist_ok=True)

    for cls, files in by_cls.items():
        random.shuffle(files)
        n = len(files); n_tr = int(SPLITS["train"]*n); n_va = int(SPLITS["val"]*n)
        splits = {"train":files[:n_tr], "val":files[n_tr:n_tr+n_va], "test":files[n_tr+n_va:]}

        for split, fps in splits.items():
            outdir = TARGET / split / cls
            outdir.mkdir(parents=True, exist_ok=True)
            for src in fps:
                shutil.copy(src, outdir / src.name)

        print(f"{cls}: {n} -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

# Download the Kaggle dataset and prepare the flattened structure
def main():

    root = Path(kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset"))
    print("Dataset at:", root)

    items = gather_images(root)

    if not items:
        raise SystemExit("No images found. Check dataset layout.")
    print("Classes detected:", sorted({c for _, c in items}))

    split_and_copy(items)
    print("Flattened dataset ready in:", TARGET)

if __name__ == "__main__":
    main()
