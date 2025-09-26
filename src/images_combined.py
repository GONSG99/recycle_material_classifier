from pathlib import Path
import shutil  

SRC_K = Path("data/images")       
SRC_O = Path("data/own_images")    
DST = Path("data/combined")


for cls in ["paper","plastic","metal"]:

    for src in (SRC_K/"train"/cls).glob("*"):
        (DST/"train"/cls).mkdir(parents=True, exist_ok=True); shutil.copy(src, DST/"train"/cls/src.name)
    for src in (SRC_O/"train"/cls).glob("*"):
        (DST/"train"/cls).mkdir(parents=True, exist_ok=True); shutil.copy(src, DST/"train"/cls/src.name)

for split in ["val","test"]:
    for cls in ["paper","plastic","metal"]:
        (DST/split/cls).mkdir(parents=True, exist_ok=True)
        for src in (SRC_K/split/cls).glob("*"):
            shutil.copy(src, DST/split/cls/src.name)

print("Combined dataset at:", DST)
