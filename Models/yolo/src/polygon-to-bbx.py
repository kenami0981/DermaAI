from pathlib import Path

from config import ROOT

SPLITS = ["train", "valid", "test"]

import shutil

SRC = ROOT / "data" / "ACNESCU.v1i.yolo26"
DST = ROOT / "data" / "ACNESCU.v1i.yolo26_bbox"

shutil.copytree(SRC, DST, dirs_exist_ok=True)


def polygon_to_bbox(parts):
    cls = parts[0]
    coords = list(map(float, parts[1:]))

    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"




for split in SPLITS:

    labels_dir = DST / split / "labels"

    for txt_file in labels_dir.glob("*.txt"):

        converted = []

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # już jest bbox
                if len(parts) == 5:
                    converted.append(line)
                    continue

                converted.append(polygon_to_bbox(parts))

        with open(txt_file, "w") as f:
            f.write("\n".join(converted))

print("Conversion finished.")