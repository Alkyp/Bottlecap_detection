import os
import cv2
import yaml
import gdown
import shutil
import zipfile
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A

# ============================================================
# Load settings.yaml
# ============================================================

CONFIG_PATH = Path(__file__).resolve().parents[1] / "settings.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


# ============================================================
# Helper: download dataset
# ============================================================

def download_dataset():
    file_id = cfg["dataset"]["zip_file_id"]
    out_zip = cfg["dataset"]["raw_zip"]
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"[INFO] Downloading dataset from Google Drive...")
    gdown.download(url, out_zip, quiet=False)
    print(f"[OK] Downloaded to {out_zip}")

    extract_to = cfg["dataset"]["extract_to"]
    os.makedirs(extract_to, exist_ok=True)

    print(f"[INFO] Extracting ZIP...")
    with zipfile.ZipFile(out_zip, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"[OK] Extracted to folder: {extract_to}")


# ============================================================
# Relabeling function
# ============================================================

def classify_color(img, box):
    h, w = img.shape[:2]
    xc, yc, bw, bh = box

    x1 = int((xc - bw/2) * w)
    y1 = int((yc - bh/2) * h)
    x2 = int((xc + bw/2) * w)
    y2 = int((yc + bh/2) * h)

    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return "others"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_blue = np.array(cfg["relabel"]["hsv_blue_lower"])
    upper_blue = np.array(cfg["relabel"]["hsv_blue_upper"])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    ratio = np.count_nonzero(mask) / mask.size

    threshold_ratio = cfg["relabel"]["blue_ratio_threshold"]

    if ratio > threshold_ratio:
        mean_v = np.mean(hsv[:, :, 2][mask > 0])
        return "light_blue" if mean_v > cfg["relabel"]["light_blue_v_mean_threshold"] else "dark_blue"

    return "others"


# ============================================================
# Process relabeling
# ============================================================

def relabel_dataset():
    sample_dir = Path(cfg["dataset"]["source_sample"])
    out_img = Path(cfg["dataset"]["yolo_clean"]["images"])
    out_lbl = Path(cfg["dataset"]["yolo_clean"]["labels"])

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    class_map = cfg["classes"]["map"]

    processed = 0

    print("[INFO] Starting relabeling...")

    for file in sample_dir.iterdir():
        if file.suffix.lower() != ".jpg":
            continue

        img = cv2.imread(str(file))
        label_file = sample_dir / (file.stem + ".txt")

        if not label_file.exists():
            continue

        new_lines = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                _, xc, yc, bw, bh = parts
                box = [float(xc), float(yc), float(bw), float(bh)]

                cls_name = classify_color(img, box)
                cls_id = class_map[cls_name]

                new_lines.append(f"{cls_id} {xc} {yc} {bw} {bh}\n")

        shutil.copy(str(file), out_img / file.name)

        with open(out_lbl / (file.stem + ".txt"), "w") as f:
            f.writelines(new_lines)

        processed += 1
        print(f"Re-labeled: {processed} → {file.name}")

    print("[OK] Relabeling completed.")


# ============================================================
# Augmentation
# ============================================================

def augment_dataset():
    out_img = Path(cfg["dataset"]["yolo_clean"]["images"])
    out_lbl = Path(cfg["dataset"]["yolo_clean"]["labels"])

    aug_img = Path(cfg["dataset"]["yolo_aug"]["images"])
    aug_lbl = Path(cfg["dataset"]["yolo_aug"]["labels"])

    aug_img.mkdir(parents=True, exist_ok=True)
    aug_lbl.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting augmentation...")

    transform = A.Compose([
        A.HorizontalFlip(p=cfg["augmentation"]["operations"]["horizontal_flip"]),
        A.VerticalFlip(p=cfg["augmentation"]["operations"]["vertical_flip"]),
        A.Rotate(limit=cfg["augmentation"]["operations"]["rotate"], p=0.5),
        A.RandomBrightnessContrast(
            p=cfg["augmentation"]["operations"]["random_brightness_contrast"]
        ),
        A.ShiftScaleRotate(
            shift_limit=cfg["augmentation"]["operations"]["shift_scale_rotate"]["shift_limit"],
            scale_limit=cfg["augmentation"]["operations"]["shift_scale_rotate"]["scale_limit"],
            rotate_limit=cfg["augmentation"]["operations"]["shift_scale_rotate"]["rotate_limit"],
            p=cfg["augmentation"]["operations"]["shift_scale_rotate"]["p"]
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for img_file in out_img.glob('*.jpg'):
        img = cv2.imread(str(img_file))
        lbl_file = out_lbl / (img_file.stem + ".txt")

        with open(lbl_file) as f:
            bboxes = []
            class_labels = []

            for line in f:
                cls, xc, yc, bw, bh = map(float, line.strip().split())
                bboxes.append([xc, yc, bw, bh])
                class_labels.append(int(cls))

        shutil.copy(img_file, aug_img / img_file.name)
        shutil.copy(lbl_file, aug_lbl / lbl_file.name)

        for i in range(cfg["augmentation"]["count_per_image"]):
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)

            if len(transformed['bboxes']) == 0:
                continue

            out_i = aug_img / f"{img_file.stem}_aug{i}.jpg"
            out_l = aug_lbl / f"{img_file.stem}_aug{i}.txt"

            cv2.imwrite(str(out_i), transformed['image'])

            with open(out_l, "w") as f:
                for bbox, cls in zip(transformed['bboxes'], transformed['class_labels']):
                    f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    print("[OK] Augmentation completed.")


# ============================================================
# Split dataset
# ============================================================

def split_dataset():
    split_root = Path(cfg["dataset"]["split_root"])

    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (split_root / sub).mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(list(Path(cfg["dataset"]["yolo_aug"]["images"]).glob("*.jpg")))
    img_names = [p.name for p in all_imgs]

    train_names, val_names = train_test_split(
        img_names,
        test_size=cfg["dataset"]["val_ratio"],
        random_state=cfg["project"]["seed"]
    )

    for n in train_names:
        shutil.copy(
            Path(cfg["dataset"]["yolo_aug"]["images"]) / n,
            split_root / "images/train" / n
        )
        shutil.copy(
            Path(cfg["dataset"]["yolo_aug"]["labels"]) / f"{Path(n).stem}.txt",
            split_root / "labels/train" / f"{Path(n).stem}.txt"
        )

    for n in val_names:
        shutil.copy(
            Path(cfg["dataset"]["yolo_aug"]["images"]) / n,
            split_root / "images/val" / n
        )
        shutil.copy(
            Path(cfg["dataset"]["yolo_aug"]["labels"]) / f"{Path(n).stem}.txt",
            split_root / "labels/val" / f"{Path(n).stem}.txt"
        )

    print(f"[OK] Train: {len(train_names)}, Val: {len(val_names)}")


# ============================================================
# Create YOLO dataset YAML
# ============================================================

def create_dataset_yaml():
    dataset_yaml = {
        "path": cfg["dataset"]["split_root"],
        "train": "images/train",
        "val": "images/val",
        "names": list(cfg["classes"]["map"].keys())
    }

    with open("dataset_bottlecap.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)

    print("[OK] dataset_bottlecap.yaml created.")


# ============================================================
# Main pipeline
# ============================================================

def main():
    download_dataset()
    relabel_dataset()
    augment_dataset()
    split_dataset()
    create_dataset_yaml()

    print("\n[PIPELINE COMPLETED ✓]\n")


if __name__ == "__main__":
    main()
