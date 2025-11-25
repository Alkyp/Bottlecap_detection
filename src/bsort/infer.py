import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from .utils import load_config

def run_inference():
    cfg = load_config()

    model_path = f"{cfg['training']['project_name']}/{cfg['training']['run_name']}/weights/last.pt"
    source_dir = Path(cfg["inference"]["source_images"])
    conf = cfg["inference"]["conf"]
    img_size = cfg["inference"]["img_size"]
    show_n = cfg["inference"]["show_first_n"]

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)

    images = sorted(list(source_dir.glob("*.jpg")))[:show_n]

    for img_file in images:
        res = model.predict(source=str(img_file), conf=conf, imgsz=img_size)
        pred_img = res[0].plot()

        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        plt.title(img_file.name)
        plt.axis("off")
        plt.show()
if __name__ == "__main__":
    run_inference()