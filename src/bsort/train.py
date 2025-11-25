from ultralytics import YOLO
from .utils import load_config

def run_training():
    cfg = load_config()

    model_name = cfg["training"]["model"]
    epochs = cfg["training"]["epochs"]
    img_size = cfg["training"]["img_size"]
    batch_size = cfg["training"]["batch_size"]
    project = cfg["training"]["project_name"]
    run_name = cfg["training"]["run_name"]

    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)

    print("[INFO] Starting YOLO training...")
    model.train(
        data="dataset_bottlecap.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project,
        name=run_name,
    )

    print("[OK] Training completed.")
if __name__ == "__main__":
    run_training()
