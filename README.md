# Bottle Cap Detection & Color Classification

**Status:** Work in Progress

## Project Overview
This project aims to build a real-time computer vision system for detecting bottle caps and classifying their colors. The project is still under development, and some parts are incomplete.

### What’s included so far:
- Initial dataset preparation and labeling
- Preliminary object detection model setup
- Some experiments with YOLO-based model inference

### Note
This is submitted as part of an application task. While the project is not fully completed, it demonstrates my understanding of:
- Dataset handling and preprocessing
- Object detection model setup
- Approach to color classification

---

## 🚀 Key Features
- **Object detection (YOLO-based)** for bottle caps
- **Automatic color classification** (light blue, dark blue, other) based on HSV analysis
- **Augmentation + automatic relabeling**
- **Experimentation notebook** for exploration & model experiments

---

## 🧰 Configuration (YAML)
Example `settings.yaml`:
```yaml
model:
  name: yolov8n
  epochs: 50
  img_size: 320

data:
  train: dataset/relabeled/images/train
  val: dataset/relabeled/images/val
  nc: 3
  names: ["light_blue", "dark_blue", "other"]

```


---

## 📘 Experimentation Notebook
Notebook available at::
```
notebook/model_development_and_experimentation.ipynb
```
It contains:
- Bounding box visualization
- Model experiments

---

## 📊 Dataset
The initial dataset is in YOLO format. Relabeling was performed based on:
- HSV range for **light blue**
- HSV range for **dark blue**
- Everything else → labeled as **other**

To try out the code, open the Jupyter Notebook located in the notebook/ folder. The notebook contains the complete workflow from downloading the dataset, training the model, to testing the results.
