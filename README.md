#  Bottle Cap Detection & Color Classification

Status: Work in Progress

## This project is an attempt to build a real-time computer vision system for detecting bottle caps and classifying their colors. The project is still under development, and some parts are incomplete.

What’s included so far:

- Initial dataset preparation and labeling

- Preliminary model setup for object detection

- Some experiments with YOLO-based model inference

##  Note

- I am submitting this as part of the application task. While the project is not fully completed, I have tried to demonstrate my understanding of:

- Dataset handling and preprocessing

- Object detection model setup

- Approach to color classification

---

## 🚀 Fitur Utama
- **Deteksi objek (YOLO-based)** untuk tutup botol.
- **Klasifikasi warna otomatis** (light blue, dark blue, other) berdasarkan analisis HSV.
- **Augmentasi + relabeling otomatis**.
- **Notebook eksperimen** untuk eksplorasi & eksperimen model.

---


---



## 🧰 Konfigurasi (YAML)
Contoh `settings.yaml`:
```
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

## 📘 Notebook Eksperimen
Notebook tersedia di:
```
notebook/model_development_and_experimentation.ipynb
```
Berisi:
- Visualisasi bounding box
- Eksperimen model

---

## 📊 Dataset
Dataset awal berformat YOLO. Relabel dilakukan berdasarkan:
- Rentang HSV warna **light blue**
- Rentang HSV warna **dark blue**
- Selain itu → label **other**

To try out the code, open the Jupyter Notebook located in the notebook/ folder. The notebook contains the complete workflow from downloading the dataset, training the model, to testing the results.
