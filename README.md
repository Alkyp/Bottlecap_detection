# Bsort: Bottle Cap Detection & Color Classification

pipeline machine learning lengkap untuk mendeteksi tutup botol dan mengklasifikasikan warnanya (light blue, dark blue, dan other) menggunakan dataset YOLO.

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
