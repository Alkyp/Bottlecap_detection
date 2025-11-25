# Bsort: Bottle Cap Detection & Color Classification

pipeline machine learning lengkap untuk mendeteksi tutup botol dan mengklasifikasikan warnanya (light blue, dark blue, dan other) menggunakan dataset YOLO. Proyek ini menyediakan struktur ML yang siap digunakan untuk training, inferensi, evaluasi, hingga deployment menggunakan Docker & CLI.

---

## 🚀 Fitur Utama
- **Deteksi objek (YOLO-based)** untuk tutup botol.
- **Klasifikasi warna otomatis** (light blue, dark blue, other) berdasarkan analisis HSV.
- **Python CLI (`bsort`)** untuk training & inferensi.
- **Pipeline ML terstruktur** dengan konfigurasi YAML.
- **CI/CD GitHub Actions** lengkap: linting, formatting, unit test, docker build.
- **Augmentasi + relabeling otomatis**.
- **Notebook eksperimen** untuk eksplorasi & eksperimen model.

---

## 📂 Struktur Proyek
```
bsort-project/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── settings.yaml
│
├── dataset/
│   ├── bottlecap_dataset/                
│   ├── yolo_dataset/           
│   └── yolo_dataset_aug/ 
|   └── yolo_dataset_split/        
│
├── tools/
│   └── relabel_and_augment.py
│
├── src/bsort/
│   ├── cli.py
│   ├── train.py
│   ├── infer.py
│   └── utils.py
│
├── tests/
│   └── test_utils.py
│
├── notebooks/
│   └── 01_experiments.ipynb
│
└── .github/workflows/ci.yml
```

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

---

## 🧪 Unit Test
Test disimpan di folder `tests/`.
Menjalankan test:
```
pytest -q
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

"To try out the code, open the Jupyter Notebook located in the notebook/ folder. The notebook contains the complete workflow from downloading the dataset, training the model, to testing the results."
