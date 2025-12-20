# Hand Gesture Analysis using sEMG

## Project Overview

This project focuses on **hand gesture classification using surface Electromyography (sEMG) signals**.
Four different approaches are implemented and compared in order to evaluate their performance on the same dataset and preprocessing pipeline:

* **Traditional Machine Learning (Feature Engineering + LDA / SVM)**
* **Deep Learning Convolutional Neural Network (CNN)**
* **Vision Transformer (ViT)**
* **Graph Neural Network (GNN)**

---

## Project Structure

```
.
├── common/            # Configuration files, metrics, and preprocessing utilities
├── data/              # Raw sEMG dataset (not included in the repository)
├── dl_approaches/     # Deep Learning models (CNN, GNN, ViT) and tensor generation
├── ml_approach/       # Feature extraction and classical ML models
├── results/           # Generated plots, metrics, and evaluation outputs
├── main.py            # Main entry point of the project
└── requirements.txt   # Python dependencies
```

---

## Setup & Installation

### Install Dependencies

Make sure you are using **Python 3.x**, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## Data Preparation (Important)

⚠️ **The raw dataset is not included in this repository.**
You must manually place the data before running the project downloaded from the following link :
[https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures](`https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures`)

### Expected Dataset Structure

1. Create a folder named `data/` at the root of the project.
2. Inside `data/`, paste the zip content with one folder per subject as following:

   ```
   data/
   ├── 01/
   ├── 02/
   ├── ...
   └── 36/
   ```
3. Each subject folder must contain two `.txt` files with the recorded sEMG signals.

---

## Configuration

Model selection and hyperparameters are defined in:

```
common/config.py
```

### Model Selection

```python
# Choose the approach: "ML", "CNN", "VIT" or "GNN"
MODE = "ML"
```

### Adjustable Parameters

You can also configure:

* Number of epochs
* Batch size
* Window size
* Other training parameters

---

## How to Run

From the root directory, execute:

```bash
python main.py
```

The script will:

1. Check if preprocessed data already exists (generate it if necessary).
2. Train the selected model.
3. Evaluate performance on the validation and test sets.
4. Save all outputs in the `results/` directory.

---

## Results

After execution, the `results/` folder will contain:

* Confusion matrices (Validation & Test)
* Training curves (Accuracy and Loss)
* Metrics comparison plots (Accuracy / F1-score)

---

## Author

*Hand Gesture Analysis using sEMG – Lucas Gumuchian, Luigi Giusiano - Academic project*
