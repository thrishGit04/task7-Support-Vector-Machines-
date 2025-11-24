# ⭐ Task 7 — Support Vector Machines (SVM) Classification

This repository contains **Task 7** of my AIML Internship project.
The goal is to implement and analyze **Support Vector Machines (SVM)** on the **Breast Cancer Wisconsin Diagnostic dataset**, including preprocessing, training with linear & RBF kernels, hyperparameter tuning, cross-validation, visualization, and exporting all results into structured output files.

---

## 📚 Table of Contents

1. [Repository Structure](#-repository-structure)
2. [Objective](#-objective)
3. [Data Preprocessing](#-data-preprocessing-steps)
4. [Model Training Pipeline](#-model-training-svm_breast_cancerpy)
5. [Visualizations](#-generated-visualizations)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [How to Run](#-how-to-run-the-project)
8. [Dataset](#-dataset)
9. [Author](#-author)

---

## 📁 Repository Structure

This structure exactly matches your project folder:

```text
├── outputs/
│   ├── cv_scores.npy
│   ├── cv_scores_summary.txt
│   ├── decision_boundary_2d.png
│   ├── empty                         # auto-created placeholder
│   ├── grid_search_best_params.json
│   ├── scaler.pkl
│   ├── scaler_2d.pkl
│   ├── svm_2d_model.pkl
│   ├── svm_linear_model.pkl
│   ├── svm_linear_report.txt
│   ├── svm_rbf_best_model.pkl
│   ├── svm_rbf_best_report.txt
│   ├── svm_rbf_default_model.pkl
│   ├── svm_rbf_default_report.txt
│
├── data.csv                          # Original dataset (uploaded)
├── processed_breast_cancer.csv       # Cleaned & preprocessed dataset
├── svm_breast_cancer.py              # Full SVM pipeline (single executable script)
├── README.md                         # Documentation (this file)
```

---

## 🎯 Objective

This task focuses on building a complete SVM classification pipeline with:

* **Binary classification** using Support Vector Machines
* **Feature standardization** with `StandardScaler`
* **Linear vs RBF kernel comparison**
* **Hyperparameter tuning** using `GridSearchCV`
* **2D decision boundary visualization**
* **Cross-validation for robust evaluation**
* **Saving all outputs** (models, metrics, plots) in a structured folder

---

## 🧹 Data Preprocessing Steps

Steps applied to `data.csv`:

1. Load dataset and remove non-useful columns (e.g., `id`, `Unnamed: 32`).
2. Convert target `diagnosis`:

   * `M` → 1 (Malignant)
   * `B` → 0 (Benign)
3. Split dataset (80% train, 20% test, stratified).
4. Normalize features using **StandardScaler**.
5. Save cleaned dataset as `processed_breast_cancer.csv`.

---

## 🤖 Model Training (`svm_breast_cancer.py`)

The script includes an end-to-end SVM workflow:

### 1️⃣ Feature Scaling

All features are normalized for optimal SVM performance.

### 2️⃣ Linear SVM

A baseline linear classifier is trained.

Outputs:

* `svm_linear_model.pkl`
* `svm_linear_report.txt`

### 3️⃣ RBF SVM (Default Hyperparameters)

A non-linear classifier using default `C` and `gamma`.

Outputs:

* `svm_rbf_default_model.pkl`
* `svm_rbf_default_report.txt`

### 4️⃣ Hyperparameter Tuning (GridSearchCV)

Search space:

```
C = [0.1, 1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1]
kernel = ['rbf']
```

Outputs:

* `svm_rbf_best_model.pkl`
* `svm_rbf_best_report.txt`
* `grid_search_best_params.json`

### 5️⃣ Cross-Validation

Evaluates the best model using 5-fold CV.

Outputs:

* `cv_scores.npy`
* `cv_scores_summary.txt`

### 6️⃣ 2D Decision Boundary Visualization

Uses `radius_mean` and `texture_mean` to create a 2D RBF boundary plot.

Output:

* `decision_boundary_2d.png`

---

## 📊 Generated Visualizations

All stored in the `outputs/` folder:

### ✔ Decision Boundary

**File:** `decision_boundary_2d.png`
Shows RBF SVM’s non-linear separation of benign vs malignant cases.

### ✔ Text Reports

Each model has its own report file with:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## 🧪 Evaluation Metrics

Models are evaluated using:

* **Accuracy (train & test)**
* **Confusion Matrices**
* **Classification Reports**
* **Cross-validation mean & standard deviation**

These provide a full understanding of model performance and kernel behavior.

---

## 🚀 How to Run the Project

### Option 1 — Google Colab (Recommended)

1. Upload these files to Colab:

   * `data.csv`
   * `svm_breast_cancer.py`
2. Run:

```python
!python svm_breast_cancer.py
```

3. The **outputs/** folder will be generated with all files.

---

### Option 2 — Local Machine

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Run:

```bash
python svm_breast_cancer.py
```

Output files will appear inside the **outputs/** directory.

---

## 📝 Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

* **Classes:** Benign (0), Malignant (1)
* **Features:** 30 numerical measurements
* **Goal:** Predict whether a tumor is benign or malignant

This dataset is widely used to benchmark binary classification algorithms.

---

## ✨ Author

**Thrishool M S**

AIML Internship — *Task 7: Support Vector Machines (SVM) Classification*
