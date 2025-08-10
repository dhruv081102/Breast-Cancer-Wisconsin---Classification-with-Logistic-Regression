## 📊 Task 4: Classification with Logistic Regression

### 🧠 Objective:

Build a binary classification model using **Logistic Regression** to predict whether a tumor is **benign (0)** or **malignant (1)** based on the features from the **Breast Cancer Wisconsin dataset**.

---

### 📁 Dataset:

* **Name:** `data.csv`
* **Source:** [Kaggle - Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Shape:** `(569, 33)`
* **Target column:** `diagnosis` (B = benign, M = malignant → encoded as 0 and 1)

---

### 🛠️ Tools Used:

* Python
* VS Code
* Pandas
* Scikit-learn
* Matplotlib

---

### 🧪 Steps Performed:

#### ✅ Step 1: Load the Dataset

* Loaded the CSV file using Pandas.
* Checked basic info, shape, and column names.

#### ✅ Step 2: Preprocess the Data

* Dropped unnecessary columns like `id` and `Unnamed: 32`.
* Converted `diagnosis` (M/B) into binary format: `M = 1`, `B = 0`.

#### ✅ Step 3: Train/Test Split

* Split data into `X` (features) and `y` (target).
* Used `train_test_split()` with a test size of 20% and stratified sampling.

#### ✅ Step 4: Feature Scaling

* Standardized features using `StandardScaler()` to bring all features to a similar scale.

#### ✅ Step 5: Model Training

* Trained a `LogisticRegression` model from `sklearn`.

#### ✅ Step 6: Model Evaluation

* Evaluated model using:

  * Accuracy
  * Confusion Matrix
  * Classification Report
  * ROC AUC Score
  * ROC Curve

#### ✅ Step 7: Threshold Tuning + Sigmoid Function

* Explained the **sigmoid function** used in logistic regression.
* Showed how to adjust classification threshold (e.g., 0.3) to optimize **recall**.
* Demonstrated impact of threshold tuning on model performance.

---

### 📉 Results:

| Metric              | Value (default threshold 0.5) |
| ------------------- | ----------------------------- |
| Accuracy            | \~96%                         |
| Precision (class 1) | High                          |
| Recall (class 1)    | High                          |
| ROC AUC Score       | \~99%                         |

> ✅ Model performs very well, even without hyperparameter tuning.

---

### 📂 Folder Structure:

```
DAY4-Task4/
├── data.csv
├── logistic_regression.py
├── logistic_regression.ipynb
└──  README.md

---

### 🚀 Conclusion:

* Logistic Regression is a powerful, interpretable baseline model for binary classification tasks.
* ROC AUC score and threshold tuning are critical for optimizing cancer detection models.
* Task completed successfully with clean, modular code in both `.py` and `.ipynb` formats.

---

