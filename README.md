# Social Media Sentiment Analyzer

An end-to-end **machine learning–based sentiment analysis system** for social media text.  
The project follows a **clean ML pipeline**: data inspection → preprocessing → feature engineering → model comparison → hyperparameter tuning → error analysis → probability calibration → inference.

This is **not a notebook-only project**; it is structured like a real ML codebase.

---

## 📌 Problem Statement

Given a social media post (tweet / comment), predict its **sentiment**  
(`positive`, `negative`, or `neutral`) and provide **confidence scores**.

---

## 📂 Dataset

- Source: Kaggle (Twitter / Social Media Sentiment Dataset)
- Files used:
  - `twitter_training.csv`
  - `twitter_validation.csv`

## 🧠 Approach & Methodology

### 1️⃣ Data Inspection
- Understand raw data structure
- Check label distribution
- Identify missing or malformed rows
File: training/inspect_data.py


### 2️⃣ Data Preprocessing
- Lowercasing
- URL removal
- Mention removal
- Emoji conversion to text
- Text normalization
- Label standardization
File:training/prepare_data.py
     preprocessing/text_cleaner.py


### 3️⃣ Feature Engineering
- TF-IDF Vectorization
- Unigrams + Bigrams
- Vocabulary learned **only from training data**

File: features/tfidf_vectorizer.py


### 5️⃣ Model Comparison
Both models were evaluated on the same validation set.

**Result:**  
Linear SVM significantly outperformed Logistic Regression in macro F1 score.

---

### 6️⃣ Hyperparameter Tuning
- Tuned **C (regularization strength)** for Linear SVM
- Metric optimized: **Macro F1**
- Best result: Best C = 10
               Best Macro F1 ≈ 0.9816
File: training/tune_svm.py

### 7️⃣ Error Analysis
- Analyzed misclassified samples
- Identified patterns:
  - Sarcasm
  - Ambiguous short texts
  - Gaming slang
  - Context-dependent language

File: training/error_analysis.py


### 8️⃣ Probability Calibration
- Linear SVM does not provide probabilities by default
- Applied **Platt scaling** using `CalibratedClassifierCV`
- Enabled confidence scores for predictions

File:training/calibrate_svm.py


### 9️⃣ Inference
Interactive CLI inference using the **calibrated SVM model**.
File: main.py
