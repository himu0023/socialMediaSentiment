import joblib 
from sklearn.metrics import classification_report, confusion_matrix
from training.prepare_data import load_dataset
from features.tfidf_vectorizer import load_and_transform

def evaluate(model_path):
    print("\n==========")
    print(f"Evaluating model: {model_path}")
    print("===========")

    # 1. Load validation data (already cleaned and standardized)
    val_df = load_dataset("data/twitter_validation.csv")

    print(f"Validation samples: {len(val_df)}")
    print("Lable Distribution:")
    print(val_df["sentiment"].value_counts())

    # 2. Vectorize text using Trained tf-idf 
    X_val = load_and_transform(val_df["text"])
    y_true = val_df["sentiment"]

    # 3. Load trained model 
    model = joblib.load(model_path)

    # 4. Predict 
    y_preds = model.predict(X_val)

    # 5. Metrics
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_true, y_preds))

    print("\n Classification Report:")
    print(classification_report(y_true, y_preds, digits=4))

if __name__ == "__main__":
    # Evaluate Logistic Regression 
    evaluate("logistic_model.pkl")

    # Evaluate Linear SVM
    evaluate("svm_model.pkl")