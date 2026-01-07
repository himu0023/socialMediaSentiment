import joblib 
import pandas as pd
from training.prepare_data import load_dataset
from features.tfidf_vectorizer import load_and_transform

def error_analysis():
    print("Loading tuned SVM model.")
    model = joblib.load("svm_best_model.pkl")

    print("Loading validation data....")
    val_df = load_dataset("data/twitter_validation.csv")

    X_val = load_and_transform(val_df["text"])
    y_true = val_df["sentiment"]
    y_preds = model.predict(X_val)

    # Attach Predictions
    analysis_df = val_df.copy()
    analysis_df["predicted"] = y_preds
    analysis_df["correct"] = analysis_df["sentiment"] == analysis_df["predicted"]

    # Extract misclassified samples
    errors = analysis_df[analysis_df["correct"] == False]

    print("\n------- Sample Misclassification -------\n")
    for _, row in errors.head(20).iterrows():
        print("TEXT :", row["text"])
        print("TRUE :", row["sentiment"])
        print("PRED :", row["predicted"])
        print("-"*60)

    # Save full error set for manual inspection 
    errors.to_csv("svm_error_analysis.csv", index = False)
    print("\c Saved all misclassified sample to : svm_error_analysis.csv")


if __name__ == "__main__":
    error_analysis()