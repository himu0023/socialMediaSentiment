import joblib 
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from training.prepare_data import load_dataset
from features.tfidf_vectorizer import load_and_transform, fit_vectorizer

C_VALUES = [0.01, 0.1, 1, 5, 10]

def tune_svm():
    print("Starting SVM tuning....")

    # 1. Load trainig data and FIT TF-TDF once 
    train_df = load_dataset("data/twitter_training.csv")
    X_train = fit_vectorizer(train_df["text"])
    y_train = train_df["sentiment"]

    # 2. Load Validation data (transform only)
    val_df = load_dataset("data/twitter_validation.csv")
    X_val = load_and_transform(val_df["text"])
    y_val = val_df["sentiment"]

    best_c = None 
    best_f1 = 0

    for c in C_VALUES:
        print(f"\n Training SVM with C={c}")
        model = LinearSVC(
            C=c,
            class_weight = "balanced",
            max_iter=5000
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        f1 = f1_score(y_val, preds, average= "macro")
        print(f"C={c} : Macro F1 = {f1:.4f}")

        if f1> best_f1:
            best_f1 = f1
            best_c = c
            joblib.dump(model, "svm_best_model.pkl")

    print("\n Tuning completed.")
    print("\n Best C:", best_c)
    print("Best Macro F1:", round(best_f1,4))

if __name__ == "__main__":
    tune_svm()