import joblib
from sklearn.calibration import CalibratedClassifierCV
from training.prepare_data import load_dataset
from features.tfidf_vectorizer import load_and_transform

def calibrate_svm():
    print("Loading tuned SVM model...")
    base_model = joblib.load("svm_best_model.pkl")

    print("Loading traninig data for calibration....")
    train_df = load_dataset("data/twitter_training.csv")

    X_train = load_and_transform(train_df["text"])
    y_train = train_df["sentiment"]

    print("Calibrating SVM with Platt scaling (sigmoid)....")
    calibrate_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    calibrate_model.fit(X_train, y_train)

    joblib.dump(calibrate_model, "svm_aclibrated_model.pkl")
    print("Calibration Complete.")
    print("Saved Model: svm_calibrated_model.pkl")


if __name__ == "__main__":
    calibrate_svm()