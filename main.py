import joblib
from preprocessing.text_cleaner import clean_text
from features.tfidf_vectorizer import load_and_transform

def predict_sentiment(text: str):
    """
    Predict sentiment with confidence using calibrated SVM.
    """

    # 1. Load calibrated model
    model = joblib.load("svm_aclibrated_model.pkl")

    # 2. Clean input text (same pipeline as training)
    cleaned_text = clean_text(text)

    # 3. Vectorize text using trained TF-IDF
    X = load_and_transform([cleaned_text])

    # 4. Predict sentiment and Probabilities 
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # 5. Map class -> probbability
    class_probs = dict(zip(model.classes_, probabilities))

    return prediction, class_probs

if __name__ == "__main__":
    print("--- Social Media Sentiment Analyzer ---\n")
    
    while True:
        text = input("Enter text (or type 'exit'): ").strip()

        if text.lower() == "exit":
            print("Exiting....")
            break

        sentiment, probs = predict_sentiment(text)

        print("\n Predicted Sentiment: ", sentiment)
        print("Confidence Scores:")
        for label, score in probs.items():
            print(f"{label}: {score:.4f}")

        print("-"*50)