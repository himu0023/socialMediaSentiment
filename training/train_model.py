import joblib 
from training.prepare_data import load_dataset
from features.tfidf_vectorizer import fit_vectorizer
from models.logistic_model import LogisticSentimentModel
from models.svm_model import SVMSentimentModel

def train(model_type = "logistic"):
    train_df = load_dataset("data/twitter_training.csv")

    X_train = fit_vectorizer(train_df["text"])
    y_train = train_df["sentiment"]

    if model_type == "logistic":
        model = LogisticSentimentModel()
        model_name = "logistic_model.pkl"
    elif model_type == "svm":
        model = SVMSentimentModel()
        model_name = "svm_model.pkl"
    else:
        raise ValueError("Unknow model type")
    
    model.train(X_train, y_train)
    joblib.dump(model, model_name)


    print(f"{model_type.upper()} model trained on saved as {model_name}")


if __name__ == "__main__":
    train("logistic")
    train("svm")