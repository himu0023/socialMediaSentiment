import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer

# Global vectorizer configuration 
_vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9
)

def fit_vectorizer(texts):
    """
    Fits TF-IDF only on training data.
    Saves the vectorizer to dick.
    """

    X = _vectorizer.fit_transform(texts)
    joblib.dump(_vectorizer, "tfidf_vectorizer.pkl")
    return X

def load_and_transform(texts):
    """
    Loads saved TF_IDF vectorizer and transform new text.
    Used for validation and inference.
    """
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return vectorizer.transform(texts)