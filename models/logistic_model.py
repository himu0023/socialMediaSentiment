from sklearn.linear_model import LogisticRegression

class LogisticSentimentModel:
    def __init__(self):
        self.model = LogisticRegression(
            max_iter = 2000,
            class_weight="balanced",
            n_jobs=-1
        )

    def train(self, X, y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X,y):
        return self.model.score(X,y)