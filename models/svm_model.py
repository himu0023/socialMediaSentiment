from sklearn.svm import LinearSVC

class SVMSentimentModel:
    def __init__(self):
        self.model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000
        )

    def train(self, X,y):
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X,y)