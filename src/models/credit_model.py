class CreditModel:
    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        return self.model.predict(X)