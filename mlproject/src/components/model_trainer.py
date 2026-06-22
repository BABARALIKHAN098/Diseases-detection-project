from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model
