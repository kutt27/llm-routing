import joblib
import os

class Router:
    def __init__(self, model_path="models/router_clf.joblib", embedder=None):
        self.model_path = model_path
        self.embedder = embedder
        self.clf = None
        
        if os.path.exists(model_path):
            self.clf = joblib.load(model_path)
    
    def is_trained(self):
        return self.clf is not None

    def train(self, X, y):
        """Standard placeholder if we want to train via the class."""
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)
        # Save after training
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.clf, self.model_path)

    def route(self, query):
        if not self.clf:
            raise ValueError("Router model not loaded. Please train the model first.")
        if not self.embedder:
            raise ValueError("Embedder not initialized.")
            
        embedding = self.embedder.encode(query)
        prediction = self.clf.predict(embedding)
        return prediction[0]
