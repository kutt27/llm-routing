from src.data_loader import load_jsonl
from src.embedder import Embedder
from src.classifier import Router
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    print("--- LLM Router Training ---")
    
    # 1. Load data
    data_file = 'data/data.jsonl'
    print(f"Loading data from {data_file}...")
    queries, labels = load_jsonl(data_file)
    
    # 2. Embed
    print("Initializing embedder...")
    embedder = Embedder()
    print("Generating embeddings (this may take a while)...")
    X = embedder.encode(queries, show_progress=True)
    y = labels
    
    # 3. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training classifier...")
    router = Router(embedder=embedder)
    router.train(X_train, y_train)
    
    # 4. Eval
    # We use the internal clf for eval
    y_pred = router.clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Complete! Accuracy: {accuracy:.4f}")
    print("Model saved to models/router_clf.joblib")

if __name__ == "__main__":
    main()
