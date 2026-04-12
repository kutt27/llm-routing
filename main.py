import sys
from src.embedder import Embedder
from src.classifier import Router

def main():
    # Initialize components
    embedder = Embedder()
    router = Router(embedder=embedder)
    
    if not router.is_trained():
        print("Error: Router model not found in models/router_clf.joblib.")
        print("Please run 'python train_router.py' first.")
        sys.exit(1)
    
    print("--- LLM Router (Type 'exit' to quit) ---")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        try:
            model = router.route(query)
            print(f"Optimal Model: {model}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
