import json
import os

def load_jsonl(filepath):
    """Loads query-label pairs from a JSONL file, skipping comments."""
    queries = []
    labels = []
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            try:
                obj = json.loads(line)
                if 'query' in obj and 'label' in obj:
                    queries.append(obj['query'])
                    labels.append(obj['label'])
            except json.JSONDecodeError:
                continue
                
    return queries, labels
