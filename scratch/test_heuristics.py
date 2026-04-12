import re

def extract_features(query):
    token_count = len(query.split())
    word_count = len(re.findall(r'\w+', query))
    complexity_keywords = ["explain", "debug", "analyze", "compare", "why", "how", "diff", "refactor"]
    keyword_count = sum(1 for kw in complexity_keywords if kw in query.lower())
    entities = re.findall(r'\b[A-Z][a-z]+\b', query)
    entity_count = len(entities)
    question_count = query.count('?')
    multi_part = 1 if question_count > 1 or ';' in query or '\n' in query else 0
    tech_terms = ["function", "class", "def", "const", "api", "json", "sql", "import", "null", "undefined", "{ ", "}", "python", "javascript", "bash"]
    tech_score = sum(2 for term in tech_terms if term in query.lower())
    
    return {
        "tokens": token_count,
        "words": word_count,
        "keywords": keyword_count,
        "entities": entity_count,
        "questions": question_count,
        "multi_part": multi_part,
        "tech_score": tech_score
    }

def route_query(query):
    feats = extract_features(query)
    score = (feats['tokens'] * 0.1) + \
            (feats['keywords'] * 1.5) + \
            (feats['questions'] * 0.5) + \
            (feats['tech_score'] * 1.0) + \
            (feats['multi_part'] * 1.0)
    
    threshold = 3.0
    if score >= threshold:
        return "Strong Model (GPT-4o)"
    else:
        return "Weak Model (Llama-3-8B)"

samples = [
    "Hi there!",
    "What time is it in New York?",
    "Explain the difference between TCP and UDP in detail.",
    "Debug this Python code: def hello(): print('world')",
    "Analyze the fiscal policy of the US in 2023 vs 2024.",
    "How do I implement a binary search tree in C++?",
    "Can you compare React and Vue for a small scale application?"
]

print(f"{'Query':<60} | {'Selection':<25}")
print("-" * 90)
for q in samples:
    selection = route_query(q)
    print(f"{q[:58]:<60} | {selection:<25}")
