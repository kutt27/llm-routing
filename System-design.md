# LLM Routing System Design

Design decision: The code + design are inspired by the RouteLLM paper. This repo is for learning - not production. Data collection is synthetic, low quality, not tested enough.

## Approach

- Learns from preference data
- Uses Llama as judge (available via Groq)

---

# 1. System Architecture

```
Client → Router → Target Model (name only)
```

Router outputs **model name only** - not calling them.

**Judge model (Llama via Groq):**

| Role | Model |
|------|-------|
| Judge | llama-3.1-405b-instant |

**Target models (extended - any model):**

| Provider | Model |
|----------|-------|
| Llama | llama-3.1-8b-instant, llama-3.1-70b-versatile, llama-3.1-405b-instant |
| GPT | gpt-4o, gpt-4o-mini, o1-preview |
| Claude | claude-3-5-sonnet-20241022, claude-3-opus-20240229 |
| Mistral | mistral-large-latest, mistral-small-latest |
| MiniMax | MiniMax-M2.1 |
| Kimi | kimi-co |
| Qwen | qwen-plus, qwen-turbo |
| DeepSeek | deepseek-chat, deepseek-coder |

*Router learns which model fits each query - outputs name only.*

---

# 2. Router Output

```python
def route(query):
    model_name = clf.predict([query])[0]
    return model_name  # e.g., "gpt-4o" or "qwen-plus"
```

---

# 3. Data Collection Plan

## Format: JSONL (one JSON per line)

```
data.jsonl
```

```json
{"query": "Explain TCP congestion control", "label": "gpt-4o"}
{"query": "What is 2+2?", "label": "llama-3.1-8b-instant"}
{"query": "Write a python function", "label": "claude-3-5-sonnet-20241022"}
```

## Steps to Collect Data

| Step | Action |
|------|--------|
| 1 | Prepare query list (~1k queries) |
| 2 | (Optional) Get responses from target models via API |
| 3 | Use Llama judge to compare responses |
| 4 | Label = best model for that query |
| 5 | Save to data.jsonl |

## Simplified Approach (no API calls)

Since we only output model names, you can:

1. **Manual labeling** - for each query, manually assign best model
2. **Heuristic labeling** - based on query complexity:
   - Simple ("What is 2+2?") → small model
   - Complex ("Explain TCP congestion control") → large model
3. **Hybrid** - generate queries → use judge to label

## Judge Prompt (Llama)

```python
prompt = f"""Compare these answers to the query.

Query: {query}

Answer from model A: {response_a}
Answer from model B: {response_b}

Which is better? Output only the model name (e.g., gpt-4o, claude-3-5-sonnet).
If equal, output the better model: the one with lower latency/cost."""
```

## Query Sources

1. **Use existing datasets**:
   - Instruction tuning datasets
   - QA datasets (HotPopQA, NaturalQuestions)
   - Coding (Humaneval, MBPP)

2. **Generate synthetic**:
   - Use LLM to generate diverse queries

3. **Manual**:
   - Write 100-500 queries yourself

---

# 4. Router Training

## Embeddings + Classifier

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(queries)

clf = LogisticRegression()
clf.fit(X, labels)  # labels are model names
```

---

# 5. Build Steps

## Step 1: Define target models
List all models you want to route to

## Step 2: Collect queries
Get or generate ~1k queries

## Step 3: Label data
Use judge or manual labeling

## Step 4: Train router
Embeddings + classifier

## Step 5: Test
Verify routing decisions

---

# 6. Production Considerations

| Concern | Solution |
|---------|----------|
| Latency | Router < 10ms |
| Fallback | Default to mid-tier model |
| Caching | Cache query → decision |