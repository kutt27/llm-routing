**Complexity Heuristics Approach Plan**

```text
Query → Feature Extraction → Classifier → Route Decision
```

Features to extract:
1. Token count (query length)
2. Word count
3. Presence of complexity keywords ("explain", "debug", "analyze", "compare", "why", "how")
4. Named entity count (people, places, organizations)
5. Question marks / multi-part questions
6. Technical terms detection

Labeling strategy (no judge needed):
- Manual labeling for ~200-500 queries initially
- Or use proxy: if query contains "complex" keywords → label=1 (strong better)

Router options (progressive):
1. Rule-based (threshold on token count)
2. Logistic regression on features
3. Decision tree (more interpretable)

Future extension:
- Agent edits the feature array to adjust classification
- Can add new heuristics dynamically