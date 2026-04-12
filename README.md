# LLM Routing for Agents

## The Problem

- **Strong model**: high quality, high cost
- **Weak model**: lower quality, low cost

**Goal**: Minimize cost while preserving quality.

## Router

A router maps a query to either the weak or strong model:

```
R(q) → {weak, strong}
```

## Two Components of Routing

### 1. Win Probability Model

Given a query `q`, what is the probability that the strong model outperforms the weak model?

```
P(strong wins | q)
```

This is learned from preference data:
- Human labels: strong better / weak better / tie

This becomes a binary classification or ranking problem.

### 2. Threshold Decision (α)

Once we have the probability:

- If `P(strong wins | q) ≥ α` → use strong
- Else → use weak

The threshold `α` controls the cost vs. quality tradeoff:

| α | Effect |
|---|--------|
| High | Cheaper (more weak usage) |
| Low | Higher quality (more strong usage) |

## System View

```
User Query
     ↓
Router (cheap model)
     ↓
Decision
   ↙     ↘
Weak     Strong
Model    Model
```

**Key insight**: Router must be cheap and fast — otherwise routing defeats the purpose.

## Metrics

### 1. Cost Metric
% of calls routed to the strong model.

### 2. Performance Metric
Average response quality.

### 3. PGR (Performance Gap Recovered)

How much of the quality gap between weak and strong did we recover?

| Value | Meaning |
|-------|---------|
| 0 | Same as weak model |
| 1 | Same as strong model |

### 4. APGR (Area under PGR Curve)

Measures overall efficiency of the router across all cost-quality tradeoffs.

### 5. CPT (Call Performance Threshold)

> "How much strong model usage do I need to reach X% performance?"

Practical example: need 90% GPT-4 quality → how much GPT-4 usage is required?

## Critical Observations

### Data Quality Matters

- Without proper data → routers perform ≈ random
- With domain-aligned data → large gains

> **Benchmark–dataset similarity score** — Routing quality depends on how similar training queries are to real queries.

### Two Deeper Insights

**1. Routing generalizes across models**

- Router trained on GPT-4 vs Mixtral works for Claude/LLaMA without retraining
- Router learns query complexity, not model specifics

**2. Data > Model**

- Small amount of high-quality labeled data beats large raw preference data
