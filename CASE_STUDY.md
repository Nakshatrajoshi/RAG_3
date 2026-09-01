# Project 3: Agentic RAG System (Hybrid Search + Memory + Tools)

## 📋 Problem Statement

**What problem does this solve?**

Projects 1 and 2 could only do one thing: retrieve documents and answer questions about them. Real-world assistants need more:
- Answer general questions (math, current date) without needing documents
- Remember conversation context across multiple turns ("what about that?")
- Combine multiple retrieval strategies for better accuracy
- Extend their own knowledge base on demand (add a new source without restarting)

**Real-world use case:** An internal team assistant that can answer from company docs, do quick calculations, tell you the date, and let you add new reference material (a wiki page, a doc) on the fly — all through one conversational interface.

---

## 🏗️ Solution Architecture

```
User Question
    ↓
[Conversation Memory] ← rewrites follow-ups into standalone questions using chat history
    ↓
[Router] ← rule-based classifier: calculator / datetime / retrieve?
    ↓
    ├─→ [Calculator Tool] → direct math evaluation
    ├─→ [Datetime Tool] → current date/time
    └─→ [Hybrid Retriever]
            ├─ BM25 (keyword search)
            ├─ Semantic search (FAISS + embeddings)
            └─ Reciprocal Rank Fusion (RRF) → merges both rankings
    ↓
[LLM] ← generates answer from context (via OpenRouter)
    ↓
Answer (+ saved to memory for next turn)
```

**Key architectural decisions and why:**

| Decision | Choice | Why |
|---|---|---|
| Routing | Rule-based (regex/keywords) | Fast, free, predictable — no extra LLM call needed for simple classification |
| Retrieval | Hybrid (BM25 + Semantic + RRF) | Semantic search misses exact terms (codes, names); BM25 misses paraphrased meaning. Combined covers both. |
| Memory | Query rewriting (LLM-based) | Handles natural follow-ups ("what about that?") by resolving references before retrieval, rather than just appending raw history to every prompt |
| Web loading | On-demand, extends existing index | Lets the knowledge base grow without restarting the app |
| LLM Provider | OpenRouter (nvidia/nemotron) | Access to varied models through one API; free tier for experimentation |

---

## 💻 Implementation Details

### 1. Router (Rule-Based Classification)

```python
def route_query(query: str) -> str:
    # Checks datetime keywords, then math symbols/keywords, else retrieve
```

Deliberately kept rule-based rather than LLM-based for this version — cheaper, faster, and the logic is fully inspectable/debuggable (important when something misroutes, as it did — see Optimizations below).

### 2. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

```python
def hybrid_search(self, query, top_k=5, rrf_k=60):
    semantic_ranked = self._semantic_search(query, k=top_k*2)
    bm25_ranked = self._bm25_search(query, k=top_k*2)
    
    # RRF: score = sum of 1/(rrf_k + rank) across both rankings
    rrf_scores = {}
    for rank, idx in enumerate(semantic_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rrf_k + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1/(rrf_k + rank + 1)
    
    # sort by combined score, return top_k
```

**Why RRF over a simple weighted average?** BM25 scores and cosine similarity scores are on completely different scales (BM25 is unbounded, cosine similarity is 0-1). Averaging raw scores would let one method dominate arbitrarily. RRF instead combines *rankings* (position 1st, 2nd, 3rd...), which is scale-independent and is the standard approach used in production hybrid search systems.

### 3. Conversation Memory via Query Rewriting

```python
def rewrite_query(self, question: str) -> str:
    if not self.history:
        return question  # first question - nothing to rewrite
    # LLM rewrites follow-up into standalone question using chat history
```

Example from actual testing:
```
Turn 1: "what are the models we have?"
        → answered about Kimi K3, K2.7

Turn 2: "let me the capability of Kimi K3"  (typo, fragment)
        → Rewritten to: "What are the capabilities of Kimi K3, the 
           3-trillion-scale open-source model with 2.8 trillion 
           parameters, built on Kimi Delta Attention (KDA)..."
        → correctly resolved despite the typo and missing context
```

### 4. Dynamic Web Loading

```python
def load_website(self, url: str):
    # Fetch → check content length → chunk → merge into existing FAISS + BM25 indexes
```

Adds new content into the *same* index used for the original PDFs, so retrieval works across everything uniformly.

---

## 📊 Evaluation Results

**Router Accuracy:** Tested on 7 representative queries covering all three routes (calculator, datetime, retrieve), including a regression test for a bug found during development.

**End-to-End Pipeline:** Tested full flow across calculator, datetime, document retrieval, and memory-dependent follow-up questions.

**Manual Testing (Real Sessions):**
- ✅ Loaded 4 PDFs (170 pages → 812 chunks) successfully
- ✅ Calculator: "What is 25 + 17?" → correctly answered 42
- ✅ Dynamic web loading: successfully loaded and queried external documentation (Kimi K3 docs)
- ✅ Memory rewriting: correctly resolved fragmented/typo'd follow-ups using prior conversation turns
- ✅ Correctly answered detailed multi-part questions pulling from newly-loaded web content mixed with existing PDF knowledge

*(See `evaluation-results.json` for full logged test results)*

---

## 🔧 Optimizations & Bugs Found (Real Debugging Log)

This section is the most valuable part — real problems found and fixed during development, not hypothetical ones.

### Bug 1: Router False Positive on Product Names

**What happened:** The query "What is Kimi K3?" was incorrectly routed to the calculator tool.

**Root cause:** The router's calculator rule checked for the phrase `"what is"` (too generic — matches most knowledge questions) combined with `has_numbers` (which matched the digit in "K3").

**Fix:** Removed `"what is"` from calculator trigger keywords; now requires either an explicit math symbol (`\d+\s*[+\-*/]\s*\d+`) or an unambiguous calculation phrase (`"calculate"`, `"plus"`, `"multiplied by"`).

**Lesson:** Keyword-based routing is fast but fragile — any keyword broad enough to catch real intent will also catch false positives. This is exactly the kind of tradeoff that pushes production systems toward LLM-based or hybrid routing at scale.

### Bug 2: Empty Memory Rewrites

**What happened:** Occasionally the query-rewriting LLM call would return an empty string, which then got sent to the router/retriever as an empty question, producing a confused "I don't see a question" response.

**Fix:** Added a fallback in `ConversationMemory.rewrite_query()` — if the rewritten result is empty or suspiciously short (<5 chars), fall back to the original question instead of using the broken rewrite.

**Lesson:** LLM outputs need defensive validation even in "simple" internal pipeline steps, not just user-facing generation.

### Bug 3: Web Scraping Failures on JS-Heavy / Bot-Protected Sites

**What happened:** Attempting to load Amazon and LinkedIn pages either returned near-empty content (1 chunk, ~87 characters — an error/blocked page) or, in one case, technically "succeeded" (2513 characters from GeeksforGeeks' homepage) but the content was mostly navigation/boilerplate, not useful article text — leading to irrelevant answers when queried.

**Root cause:** `WebBaseLoader` fetches raw HTML only. Sites that render content client-side via JavaScript (most e-commerce, social media, SPA-based sites) return an empty shell to a basic HTTP fetch. Separately, sites like Amazon/LinkedIn have active bot-detection that blocks non-browser requests entirely — this is a deliberate, ToS-enforced restriction, not just a technical limitation to work around.

**Fix:** 
1. Added a content-length safety check (`< 500 chars` → reject and warn)
2. Added an explicit blocklist for known JS-heavy/bot-protected domains, with a clear message explaining *why* rather than a silent failure

**Lesson:** Not all web content is fair game for automated retrieval — some sites explicitly prohibit scraping in their ToS and enforce it technically. A production system needs to distinguish "technically can't fetch this" from "shouldn't attempt to fetch this," and fail clearly rather than silently ingesting garbage or attempting to circumvent protections. This is a real constraint to design around, not a bug to eliminate.

---

## 🎯 Comparison: Project 1 vs 2 vs 3

| Aspect | Project 1 | Project 2 | Project 3 |
|---|---|---|---|
| Retrieval | Semantic only (FAISS) | Semantic only (ChromaDB) | **Hybrid (BM25 + Semantic + RRF)** |
| Documents | Single text file | Multiple PDFs | Multiple PDFs + dynamic web sources |
| Memory | None | None | **Conversation memory + query rewriting** |
| Tools | None | None | **Calculator, datetime, web loader** |
| Routing | N/A | N/A | **Rule-based agent router** |
| Knowledge base | Fixed at build time | Fixed at build time | **Extensible at runtime** |

Project 3 demonstrates the clearest production-thinking progression: from "answer questions about fixed documents" to "an assistant that decides how to answer and can grow its own knowledge."

---

## 📚 Key Learnings

1. **RRF fusion is worth the complexity** — combining rankings (not raw scores) correctly handles the scale mismatch between BM25 and cosine similarity, and is straightforward to implement once understood.

2. **Rule-based routing is a real, valid production choice** — not just a "beginner" approach. It's faster and cheaper than LLM-based routing, and its failure modes (like the K3 bug) are debuggable and fixable, unlike opaque LLM misclassifications.

3. **Memory via query rewriting > raw history append** — rewriting follow-ups into standalone questions before retrieval means the retrieval step doesn't need to "understand" conversational context itself; that complexity is isolated to one place.

4. **Not all web content is retrievable, and that's a legitimate constraint** — production RAG systems need clear boundaries around what they will and won't attempt to fetch, both for technical reasons (JS-rendering) and legitimate access-control/ToS reasons (bot detection, scraping restrictions).

5. **Defensive coding matters even in "internal" pipeline steps** — the empty-memory-rewrite bug showed that every LLM call in a pipeline (not just the final user-facing one) needs validation and fallback behavior.

---

## 🔮 Future Improvements

- [ ] Source-based filtering (query only specific loaded sources, not the whole mixed pool)
- [ ] Incremental index updates (currently rebuilds FAISS+BM25 fully on each new source — fine at this scale, wouldn't scale to frequent updates)
- [ ] LLM-based routing as a fallback when rule-based routing has low confidence
- [ ] Streamlit UI (currently CLI-based, like Projects 1 & 2's initial versions)
- [ ] Persistent memory across sessions (currently resets each run)

---

## 🚀 Deployment

**Local Setup:**
```bash
uv pip install -r requirements.txt
python main.py
```

**Usage:**
```
You: load <url>          # add a website to the knowledge base
You: preview <url>       # check if a URL is loadable before committing
You: <any question>      # calculator, datetime, or document Q&A
You: quit                # exit
```

**Repository:** [Your GitHub Link]

---

## 📄 License
MIT License

## 👨‍💻 Author
**Nakshatra Joshi**