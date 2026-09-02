# Agentic RAG System
### Hybrid Search (BM25 + Semantic + RRF) · Conversation Memory · Tool Routing · Dynamic Knowledge Base

A retrieval-augmented generation system that goes beyond simple document Q&A — it routes questions to the right tool (calculator, date/time, or document retrieval), remembers conversation context, and can expand its own knowledge base with new web sources at runtime.

Built as the third and most advanced project in a progressive RAG learning series ([Project 1](https://github.com/Nakshatrajoshi/RAG) → [Project 2](https://github.com/Nakshatrajoshi/RAG_2) → Project 3).

---

## ✨ Features

- **Hybrid Retrieval** — combines BM25 keyword search with semantic embedding search, merged via Reciprocal Rank Fusion (RRF)
- **Agentic Tool Routing** — rule-based router directs queries to a calculator, date/time tool, or document retrieval
- **Conversation Memory** — rewrites follow-up questions ("what about that?") into standalone questions using chat history
- **Dynamic Knowledge Base** — load new websites into the vector store at runtime, alongside existing PDFs
- **Safety Checks** — rejects thin/blocked web content (JS-rendered pages, bot-protected sites) instead of silently ingesting garbage

## 🛠️ Tech Stack

- **Framework:** LangChain
- **LLM:** OpenRouter (nvidia/nemotron, OpenAI-compatible API)
- **Vector Store:** FAISS
- **Keyword Search:** BM25 (rank_bm25)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Document Loading:** PyPDF (PDFs), WebBaseLoader (websites)
- **Language:** Python 3.12

## 📂 Project Structure

```
RAG_3/
├── main.py                   # Entry point - interactive chat loop
├── router.py                 # Rule-based query routing logic
├── tools.py                  # Calculator and datetime tools
├── retrieval.py               # Hybrid search (BM25 + semantic + RRF), web loading
├── memory.py                  # Conversation memory / query rewriting
├── config.py                  # Configuration (model, chunk size, paths)
├── evaluate.py                 # Evaluation script (router accuracy, end-to-end tests)
├── evaluation-results.json     # Saved evaluation output
├── CASE_STUDY.md               # Detailed write-up: architecture, decisions, bugs found & fixed
├── requirements.txt
├── .env                        # API keys (not committed)
└── data/                       # PDF source documents (not committed - see below)
```

## 🚀 Installation

```bash
git clone https://github.com/Nakshatrajoshi/RAG_3.git
cd RAG_3

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

## 📄 Data

PDF files are **not included** in this repository (excluded via `.gitignore` to keep the repo lightweight). To run this project:

1. Create a `data/` folder in the project root
2. Add your own PDF files to it
3. Run the app — it will automatically load and index all PDFs found in `data/`

## ▶️ Usage

```bash
python main.py
```

Once running, you get an interactive chat prompt supporting:

| Input | Behavior |
|---|---|
| `What is 25 + 17?` | Routes to calculator |
| `What's today's date?` | Routes to datetime tool |
| `What does the document say about X?` | Routes to hybrid document retrieval |
| `load <url>` | Fetches a webpage and adds it to the knowledge base |
| `preview <url>` | Checks if a URL is loadable before committing it |
| `quit` / `exit` | Ends the session |

**Example session:**
```
You: load https://en.wikipedia.org/wiki/Retrieval-augmented_generation
✅ Website added to my knowledge base — ask me about it!

You: what does the article say about RAG evaluation?
   → Routed to: retrieve
AI: [answer generated from hybrid search over PDFs + loaded webpage]

You: can you go deeper on that?
   (Rewritten to: 'Can you provide more detail on RAG evaluation methods...')
   → Routed to: retrieve
AI: [answer, using conversation-aware rewritten query]
```

## 📊 Evaluation

```bash
python evaluate.py
```

Runs two test suites:
1. **Router accuracy** — verifies queries are correctly classified (calculator / datetime / retrieve)
2. **End-to-end pipeline** — verifies the full flow produces sensible answers, including memory-dependent follow-ups

Results are saved to `evaluation-results.json`.

## ⚠️ Known Limitations

- **JS-rendered / bot-protected websites cannot be loaded** (e.g., Amazon, LinkedIn, social media) — `WebBaseLoader` only fetches raw HTML, and these sites either render content client-side or actively block automated access. The system detects and rejects these cases rather than silently ingesting empty/garbage content.
- **No source-based filtering** — currently, retrieval searches across all loaded content (PDFs + all added websites) as one pool; there's no way to query only a specific source.
- **Full index rebuild on each addition** — adding a new website rebuilds the entire FAISS + BM25 index rather than updating incrementally. Fine at current scale, would need optimization for frequent large-scale updates.

See [`CASE_STUDY.md`](./CASE_STUDY.md) for a full write-up of the architecture, design decisions, and real bugs found and fixed during development.

## 🔮 Future Improvements

- [ ] Source-based filtering (query specific loaded documents/sites)
- [ ] Incremental index updates
- [ ] LLM-based routing fallback for ambiguous queries
- [ ] Streamlit UI
- [ ] Persistent memory across sessions

## 📄 License

MIT License

## 👨‍💻 Author

**Nakshatra Joshi**
[GitHub](https://github.com/Nakshatrajoshi) · [LinkedIn](http://www.linkedin.com/in/nakshatra-joshi-97a773212)
