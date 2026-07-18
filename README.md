# 🧠 Agentic RAG System

**A smart, autonomous document assistant that knows *how* to find your answers.**

Unlike standard chatbots that just blindly search a database for every question, this system uses an **Agentic Router**. When you ask a question, the AI pauses to think: *"Should I search the user's uploaded documents? Should I search the live web? Or do I already know the answer?"*

It automatically picks the best strategy, executes a high-precision search, and **streams the answer token-by-token** as it's generated — no waiting for the full response.

---

## 🌟 What Can You Do With It?

1. **Upload Anything:** Drop in PDFs, Markdown files, Text files, or even paste a GitHub repository URL.
2. **Ask Anything:**
   - Ask about your documents: *"Summarize my uploaded resume."* (Routes to local documents)
   - Ask about the world: *"What is the weather in London today?"* (Routes to Web Search)
   - Ask general questions: *"What is the capital of France?"* (Routes to LLM Knowledge)
   - Just chat: *"Hi!"*, *"How are you?"* — works as a normal AI assistant too
3. **Get Precise Answers:** The system doesn't just find keywords. It reads the context around the keywords (Hierarchical Retrieval) and re-evaluates the results (Cross-Encoder Reranking) to give you the most accurate answer possible.
4. **Watch It Think:** A live status box shows every pipeline step — cache check, routing, retrieval, generation — as it happens. The answer streams word-by-word in real time.

---

## 🚀 Quickstart / How To Run

You can run the app using either native Python or Docker.

### Option 1: Native Python (Development)

1. **Set up a virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure your API Keys:**
Create a `.env` file at the root of the project (copy from `.env.example`):
```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # optional, for web search
```

4. **Start the app:**
```bash
streamlit run app.py
```
*App opens at `http://localhost:8501`*

### Option 2: Docker (Production)

If you have Docker installed, simply run:
```bash
docker compose up -d --build
```
*You can then access the application at `http://localhost:8501`*

---

## ⚙️ How It Works Under The Hood

This is a production-grade Retrieval-Augmented Generation (RAG) system orchestrated by **LangGraph**.

### 1. Guardrails & Caching
Before any intensive processing, queries are checked against a Redis-backed cache to instantly return known answers. A **Guardrail** node then validates the query — blocking harmful or clearly malicious requests while allowing greetings, casual conversation, general knowledge questions, and document queries to pass through.

### 2. Intelligent Routing
Every query passes through an LLM router (powered by Groq and Pydantic structured output) which classifies the intent into one of four paths:
- `llm_knowledge` — Answer directly from the model's parametric knowledge.
- `targeted_search` — Restrict search space to a specific named document.
- `all_docs` — Perform semantic search across all indexed sources.
- `web_search` — Fetch live, up-to-date results using Tavily Search.

### 3. Precision Retrieval & Self-Correction
When searching your documents, the system employs a sophisticated pipeline:
- **Hybrid Search:** Fuses dense vector search (MMR) with BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Hierarchical Expansion:** Expands matched chunks to include their parent document context.
- **Cross-Encoder Reranking:** Re-scores the expanded candidates for maximum relevance.
- **Self-Correction:** If retrieval confidence is low, a **Query Rewriter** autonomously reformulates the question and retries.

### 4. Token-Level Streaming
The LLM answer streams **token-by-token** directly to the UI as it's generated — using LangChain's `.stream()` API piped through a `queue.SimpleQueue` into a live `st.empty()` placeholder with a blinking `▌` cursor. No waiting for the full response to complete.

### 5. Web Search Fallback
If the system searches your documents and realizes the answer isn't there, it won't hallucinate. The LangGraph state machine autonomously triggers a web search fallback, fetches live results, and generates a grounded answer.

---

## 🏗️ System Architecture

```text
Streamlit UI (app.py)
        │
        ├── Document Ingestion
        │       loader.py → chunker.py → vectordb.py (Qdrant)
        │
        └── Chat Pipeline (LangGraph StateGraph)
                │
                ▼
          [Cache Check] ──(hit)──► [END]
                │(miss)
                ▼
          [Guardrails] ──(blocked)─► [END]
                │(pass)
                ▼
           [Router Node]
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
[Retriever] [WebSearch] [Generate] ──► [Cache Store] ──► [END]
     │          │    ▲      │(fallback)
     ▼          └────┘      └──► [WebSearch]
 [Evaluate] ──(confident)──►[Generate]
     │
   (low)
     ▼
 [Rewrite] ──► (retries Retriever)
```

**Streaming flow inside `[Generate]`:**
```
chain.stream() → token chunks → SimpleQueue → ("token", chunk) events → st.empty() placeholder
```

---

## 📂 Project Structure

```text
AGENTIC_RAG1/
├── app.py                    # Streamlit UI entry point
├── docker-compose.yml        # Container orchestration (Redis + Qdrant + App)
├── Dockerfile                # Production-ready slim image
├── backend/
│   ├── cache.py              # Redis caching logic
│   ├── chunker.py            # Format-aware document chunking
│   ├── config.py             # Single source of truth for all constants
│   ├── graph.py              # LangGraph node definitions + compilation
│   ├── guardrail.py          # Input validation, greeting bypass, off-topic LLM check
│   ├── llm.py                # Groq LLM factory (lru_cache)
│   ├── loader.py             # File and Web loaders
│   ├── logger.py             # Centralised logging
│   ├── rag.py                # Pipeline entry points + token streaming
│   ├── retrieval.py          # MMR, parent expansion, reranker
│   ├── rewrite.py            # Query rewriting and self-correction
│   ├── router.py             # Pydantic structured output router
│   ├── state.py              # AgentState TypedDict (incl. token_queue)
│   └── vectordb.py           # Qdrant client, payload indices
├── ui/
│   ├── chat.py               # Chat interface, live token streaming UI
│   ├── sidebar.py            # File uploader, DB management
│   └── style.py              # Custom CSS styling
├── eval/                     # Router accuracy + latency benchmark suite
├── uploaded_docs/            # Persistent volume for raw uploads
└── local_qdrant/             # Persistent volume for the Vector Database
```

---

## ✨ Recent Updates

| # | Feature |
|---|---|
| 14 | **Token-level streaming** — LLM answer streams word-by-word with a live `▌` cursor; no waiting for full response |
| 14 | **Relaxed guardrail** — Greetings, casual chat, and personal intros fast-pass without an LLM call; off-topic prompt tuned to be more permissive |

---

## 📊 Benchmark the Router

You can automatically benchmark the LLM router's accuracy and latency using the included evaluation suite:

```bash
python eval/benchmark.py
```
*Results will be automatically written to `eval/results.md`*
