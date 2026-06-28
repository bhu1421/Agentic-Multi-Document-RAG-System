# 🧠 Agentic RAG System

**A smart, autonomous document assistant that knows *how* to find your answers.**

Unlike standard chatbots that just blindly search a database for every question, this system uses an **Agentic Router**. When you ask a question, the AI pauses to think: *"Should I search the user's uploaded documents? Should I search the live web? Or do I already know the answer?"* 

It automatically picks the best strategy, executes a high-precision search, and generates a grounded response.

---

## 🌟 What Can You Do With It?

1. **Upload Anything:** Drop in PDFs, Markdown files, Text files, or even paste a GitHub repository URL.
2. **Ask Anything:** 
   - Ask about your documents: *"Summarize my uploaded resume."* (Routes to local documents)
   - Ask about the world: *"What is the weather in London today?"* (Routes to Web Search)
   - Ask general questions: *"What is the capital of France?"* (Routes to LLM Knowledge)
3. **Get Precise Answers:** The system doesn't just find keywords. It reads the context around the keywords (Hierarchical Retrieval) and re-evaluates the results (Cross-Encoder Reranking) to give you the most accurate answer possible.

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

3. **Configure your API Key:**
Create a `.env` file at the root of the project and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Start the app:**
```bash
streamlit run app.py
```

### Option 2: Docker (Production)

If you have Docker installed, simply run:
```bash
docker compose up -d --build
```
*You can then access the application at `http://localhost:8501`*

---

## ⚙️ How It Works Under The Hood

This is a production-grade Retrieval-Augmented Generation (RAG) system orchestrated by **LangGraph**.

### 1. Intelligent Routing
Every query passes through an LLM router (powered by Groq and Pydantic structured output) which classifies the intent into one of four paths:
- `llm_knowledge` — Answer directly from the model's parametric knowledge.
- `targeted_search` — Restrict search space to a specific named document.
- `all_docs` — Perform semantic search across all indexed sources.
- `web_search` — Fetch live, up-to-date results from DuckDuckGo.

### 2. Two-Stage Precision Retrieval
When searching your documents, the system employs a sophisticated pipeline:
- **Stage 1 (Fast Recall):** Maximal Marginal Relevance (MMR) search retrieves up to 30 diverse candidate chunks from a local Qdrant vector database.
- **Stage 2 (High Precision):** Hierarchical parent expansion restores the full document context (so the AI doesn't read fragmented sentences), followed by cross-encoder reranking (`BAAI/bge-reranker-large`) to precisely score and select the best context.

### 3. Web Search Fallback Cycle
If the system searches your documents and realizes the answer isn't there, it won't hallucinate. The LangGraph state machine will autonomously trigger a web search fallback cycle, browse the internet, and try again.

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
           [Router Node]  ← LLM + Pydantic structured output
                │
     ┌──────────┼──────────┐
     ▼          ▼          ▼
[Retriever] [WebSearch] [Generate]
     │                     ▲
     └─ MMR Search          │
     └─ Parent Expansion    │
     └─ Cross-Encoder ──────┘
```

---

## 📂 Project Structure

```text
AGENTIC_RAG1/
├── app.py                    # Streamlit UI entry point
├── docker-compose.yml        # Container orchestration
├── Dockerfile                # Production-ready slim image
├── backend/
│   ├── config.py             # Single source of truth for all constants
│   ├── graph.py              # LangGraph node definitions + compilation
│   ├── rag.py                # Pipeline entry points
│   ├── router.py             # Pydantic structured output router
│   ├── retrieval.py          # MMR, parent expansion, reranker
│   ├── vectordb.py           # Qdrant client, payload indices
│   ├── chunker.py            # Format-aware document chunking
│   ├── loader.py             # File and Web loaders
│   ├── llm.py                # Groq LLM factory (lru_cache)
│   ├── state.py              # AgentState TypedDict
│   └── logger.py             # Centralised logging
├── ui/
│   ├── chat.py               # Chat interface & streaming progress
│   ├── sidebar.py            # File uploader, DB management
│   └── style.py              # Custom CSS styling
├── eval/                     # Router accuracy + latency benchmark suite
├── uploaded_docs/            # Persistent volume for raw uploads
└── local_qdrant/             # Persistent volume for the Vector Database
```

---

## 📊 Benchmark the Router

You can automatically benchmark the LLM router's accuracy and latency using the included evaluation suite:

```bash
python eval/benchmark.py
```
*Results will be automatically written to `eval/results.md`*
