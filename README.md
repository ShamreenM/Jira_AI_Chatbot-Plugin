# 🤖 JIRA AI Chatbot — Forge Plugin

> A production-grade RAG-based AI chatbot embedded directly inside JIRA Cloud as an Atlassian Forge plugin. Ask natural language questions about your JIRA issues and get instant, grounded answers — without leaving your project.

---

## 📸 Architecture Overview

**Hybrid + Query Routing + MultiQuery + Re-ranking + Forge Plugin + In-Memory Caching**

```
User (JIRA Cloud)
      │
      ▼
Atlassian Forge Plugin  (React Chat UI — Custom UI project page)
      │  HTTPS POST /query
      ▼
Cloudflare Tunnel  (public URL → localhost:8000)
      │
      ▼
FastAPI Backend  —  api.py
      │
      ▼
JiraAIChatbot4.py  —  RAG Core
      ├── Query Intent Router
      │     ├── Ticket Key Query     → Chroma metadata filter (issue_key)
      │     ├── Date Query           → In-memory Date Index Cache scan
      │     ├── Structured Filter    → Chroma $where (status/priority/resolution/severity)
      │     └── General Semantic     → MultiQuery Expansion → Hybrid Search
      │
      ├── Hybrid Retrieval
      │     ├── Dense  — ChromaDB vector search  (BAAI/bge-base-en-v1.5)  weight 0.7
      │     └── Sparse — BM25 keyword index                                weight 0.3
      │
      ├── Dynamic Top-K Filtering + Deduplication
      ├── Cross-Encoder Re-ranking  (BAAI/bge-reranker-base)
      │
      └── Intent-Based Prompting → Google Gemini 2.5 Flash
            ├── Extraction Mode  — strict field extraction
            └── Analysis Mode    — reasoning with anti-hallucination guardrails
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Forge Plugin** | Embedded as a native JIRA project page — no external tab needed |
| **Query Routing** | 4 intent paths: ticket key, date, structured filter, semantic |
| **Hybrid Search** | Dense (ChromaDB) + Sparse (BM25) with weighted score fusion |
| **Cross-Encoder Reranking** | `BAAI/bge-reranker-base` selects top 5 most relevant docs |
| **Structured Metadata Filter** | Direct Chroma `$where` filter for status/priority/resolution/severity queries |
| **Exact Count Injection** | Real ticket counts fetched from full dataset and injected into LLM context |
| **Date Index Cache** | Full collection scanned once per session and cached in memory |
| **Anti-Hallucination Guardrails** | Analysis prompt restricts LLM to only fields present in context |
| **Settings Panel** | API URL and Gemini key configurable at runtime — no redeployment needed |

---

## 🗂 Repository Structure

```
Jira_AI_Chatbot-Plugin/
│
├── JiraAIChatbot4.py          # RAG core — retrieval, routing, prompting
├── api.py                     # FastAPI wrapper exposing /query and /health
├── requirements.txt           # Python dependencies
│
└── jira-chatbot-plugin/       # Atlassian Forge app
    ├── manifest.yml           # Forge app config + external fetch permissions
    └── static/
        └── hello-world/
            └── src/
                └── App.js     # React chat UI
```

---

## 🛠 Tech Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- LangChain Community (`langchain-community`, `langchain-core`)
- ChromaDB — local vector store
- HuggingFace `BAAI/bge-base-en-v1.5` — dense embeddings
- `sentence-transformers` `BAAI/bge-reranker-base` — cross-encoder reranking
- `rank-bm25` — sparse keyword retrieval
- `langchain-google-genai` + Google Gemini 2.5 Flash — LLM

**Frontend / Plugin**
- Atlassian Forge (Custom UI)
- React (App.js)
- `@forge/bridge` for Forge runtime

**Tunnel**
- Cloudflare Tunnel (`cloudflared`) — exposes local FastAPI to Atlassian Cloud

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Atlassian developer account (free at [developer.atlassian.com](https://developer.atlassian.com))
- Forge CLI: `npm install -g @forge/cli`
- Cloudflare Tunnel: download from [developers.cloudflare.com](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
- Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))

### 1. Clone the repo

```bash
git clone https://github.com/ShamreenM/Jira_AI_Chatbot-Plugin.git
cd Jira_AI_Chatbot-Plugin
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up ChromaDB

Place your pre-built `chroma_db_backup/` folder in the project root. This folder is excluded from the repository (`.gitignore`) due to size — it must be set up locally.

If building from scratch, ingest your JIRA export CSV into ChromaDB first using your ingestion script.

### 4. Start the FastAPI backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### 5. Start Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

Note the public URL printed in the terminal, e.g.:
```
https://your-tunnel-name.trycloudflare.com
```

### 6. Deploy the Forge plugin

```bash
cd jira-chatbot-plugin
forge login           # authenticate with your Atlassian account
```

Build the React UI:

```bash
cd static/hello-world
npm install
npm run build
cd ../..
```

Deploy and install:

```bash
forge deploy
forge install         # select Jira → enter your Atlassian site URL
```

### 7. Configure in JIRA

1. Open any JIRA project → find **JIRA AI Chatbot** in the left sidebar
2. Click **⚙️ Settings**
3. Enter your **Gemini API key**
4. Confirm the **API URL** matches your Cloudflare Tunnel URL
5. Start chatting!

---

## 💬 Example Queries

| Query type | Example |
|---|---|
| Ticket lookup | `What is the status of SRCTREEWIN-14037?` |
| Date query | `Show issues updated on 15/Jan/2024` |
| Status filter | `List all In Progress tickets` |
| Priority filter | `Show all High priority bugs` |
| Count comparison | `How many tickets are High vs Medium priority?` |
| Analysis | `Why are so many tickets stuck in Needs Triage?` |
| Resolution | `Show all tickets resolved as Fixed` |

---

## 🔒 Security Notes

- The Gemini API key is **never stored server-side** — it is entered by the user in the plugin Settings panel and sent with each request.
- Cloudflare Tunnel exposes only port 8000 with no credentials by default. For production use, add a shared secret header in `api.py` and validate it on every request.
- The free `trycloudflare.com` tunnel URL changes on every restart. For a stable URL, set up a named Cloudflare tunnel with your own domain.

---

## ⚠️ Known Limitations

- The local FastAPI server and Cloudflare Tunnel must be **running on your machine** for the plugin to work — this is a POC/demo architecture, not a 24/7 production setup.
- The `_date_index_cache` is in-memory only. If ChromaDB is updated while the server is running, restart the server to refresh the cache.
- The free Cloudflare Quick Tunnel URL changes on every restart. Update the URL in the plugin Settings panel after each restart.

---

## 🗺 Roadmap

- [ ] Migrate ChromaDB to Qdrant Cloud (eliminate local machine dependency)
- [ ] Add shared-secret authentication header between Forge plugin and FastAPI
- [ ] Deploy FastAPI to a cloud server (Render / Railway) for 24/7 availability
- [ ] Support multi-project JIRA data ingestion
- [ ] Add conversation history / multi-turn chat support

---

## 📄 License

This project is for personal and demonstration purposes.
