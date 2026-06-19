# AIBC-2026-Compass

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about CPF (Central Provident Fund) policies in Singapore. It scrapes content from cpf.gov.sg, embeds it into a vector database, and uses GPT-4o-mini to generate grounded answers with source citations.

## Architecture

```
User Query → Self-Query (category detection) → Vector DB Search → Context Builder → LLM → Response
```

### Pipeline

1. **Data Scraper** (`Source/Data_Scraper.py`) — Crawls cpf.gov.sg sitemap, extracts markdown content, removes boilerplate
2. **Vector Database** (`Source/Vector_Database.py`) — Chunks markdown files and embeds into ChromaDB for semantic search
3. **Chat Engine** (`Source/Chat_Engine.py`) — Orchestrates retrieval and generation with conversation history

## Project Structure

```
AIBC-2026-Compass/
├── Source/
│   ├── __init__.py          # Package initialiser
│   ├── Constants.py         # Centralised configuration
│   ├── Prompts.py           # LLM prompt templates
│   ├── Data_Scraper.py      # Web scraping pipeline
│   ├── Vector_Database.py   # Embedding and vector store
│   └── Chat_Engine.py       # RAG chat orchestrator
├── Data/                    # Generated at runtime (gitignored)
│   ├── scraped_pages/       # Scraped markdown files
│   ├── vector_store/        # ChromaDB persistent storage
│   ├── resources/           # Categories, logs, template lines
│   └── chat_history.json    # Archived conversation logs
├── app.py                   # Streamlit web interface (primary UI)
├── main.py                  # CLI entry point (scrape + embed pipeline)
├── requirements.txt
└── .gitignore
```

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key (for LLM and production embeddings)

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
OPENAI_API_KEY=sk-...   # Required for Chat Engine and production embeddings
```

## Running the Application

### Web Interface (Primary)

```bash
python -m streamlit run app.py
```

This launches a two-tab Streamlit application:
- **CPF Chat** — Ask questions about CPF policies and receive grounded answers with source citations
- **Data Management** — Re-scrape the CPF website and update the vector database on demand

Default login credentials: `Admin` / `Beacon`

### CLI Pipeline

```bash
python main.py
```

Runs the data preprocessing pipeline only (scrape + embed). Does not start the chat interface.

## Configuration

All settings are in `Source/Constants.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 2000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `USE_LOCAL_EMBEDDINGS` | True | Use ChromaDB ONNX model (offline) vs OpenAI |
| `LLM_MODEL_ID` | gpt-4o-mini | LLM for generation and self-query |
| `DEFAULT_TOP_K` | 5 | Number of chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | 0.3 | Minimum similarity score for results |
| `MAX_HISTORY_TURNS` | 5 | Conversation turns before summarisation |

### Embedding Models

| Mode | Model | Dimensions | Use Case |
|------|-------|-----------|----------|
| Local (`True`) | all-MiniLM-L6-v2 (ONNX) | 384 | Offline development/testing |
| Production (`False`) | text-embedding-3-small | 1536 | Streamlit Cloud deployment |

> **Note:** Switching embedding models requires deleting `Data/vector_store/` and re-embedding.

## Key Features

### Data Scraper (`Source/Data_Scraper.py`)
- Crawls cpf.gov.sg sitemap with robots.txt compliance
- Converts custom HTML heading divs (`headline-N`) to proper markdown headings
- Strips template noise: nav, footer, social sharing, site-wide notices
- Truncates trailing boilerplate (FAQ links, Resources, Related Reads)
- Detects repeated template lines across 50%+ of files for manual review
- Skips pages that produce empty content after processing
- Generates hierarchical category paths from URL structure
- Exports both a flat category list and visual category tree
- Tracks content hashes for change detection between scrapes

### Vector Database (`Source/Vector_Database.py`)
- Chunks markdown using LangChain's MarkdownTextSplitter (heading-aware)
- Supports local ONNX embeddings (offline) and OpenAI embeddings (production)
- Hash-based change detection: only re-embeds modified pages
- Purges vectors for deleted files automatically
- Accepts pre-built where filters for category-based retrieval

### Chat Engine (`Source/Chat_Engine.py`)
- LLM-powered self-query retriever parses queries into search terms + category filters
- Category prefix expansion: a root category includes all sub-categories via `$in`
- Conversation-aware: passes recent history to self-query for follow-up resolution
- Formats retrieved chunks with category path and source URL for citation
- Automatic history summarisation when conversation exceeds turn limit
- Archives conversations to JSON before reset

### Prompts (`Source/Prompts.py`)
- `SELF_QUERY_PROMPT` — Instructs LLM to extract search query + category filters from user input
- `SYSTEM_PROMPT` — Grounds LLM responses to only use provided context with citations
- `SUMMARY_PROMPT` — Compresses older conversation turns while preserving key details

### Streamlit Frontend (`app.py`)
- Login-protected interface
- Real-time chat with source citation expandables
- Data Management tab for triggering scrape/embed with progress bars
- Status metrics: scraped page count, vector count, active embedding model

## Chat Pipeline

When a user asks a question, the Chat Engine executes this pipeline:

```
1. Self-Query        → LLM analyses query + category tree → refined query + category filters
2. Category Expansion → Prefix filters expanded to all matching sub-paths via $in
3. Vector Search     → Semantic similarity search with category filter applied
4. Context Building  → Top-k chunks formatted with category and source URL
5. Prompt Assembly   → System prompt + conversation history + context + question
6. LLM Generation   → GPT-4o-mini produces grounded answer with citations
7. History Update    → Exchange stored; older turns summarised if limit exceeded
```

### Example Flow

**User:** "What insurance covers hospitalisation?"

1. **Self-Query** → refined query: "insurance coverage for hospitalisation", categories: `["healthcare financing"]`
2. **Expansion** → `["healthcare financing", "healthcare financing/medishield life", "healthcare financing/using your medisave savings", ...]`
3. **Search** → 5 chunks retrieved from healthcare financing pages
4. **Context** → Each chunk prefixed with `[Source 1: healthcare financing/medishield life | https://...]`
5. **Generation** → LLM answers using only retrieved context, citing source URLs

### Follow-up Handling

The self-query LLM receives the last 2 exchanges as context, enabling it to resolve ambiguous follow-ups:

- User: "Can I invest my special account?" → category: `growing your savings/earning higher returns`
- User: "How much do I need?" → same category inferred from conversation context

## Deployment

For Streamlit Cloud:
1. Set `USE_LOCAL_EMBEDDINGS = False` in `Source/Constants.py`
2. Add `OPENAI_API_KEY` to Streamlit secrets
3. Run the data pipeline at least once locally or via the Data Management tab — `Data/` is not auto-generated on first launch
4. The run command is `streamlit run app.py`
