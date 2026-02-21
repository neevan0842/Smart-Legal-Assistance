# Smart Legal Assistance

A RAG-powered legal assistant using FastAPI, Pinecone, and Groq LLM to answer questions based on BNS (Bharatiya Nyaya Sanhita) and BNSS (Bharatiya Nagarik Suraksha Sanhita) legal documents.

## Features

- 🔍 Hybrid search using Pinecone dense and sparse embeddings
- 🤖 AI-powered answers using Groq LLM (Llama 3.3 70B)
- ⚡ Fast API with streaming response support
- 📄 PDF document processing and chunking
- 🎯 Semantic reranking for accurate results

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Pinecone account
- Groq API account

## Setup

### 1. Install uv

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Navigate

```bash
git clone https://github.com/neevan0842/Smart-Legal-Assistance.git
cd Smart-Legal-Assistance
```

### 3. Set Up Environment Variables

Copy the sample environment file:

```bash
cp .env.sample .env
```

Edit `.env` and fill in your credentials:

```env
# PINECONE CONFIGURATION
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_DENSE_INDEX_NAME=dense-index-name
PINECONE_SPARSE_INDEX_NAME=sparse-index-name
PINECONE_DENSE_INDEX_MODEL=llama-text-embed-v2
PINECONE_SPARSE_INDEX_MODEL=pinecone-sparse-english-v0
PINECONE_DENSE_HOST=https://your-dense-index-host.pinecone.io
PINECONE_SPARSE_HOST=https://your-sparse-index-host.pinecone.io
PINECONE_RERANKING_MODEL=bge-reranker-v2-m3

# GROQ CONFIGURATION
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile

# FRONTEND CONFIGURATION
FRONTEND_URLS=http://localhost:3000
```

#### Where to Find API Keys:

**Pinecone:**

1. Sign up at [pinecone.io](https://www.pinecone.io/)
2. Go to **API Keys** section
3. Copy your API key
4. After creating indexes (see below), copy the host URLs from the index details

**Groq:**

1. Sign up at [console.groq.com](https://console.groq.com/)
2. Go to **API Keys**
3. Create and copy your API key

### 4. Install Dependencies

```bash
uv sync
```

### 5. Place Legal Documents

Add your PDF documents to `scripts/documents/`:

- `BNS.pdf` - Bharatiya Nyaya Sanhita
- `BNSS.pdf` - Bharatiya Nagarik Suraksha Sanhita

### 6. Populate Pinecone Database

Run the script to extract sections from PDFs and upload to Pinecone:

```bash
uv run -m scripts.pinecone_seeding
```

### 7. Start the Server

**Development mode with auto-reload:**

```bash
uv run fastapi dev
```

**Production mode:**

```bash
uv run fastapi run
```

The API will be available at `http://127.0.0.1:8000`

## API Documentation

Once the server is running, access:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## Project Structure

```
Smart-Legal-Assistance/
├── LICENSE
├── pyproject.toml              # Project dependencies
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── generate/
│   │       ├── __init__.py
│   │       ├── api.py          # Route handlers
│   │       ├── schema.py       # Pydantic models
│   │       └── service.py      # Business logic
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Settings management
│   └── middlewares/
│       ├── __init__.py
│       └── logger.py           # Logging & security headers
├── scripts/
│   ├── __init__.py
│   ├── bns_bnss_to_pinecone.py # Data ingestion script
│   ├── query.ipynb             # Query notebook
│   ├── utils.py                # Helper functions
│   └── contents/               # Extracted sections (auto-generated)
│   └── documents/              # Place PDFs here
└── .env                        # Environment variables (create from .env.sample)
└── .env.sample                 # Environment template
```

## Development

### Using uv Commands:

```bash
# Sync dependencies
uv sync

# Add new package
uv add package-name

# Run FastAPI
uv run fastapi dev
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
