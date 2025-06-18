# LangChain RAG & SQL Agent

An advanced RAG (Retrieval-Augmented Generation) and SQL Agent system built with LangChain. It supports intelligent Q&A on multiple document formats and natural language interactions with SQL databases.

The latest version integrates support for **local large language models (via Ollama)**, enabling flexible switching between OpenRouter cloud models and local models.

## 🚀 Features

- **Latest LangChain Version**: Built on LangChain `0.3+` for optimal performance
- **🤖 Dual Intelligence Mode**:
    - **RAG Q&A**: Intelligent retrieval and Q&A on PDF, DOCX, TXT, MD, CSV, Excel, and other document formats
    - **SQL Agent**: Natural language interaction with SQL databases for querying and analysis
- **🔄 Flexible LLM Support**:
    - **☁️ Cloud Models**: Default support for multiple mainstream models (GPT, Claude, Llama) via OpenRouter
    - **💻 Local Models**: Support for local model deployment via **Ollama** (Mistral, Llama 3, etc.)
- **💭 Conversation Memory**: Multi-turn conversation context memory support
- **🔗 LCEL Chain Structure**: Built with LangChain Expression Language (LCEL) for clear, scalable processing flows
- **📊 Vector Storage**: Efficient vector storage and retrieval using FAISS
- **🔥 Dynamic Vector Store (NEW!)**: 
    - **📁 File System Monitoring**: Automatic detection of file changes using **Watchdog**
    - **🔄 Real-time Updates**: Automatic addition/removal of documents from vector store
    - **🛠️ MCP Integration**: Model Context Protocol support for filesystem operations
    - **⚡ Hot Reload**: No need to rebuild vector store when documents change
- **🌐 Web API**: Complete API service based on FastAPI and LangServe
- **🎛️ Multiple Deployment Modes**: Support for command-line, interactive session, and API service modes

## 🛠 Installation and Configuration

### 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# If you plan to use Ollama local models, install additional Ollama dependencies
pip install langchain-ollama
```

### 2. Environment Configuration

Create a `.env` file and configure environment variables according to your needs.

#### Configuration 1: Using OpenRouter Cloud Models (Default)

This is the simplest out-of-the-box configuration using cloud LLMs provided by OpenRouter.

```env
# 1. Set LLM provider to "openrouter"
LLM_PROVIDER="openrouter"

# 2. Provide your OpenRouter API key
OPENROUTER_API_KEY="sk-or-v1-..."

# 3. [Optional] Specify default model
DEFAULT_MODEL="gpt-4o-mini"
```

#### Configuration 2: Using Ollama Local Models

If you have deployed models locally via Ollama (such as Mistral, Llama 3), use this configuration.

**Prerequisites**: Ensure you have installed and are running [Ollama](https://ollama.com/).

```env
# 1. Set LLM provider to "ollama"
LLM_PROVIDER="ollama"

# 2. [Optional] Ollama service URL, if not the default http://localhost:11434
# OLLAMA_BASE_URL="http://localhost:11434"

# 3. [Optional] Ollama model name to use, must be a model you have locally
# OLLAMA_MODEL="mistral"
```

### 3. Prepare Data

- **Documents**: Place documents for Q&A in the `data/document/` directory
- **Database**: The system includes a sample SQLite database at `data/database/erp.db`

## 🎯 Usage Methods

### Method 1: Command Line Usage

#### 1. Build Vector Store (Traditional Mode)

Before first run or after document updates, you need to build or update the vector store.

```bash
# Build vector store for documents in `data/document/`
python src/main.py build

# Force rebuild vector store
python src/main.py build --force-rebuild
```

#### 1.1. Dynamic Vector Store (DEFAULT!)

**No need to rebuild!** Dynamic vector store is now **enabled by default**. Documents are automatically processed when added/modified/removed.

```bash
# Default mode (dynamic vector store enabled)
python src/main.py chat

# Disable dynamic mode if needed (use traditional static mode)
python src/main.py chat --no-dynamic
```

#### 2. Interactive Chat Mode

Start interactive command-line interface.

```bash
# Start interactive chat
python src/main.py chat

# Start with specific mode
python src/main.py chat --mode doc    # Start in document Q&A mode
python src/main.py chat --mode sql    # Start in SQL query mode
```

After system startup, you can ask questions directly. It supports two modes:

- **📄 Document Q&A Mode (doc)**: Default mode, answers questions about documents using RAG
- **🗃️ SQL Query Mode (sql)**: Natural language interaction with database

**Interactive Commands**:
- `mode doc`: Switch to document Q&A mode
- `mode sql`: Switch to SQL query mode
- `clear`: Clear current mode's conversation memory
- `quit` or `exit`: Exit program

**Vector Store Management Commands** (available by default):
- `status`: Show vector store status and file tracking info
- `sync`: Force synchronize with filesystem
- `files`: List tracked files and their document counts

#### 3. Start API Server

```bash
# Start API server (default configuration)
python src/main.py server

# Start with custom host and port
python src/main.py server --host 127.0.0.1 --port 8080

# Start in debug mode
python src/main.py server --debug
```

The service will start at `http://localhost:8000` and provide complete API documentation (Swagger UI).

### Method 2: Direct Python Script Usage

#### Single Document Q&A

```python
from src.chains import DocumentQAChain
from src.vectorstores import VectorStoreManager

# Initialize vector store
vector_manager = VectorStoreManager()
vector_store = vector_manager.get_or_create_vector_store()

# Create Q&A chain
qa_chain = DocumentQAChain(vector_store_manager=vector_manager)

# Ask question
result = qa_chain.invoke("What is the main content of the document?")
print(result["answer"])
```

#### Single SQL Query

```python
from src.agents import SQLAgent

# Initialize SQL agent
sql_agent = SQLAgent()

# Execute query
result = sql_agent.invoke("Show me the top 5 products by sales")
print(result["output"])
```

### Method 3: API Calls

#### Using cURL

```bash
# Document Q&A
curl -X POST "http://localhost:8000/qa/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"question": "What is discussed in the document?"}}'

# SQL Query
curl -X POST "http://localhost:8000/sql/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"question": "Show me total sales by product category"}}'
```

#### Using Python requests

```python
import requests

# Document Q&A
response = requests.post(
    "http://localhost:8000/qa/invoke",
    json={"input": {"question": "What is the main topic?"}}
)
print(response.json())

# SQL Query
response = requests.post(
    "http://localhost:8000/sql/invoke", 
    json={"input": {"question": "List all customers"}}
)
print(response.json())
```

## 🚀 All Startup Commands

### Development and Testing Commands

```bash
# 1. Build/Rebuild Vector Store
python src/main.py build                    # Build vector store
python src/main.py build --force-rebuild    # Force rebuild

# 2. Interactive Chat Sessions
python src/main.py chat                     # Default mode (document Q&A)
python src/main.py chat --mode doc          # Document Q&A mode
python src/main.py chat --mode sql          # SQL query mode
python src/main.py chat --use-workflow      # workflow mode

# 3. Vector Store Modes
python src/main.py chat                     # Default: Dynamic vector store (recommended)
python src/main.py chat --no-dynamic        # Traditional static mode

# 3. API Server Startup
python src/main.py server                   # Default (0.0.0.0:8000)
python src/main.py server --host 127.0.0.1 # Custom host
python src/main.py server --port 8080       # Custom port
python src/main.py server --debug           # Debug mode
python src/main.py server --host 0.0.0.0 --port 9000 --debug  # Full custom

# Show help
python src/main.py --help                   # General help
python src/main.py build --help             # Build command help
python src/main.py chat --help              # Chat command help
python src/main.py server --help            # Server command help
```

### Production Deployment Commands

```bash
# Production API server with custom configuration
python src/main.py server --host 0.0.0.0 --port 8000

# Using gunicorn for production (if installed)
gunicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Commands (if using Docker)

```bash
# Build Docker image
docker build -t langchain-rag-sql .

# Run container
docker run -p 8000:8000 langchain-rag-sql

# Run with environment file
docker run --env-file .env -p 8000:8000 langchain-rag-sql

# Run in background
docker run -d --name rag-agent -p 8000:8000 langchain-rag-sql
```

### Model-Specific Startup Examples

#### For OpenRouter Models

```bash
# Set environment and start
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_api_key
export DEFAULT_MODEL=gpt-4o-mini
python src/main.py chat

# Or with .env file configured
python src/main.py server
```

#### For Ollama Local Models

```bash
# Ensure Ollama is running first
ollama serve

# Pull a model if needed
ollama pull mistral

# Set environment and start
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=mistral
python src/main.py chat

# Start API server with local model
python src/main.py server
```

## 🔥 Dynamic Vector Store

### What is Dynamic Vector Store?

The Dynamic Vector Store is a revolutionary feature that automatically keeps your vector database in sync with your document directory. No more manual rebuilds when you add, modify, or remove documents!

### Key Features

#### 📁 File System Monitoring (Watchdog)
- **Real-time Detection**: Automatically detects when files are added, modified, or deleted
- **Supported Formats**: PDF, DOCX, DOC, TXT, MD, CSV, XLSX, XLS
- **Smart Processing**: Avoids duplicate processing and handles temporary files
- **Background Operation**: Non-blocking file monitoring that doesn't affect performance

#### 🛠️ MCP Integration (Model Context Protocol)
- **Standardized Access**: Uses MCP for secure filesystem operations
- **Permission Control**: Configurable directory access restrictions
- **Future-Proof**: Built on emerging industry standards

#### ⚡ Automatic Operations
- **Add Documents**: New files are automatically processed and added to vector store
- **Update Documents**: Modified files are re-processed (old version removed, new version added)
- **Remove Documents**: Deleted files are automatically removed from vector store
- **Incremental Updates**: Only processes changed files, not the entire directory

### Usage Examples

#### Default Mode (Recommended)
```bash
# Dynamic vector store enabled by default
python src/main.py chat

# This includes:
# ✅ File system monitoring
# ✅ MCP support  
# ✅ Automatic document updates
```

#### Traditional Mode (Optional)
```bash
# Use static vector store (legacy mode)
python src/main.py chat --no-dynamic

# Requires manual rebuild:
python src/main.py build --force-rebuild
```

#### Interactive Commands
When dynamic mode is active, you have additional commands:

```bash
# Check system status
> status
📊 Dynamic Vector Store Status:
   - File monitoring: Enabled
   - MCP support: Enabled
   - Tracked files: 5
   - Processing files: 0

# View tracked files
> files
📁 Tracked Files (5 files):
   ✅ /data/document/report.pdf (3 documents)
   ✅ /data/document/manual.docx (7 documents)

# Force sync with filesystem
> sync
🔄 Starting forced filesystem synchronization...
✅ Filesystem synchronization completed
```

### Real-World Workflow

1. **Start the System** (Dynamic mode is default):
   ```bash
   python src/main.py chat
   ```

2. **Add New Document**: Simply copy a PDF to `data/document/`
   - System automatically detects the new file
   - Processes and adds it to vector store
   - Ready for querying immediately!

3. **Modify Document**: Edit an existing document
   - System detects the change
   - Removes old version from vector store
   - Processes and adds updated version

4. **Remove Document**: Delete a file from directory
   - System detects deletion
   - Automatically removes associated vectors
   - Cleans up vector store

### Configuration

```env
# Enable MCP
MCP_ENABLED=true
MCP_FILESYSTEM_ENABLED=true

# Configure allowed directories
MCP_FILESYSTEM_ALLOWED_DIRECTORIES=["./data", "./vector_store", "./"]

# File watching settings
ENABLE_FILE_WATCHING=true
FILE_WATCH_DELAY=1
AUTO_SYNC_FILESYSTEM=true
```

### Performance Benefits

- **⚡ Faster Development**: No manual vector store rebuilds
- **🔄 Real-time Updates**: Documents available immediately after changes
- **💾 Efficient Processing**: Only processes changed files
- **🧹 Automatic Cleanup**: Removes outdated document vectors
- **📈 Scalable**: Handles large document collections efficiently

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM provider: `openrouter` or `ollama` | `openrouter` |
| `OPENROUTER_API_KEY` | **(Required for OpenRouter)** OpenRouter API key | `""` |
| `DEFAULT_MODEL` | **(OpenRouter)** Default model to use | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | **(Ollama)** Ollama service URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | **(Ollama)** Ollama model to use | `mistral` |
| `VECTOR_STORE_PATH` | Vector store path | `./vector_store` |
| `API_HOST` | API server host | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |

## 🧠 Technical Architecture

The project uses a modular design with core components:

- **`LLMFactory`**: Language model factory that dynamically creates and manages LLM instances from different providers (OpenRouter, Ollama) based on configuration
- **`VectorStoreManager`**: Handles document loading, chunking, vectorization, and FAISS storage management
- **`DocumentQAChain`**: Implements standard RAG-based document Q&A chain
- **`SQLAgent`**: Built using LangChain's `create_sql_agent`, understands natural language and generates/executes SQL queries
- **`ConversationMemoryManager`**: Manages conversation history across different modes and sessions
- **FastAPI Application**: Wraps core logic, provides RESTful API and LangServe interfaces

## 🆕 LangChain 0.3 New Features

- **Modular Architecture**: Functionality separated into different packages for better performance
- **Improved LCEL**: Clearer chain building syntax
- **Better Type Safety**: Enhanced type annotations
- **Faster Import Speed**: Reduced startup time
- **Improved Error Handling**: More detailed error messages

## 🧪 Manual Testing

### 1. Test Document Loading

```python
from src.document_loaders import DocumentLoaderManager

loader = DocumentLoaderManager()
documents = loader.load_documents_from_directory("./data")
print(f"Loaded {len(documents)} document chunks")
```

### 2. Test Vector Store

```python
from src.vectorstores import VectorStoreManager

vector_manager = VectorStoreManager(use_openai_embeddings=False)
vector_store = vector_manager.get_or_create_vector_store()

# Test retrieval
docs = vector_store.similarity_search("test query", k=3)
print(f"Retrieved {len(docs)} relevant documents")
```

### 3. Test Q&A Chain

```python
from src.chains import DocumentQAChain
from src.vectorstores import VectorStoreManager

vector_manager = VectorStoreManager(use_openai_embeddings=False)
vector_manager.get_or_create_vector_store()

qa_chain = DocumentQAChain(vector_store_manager=vector_manager)
result = qa_chain.invoke("What is the main content of the document?")
print(result["answer"])
```

## 📝 API Documentation

After starting the service, access the complete API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐛 Troubleshooting

### 1. Initialization Failure

**Issue**: `OPENROUTER_API_KEY` not set

**Solution**: Set correct API key in `.env` file

### 2. Document Loading Failure

**Issue**: Unsupported file format

**Solution**: Ensure document format is in supported list or convert to supported format

### 3. Memory Issues

**Issue**: Out of memory when processing large documents

**Solution**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters

### 4. API Call Timeout

**Issue**: OpenRouter API call timeout

**Solution**: Check network connection and API key validity

### 5. Ollama Connection Issues

**Issue**: Cannot connect to Ollama service

**Solution**: Ensure Ollama is running with `ollama serve` and the model is pulled

## 🤝 Contributing

Issues and Pull Requests are welcome to improve the project!

## 📄 License

This project is licensed under the MIT License.