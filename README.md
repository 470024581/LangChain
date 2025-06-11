# LangChain RAG & SQL Agent

一个基于 LangChain 构建的高级 RAG (Retrieval-Augmented Generation) 和 SQL Agent 系统。它不仅支持对多种格式的文档进行智能问答，还能通过自然语言与 SQL 数据库进行交互。

最新版本集成了对 **本地大语言模型 (通过 Ollama)** 的支持，允许在 OpenRouter 云端模型和本地模型之间灵活切换。

## 🚀 功能特性

- **最新 LangChain 版本**: 基于 LangChain `0.3+` 构建，性能更优。
- **🤖 双模智能**:
    - **RAG 问答**: 对 PDF, DOCX, TXT, MD, CSV, Excel 等多种格式文档进行智能检索和问答。
    - **SQL Agent**: 通过自然语言与 SQL 数据库进行交互、查询和分析。
- ** flexible LLM 支持**:
    - **☁️ 云端模型**: 默认通过 OpenRouter 支持多种主流模型 (GPT, Claude, Llama)。
    - **💻 本地模型**: 支持通过 **Ollama** 部署和使用本地模型 (如 Mistral, Llama 3)。
- **对话记忆**: 支持多轮对话的上下文记忆。
- **LCEL 链式结构**: 使用 LangChain Expression Language (LCEL) 构建清晰、可扩展的处理流程。
- **向量存储**: 使用 FAISS 进行高效的向量存储和检索。
- **Web API**: 基于 FastAPI 和 LangServe 提供完整的 API 服务。
- **多种部署模式**: 支持命令行、交互式会话和 API 服务三种模式。

## 🛠 安装和配置

### 1. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 如果您计划使用Ollama本地模型，请额外安装Ollama的依赖
pip install langchain-ollama
```

### 2. 环境配置

创建 `.env` 文件并根据您的需求配置环境变量。

#### 配置1: 使用 OpenRouter 云端模型 (默认)

这是最简单的开箱即用配置，使用 OpenRouter 提供的云端大语言模型。

```env
# 1. 设置LLM提供商为 "openrouter"
LLM_PROVIDER="openrouter"

# 2. 提供您的OpenRouter API密钥
OPENROUTER_API_KEY="sk-or-v1-..."

# 3. [可选] 指定默认模型
DEFAULT_MODEL="gpt-4o-mini"
```

#### 配置2: 使用 Ollama 本地模型

如果您在本地通过 Ollama 部署了模型（如 Mistral, Llama 3），可以使用此配置。

**前提**: 请确保您已安装并运行了 [Ollama](https://ollama.com/)。

```env
# 1. 设置LLM提供商为 "ollama"
LLM_PROVIDER="ollama"

# 2. [可选] Ollama服务的URL，如果不是默认的 http://localhost:11434
# OLLAMA_BASE_URL="http://localhost:11434"

# 3. [可选] 要使用的Ollama模型名称，必须是您本地已有的模型
# OLLAMA_MODEL="mistral"
```

### 3. 准备数据

- **文档**: 将需要进行问答的文档放入 `data/document/` 目录。
- **数据库**: 系统自带一个位于 `data/database/erp.db` 的示例 SQLite 数据库。

## 🎯 使用方法

### 方式一：命令行使用

#### 1. 构建向量存储

在首次运行或文档更新后，您需要构建或更新向量存储。

```bash
# 为 `data/document/` 下的文档构建向量存储
python src/main.py build

# 强制重建向量存储
python src/main.py build --force-rebuild
```

#### 2. 交互式对话

启动交互式命令行界面。

```bash
python src/main.py chat
```

系统启动后，您可以直接提问。它支持两种模式：

- **📄 文档问答模式 (doc)**: 默认模式，基于 RAG 回答您关于文档的问题。
- **🗃️ SQL查询模式 (sql)**: 基于自然语言与数据库进行交互。

**交互式命令**:
- `mode doc`: 切换到文档问答模式。
- `mode sql`: 切换到SQL查询模式。
- `clear`: 清空当前模式的对话记忆。
- `quit` 或 `exit`: 退出程序。

#### 3. 启动 API 服务

```bash
python src/main.py server
```

服务将在 `http://localhost:8000` 启动，并提供完整的 API 文档 (Swagger UI)。

### 方式二：API 调用

(API 端口和功能保持不变，此处省略，详情可查看服务启动后的 `/docs` 路径)

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `LLM_PROVIDER` | LLM提供商，`openrouter` 或 `ollama` | `openrouter` |
| `OPENROUTER_API_KEY` | **(OpenRouter模式下必填)** OpenRouter API密钥 | `""` |
| `DEFAULT_MODEL` | **(OpenRouter模式下)** 默认使用的模型 | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | **(Ollama模式下)** Ollama服务URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | **(Ollama模式下)** 使用的Ollama模型 | `mistral` |
| `VECTOR_STORE_PATH` | 向量存储路径 | `./vector_store` |
| `API_HOST` | API服务主机 | `0.0.0.0` |
| `API_PORT` | API服务端口 | `8000` |
| `CHUNK_SIZE` | 文档分块大小 | `1000` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `200` |

## 🧠 技术架构

项目采用模块化设计，核心组件包括：

- **`LLMFactory`**: 语言模型工厂，根据配置 (`LLM_PROVIDER`) 动态创建和管理来自不同提供商（OpenRouter, Ollama）的 LLM 实例。
- **`VectorStoreManager`**: 负责文档的加载、切分、向量化，并使用 FAISS 进行存储和管理。
- **`DocumentQAChain`**: 实现基于 RAG 的标准文档问答链。
- **`SQLAgent`**: 使用 LangChain 的 `create_sql_agent` 构建，能够理解自然语言并生成、执行 SQL 查询。
- **`ConversationMemoryManager`**: 管理不同模式、不同会话的对话历史。
- **FastAPI Application**: 封装核心逻辑，提供 RESTful API 和 LangServe 接口。

## 🆕 LangChain 0.3 新特性

- **模块化架构**: 功能分离到不同包中，提高性能
- **改进的 LCEL**: 更清晰的链构建语法
- **更好的类型安全**: 改进的类型注解
- **更快的导入速度**: 减少启动时间
- **改进的错误处理**: 更详细的错误信息

## 🧪 手动测试

### 1. 测试文档加载

```python
from src.document_loaders import DocumentLoaderManager

loader = DocumentLoaderManager()
documents = loader.load_documents_from_directory("./data")
print(f"加载了 {len(documents)} 个文档片段")
```

### 2. 测试向量存储

```python
from src.vectorstores import VectorStoreManager

vector_manager = VectorStoreManager(use_openai_embeddings=False)
vector_store = vector_manager.get_or_create_vector_store()

# 测试检索
docs = vector_store.similarity_search("测试查询", k=3)
print(f"检索到 {len(docs)} 个相关文档")
```

### 3. 测试问答链

```python
from src.chains import DocumentQAChain
from src.vectorstores import VectorStoreManager

vector_manager = VectorStoreManager(use_openai_embeddings=False)
vector_manager.get_or_create_vector_store()

qa_chain = DocumentQAChain(vector_store_manager=vector_manager)
result = qa_chain.invoke("文档的主要内容是什么？")
print(result["answer"])
```

## 📝 API 文档

启动服务后，访问以下地址查看完整的 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐛 常见问题

### 1. 初始化失败

**问题**: `OPENROUTER_API_KEY` 未设置

**解决**: 在 `.env` 文件中设置正确的 API 密钥

### 2. 文档加载失败

**问题**: 不支持的文件格式

**解决**: 确保文档格式在支持列表中，或转换为支持的格式

### 3. 内存不足

**问题**: 处理大文档时内存不足

**解决**: 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 参数

### 4. API 调用超时

**问题**: OpenRouter API 调用超时

**解决**: 检查网络连接和 API 密钥有效性

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📄 许可证

本项目采用 MIT 许可证。