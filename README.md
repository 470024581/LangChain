# 文档问答系统


基于 LangChain 0.3 构建的智能文档问答系统，支持多种文档格式、上下文记忆、LCEL 链式结构，并提供 Web API 服务。

## 🚀 功能特性

- **最新 LangChain 0.3**: 使用最新版本的 LangChain，获得更好的性能和功能
- **多格式文档支持**: PDF, DOCX, TXT, MD, CSV, Excel
- **智能问答**: 基于检索增强生成 (RAG) 的问答系统
- **对话记忆**: 支持多轮对话的上下文记忆
- **LCEL 链式结构**: 使用 LangChain Expression Language 构建清晰的处理链
- **向量存储**: FAISS 向量数据库存储和检索
- **Web API**: 基于 FastAPI 和 LangServe 的 API 服务
- **多种部署模式**: 命令行、交互式、API 服务

## 📋 版本要求

- Python 3.8+
- LangChain 0.3.7+
- 其他依赖请参考 `requirements.txt`

## 📁 项目结构

```
project_root/
├── data/                        # 文档数据目录
│   └── Long Liang.pdf          # 示例PDF文档
├── src/                        # 源代码目录
│   ├── config/                 # 配置管理
│   │   └── settings.py         # 应用设置
│   ├── document_loaders/       # 文档加载器
│   │   └── document_loader.py  # 多格式文档加载
│   ├── vectorstores/          # 向量存储
│   │   └── vector_store.py     # FAISS向量存储管理
│   ├── models/                 # 模型管理
│   │   └── llm_factory.py      # LLM工厂和管理器
│   ├── prompts/               # 提示词模板
│   │   └── prompt_templates.py # 各种场景的提示词
│   ├── memory/                # 记忆管理
│   │   └── conversation_memory.py # 对话记忆管理
│   ├── chains/                # 链式处理
│   │   └── qa_chain.py         # 问答链实现
│   ├── api/                   # API服务
│   │   └── main.py             # FastAPI应用
│   └── main.py                # 主入口文件
├── requirements.txt           # 项目依赖
├── UPGRADE_GUIDE.md          # LangChain 0.3 升级指南
└── README.md                 # 项目说明
```

## 🛠 安装和配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 环境配置

创建 `.env` 文件并配置必要的环境变量：

```bash
# OpenRouter API配置
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# 默认模型配置
DEFAULT_MODEL=gpt-4o-mini

# 向量数据库配置
VECTOR_STORE_PATH=./vector_store

# API服务配置
API_HOST=0.0.0.0
API_PORT=8000

# 文档处理配置
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 3. 准备文档

将需要问答的文档放入 `data/` 目录，支持以下格式：
- PDF (.pdf)
- Word 文档 (.docx, .doc)
- 文本文件 (.txt)
- Markdown (.md)
- CSV (.csv)
- Excel (.xlsx, .xls)

## 🎯 使用方法

### 方式一：命令行使用

#### 1. 构建向量存储

```bash
# 使用 HuggingFace embeddings（推荐，免费）
python src/main.py build

# 强制重建向量存储
python src/main.py build --force-rebuild

# 使用 OpenAI embeddings（需要API密钥）
python src/main.py build --use-openai-embeddings
```

#### 2. 交互式问答

```bash
python src/main.py interactive
```

交互式模式支持以下命令：
- 输入问题开始对话
- 输入 `clear` 清空记忆
- 输入 `quit` 或 `exit` 退出

#### 3. 启动 API 服务

```bash
python src/main.py server
```

服务将在 `http://localhost:8000` 启动。

### 方式二：API 调用

#### 1. 问答接口

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "文档的主要内容是什么？",
       "session_id": "user123",
       "use_conversational": false
     }'
```

#### 2. 获取相关文档

```bash
curl "http://localhost:8000/documents/文档主要内容?k=3"
```

#### 3. 记忆管理

```bash
# 获取记忆统计
curl "http://localhost:8000/memory/user123/stats"

# 清空记忆
curl -X DELETE "http://localhost:8000/memory/user123"

# 获取所有会话
curl "http://localhost:8000/memory/sessions"
```

#### 4. 向量存储管理

```bash
# 重建向量存储
curl -X POST "http://localhost:8000/vector_store/rebuild?force=true"
```

### 方式三：LangServe 接口

系统还提供了 LangServe 标准接口：

```bash
# 标准问答链
curl -X POST "http://localhost:8000/langserve/qa/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": "你的问题"}'

# 对话式检索链
curl -X POST "http://localhost:8000/langserve/conversational/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": "你的问题"}'
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | 必填 |
| `OPENROUTER_API_BASE` | OpenRouter API 基础URL | `https://openrouter.ai/api/v1` |
| `DEFAULT_MODEL` | 默认使用的模型 | `gpt-4o-mini` |
| `VECTOR_STORE_PATH` | 向量存储路径 | `./vector_store` |
| `API_HOST` | API服务主机 | `0.0.0.0` |
| `API_PORT` | API服务端口 | `8000` |
| `CHUNK_SIZE` | 文档分块大小 | `1000` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `200` |

### 支持的模型

系统通过 OpenRouter 支持多种模型：

- **OpenAI**: gpt-4o-mini, gpt-4o, gpt-4-turbo
- **Anthropic**: claude-3-sonnet, claude-3-haiku
- **Meta**: llama-3.1-8b, llama-3.1-70b

## 🧠 技术架构

### 核心组件

1. **文档加载器** (`DocumentLoaderManager`)
   - 支持多种文档格式
   - 自动文档分块和元数据管理

2. **向量存储** (`VectorStoreManager`)
   - FAISS 向量数据库
   - 支持 OpenAI 和 HuggingFace embeddings
   - 持久化存储和检索

3. **LLM 管理** (`LLMFactory`)
   - OpenRouter 接口集成
   - 模型切换和重试机制

4. **提示词管理** (`PromptTemplateManager`)
   - 多种场景的提示词模板
   - 对话历史格式化

5. **记忆管理** (`ConversationMemoryManager`)
   - 多会话记忆支持
   - 窗口和缓冲记忆模式

6. **问答链** (`DocumentQAChain`)
   - LCEL 构建的处理链
   - 检索增强生成 (RAG)
   - 流式输出支持

### LCEL 链结构

```python
# 标准问答链 (LangChain 0.3 语法)
chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": get_chat_history
    })
    | prompt
    | llm
    | StrOutputParser()
)
```

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