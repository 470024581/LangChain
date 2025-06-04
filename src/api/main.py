import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langserve import add_routes

from ..chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
from ..vectorstores.vector_store import VectorStoreManager
from ..memory.conversation_memory import SessionManager
from ..config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
vector_store_manager: Optional[VectorStoreManager] = None
qa_chain: Optional[DocumentQAChain] = None
conversational_chain: Optional[ConversationalRetrievalChain] = None
session_manager: SessionManager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vector_store_manager, qa_chain, conversational_chain
    
    logger.info("初始化应用...")
    
    try:
        # 初始化向量存储管理器
        vector_store_manager = VectorStoreManager(use_openai_embeddings=False)
        vector_store_manager.get_or_create_vector_store()
        
        # 初始化问答链
        qa_chain = DocumentQAChain(
            vector_store_manager=vector_store_manager,
            use_memory=True
        )
        
        # 初始化对话式检索链
        conversational_chain = ConversationalRetrievalChain(
            vector_store_manager=vector_store_manager
        )
        
        # 添加LangServe路由
        add_routes(
            app,
            qa_chain.chain,
            path="/langserve/qa",
            enabled_endpoints=["invoke", "stream", "stream_log", "playground", "input_schema", "output_schema", "config_schema"]
        )
        
        add_routes(
            app,
            conversational_chain.qa_chain,
            path="/langserve/conversational",
            enabled_endpoints=["invoke", "stream", "stream_log", "playground", "input_schema", "output_schema", "config_schema"]
        )
        
        logger.info("应用初始化完成")
        logger.info("LangServe路由已添加:")
        logger.info("  - /langserve/qa (标准问答链)")
        logger.info("  - /langserve/conversational (对话式检索链)")
        
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        raise
    
    yield
    
    logger.info("应用关闭")


# 创建FastAPI应用
app = FastAPI(
    title="文档问答系统",
    description="基于LangChain构建的文档问答API系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic模型
class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., description="用户问题")
    session_id: str = Field("default", description="会话ID")
    use_conversational: bool = Field(False, description="是否使用对话式检索")


class QuestionResponse(BaseModel):
    """问题响应模型"""
    answer: str = Field(..., description="答案")
    question: str = Field(..., description="原始问题")
    session_id: str = Field(..., description="会话ID")
    relevant_documents: list = Field(default_factory=list, description="相关文档")
    standalone_question: Optional[str] = Field(None, description="独立问题（仅对话式检索）")


class MemoryStatsResponse(BaseModel):
    """记忆统计响应模型"""
    session_id: str
    total_messages: int
    user_messages: int
    ai_messages: int
    total_characters: int
    memory_type: str


def get_qa_chain() -> DocumentQAChain:
    """获取问答链实例"""
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="问答链未初始化")
    return qa_chain


def get_conversational_chain() -> ConversationalRetrievalChain:
    """获取对话式检索链实例"""
    if conversational_chain is None:
        raise HTTPException(status_code=500, detail="对话式检索链未初始化")
    return conversational_chain


def get_vector_store_manager() -> VectorStoreManager:
    """获取向量存储管理器实例"""
    if vector_store_manager is None:
        raise HTTPException(status_code=500, detail="向量存储管理器未初始化")
    return vector_store_manager


# API路由
@app.get("/")
async def root():
    """根路径"""
    return {"message": "文档问答系统API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查各组件状态
        qa_status = qa_chain is not None
        vector_status = vector_store_manager is not None and vector_store_manager.vector_store is not None
        
        return {
            "status": "healthy" if qa_status and vector_status else "unhealthy",
            "qa_chain": "ready" if qa_status else "not ready",
            "vector_store": "ready" if vector_status else "not ready"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """处理问答请求"""
    try:
        logger.info(f"收到问题: {request.question[:100]}...")
        
        if request.use_conversational:
            # 使用对话式检索链
            result = conv_chain.invoke(
                question=request.question,
                session_id=request.session_id
            )
        else:
            # 使用标准问答链
            result = qa_chain.invoke(
                question=request.question,
                session_id=request.session_id
            )
        
        return QuestionResponse(**result)
        
    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理问题失败: {str(e)}")


@app.get("/documents/{question}")
async def get_relevant_documents(
    question: str,
    k: int = 4,
    vector_manager: VectorStoreManager = Depends(get_vector_store_manager)
):
    """获取相关文档"""
    try:
        retriever = vector_manager.get_retriever(k=k)
        documents = retriever.get_relevant_documents(question)
        
        return {
            "question": question,
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        }
    except Exception as e:
        logger.error(f"获取相关文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取相关文档失败: {str(e)}")


@app.get("/memory/{session_id}/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    session_id: str,
    qa_chain: DocumentQAChain = Depends(get_qa_chain)
):
    """获取记忆统计信息"""
    try:
        stats = qa_chain.get_memory_stats(session_id)
        if not stats:
            raise HTTPException(status_code=404, detail="会话不存在")
        
        return MemoryStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"获取记忆统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取记忆统计失败: {str(e)}")


@app.delete("/memory/{session_id}")
async def clear_memory(
    session_id: str,
    qa_chain: DocumentQAChain = Depends(get_qa_chain)
):
    """清空指定会话的记忆"""
    try:
        qa_chain.clear_memory(session_id)
        return {"message": f"会话 {session_id} 的记忆已清空"}
        
    except Exception as e:
        logger.error(f"清空记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空记忆失败: {str(e)}")


@app.get("/memory/sessions")
async def get_sessions():
    """获取所有会话列表"""
    try:
        sessions = session_manager.get_all_sessions()
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"获取会话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.post("/vector_store/rebuild")
async def rebuild_vector_store(
    force: bool = False,
    vector_manager: VectorStoreManager = Depends(get_vector_store_manager)
):
    """重建向量存储"""
    try:
        logger.info("开始重建向量存储...")
        vector_manager.get_or_create_vector_store(force_recreate=force)
        logger.info("向量存储重建完成")
        
        return {"message": "向量存储重建成功"}
        
    except Exception as e:
        logger.error(f"重建向量存储失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重建向量存储失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 