import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langserve import add_routes

from ..chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
from ..agents.sql_agent import SQLAgent
from ..vectorstores.vector_store import VectorStoreManager
from ..memory.conversation_memory import SessionManager
from ..config.settings import settings
from ..utils.langsmith_utils import langsmith_manager, get_langsmith_config

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
sql_agent: Optional[SQLAgent] = None
session_manager: SessionManager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vector_store_manager, qa_chain, conversational_chain, sql_agent
    
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
        
        # 初始化SQL Agent
        try:
            sql_agent = SQLAgent(use_memory=True, verbose=False)
            logger.info("SQL Agent初始化完成")
        except Exception as e:
            logger.warning(f"SQL Agent初始化失败: {str(e)}")
            sql_agent = None
        
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


class SQLQueryRequest(BaseModel):
    """SQL查询请求模型"""
    question: str = Field(..., description="自然语言问题")
    session_id: str = Field("default", description="会话ID")


class SQLQueryResponse(BaseModel):
    """SQL查询响应模型"""
    answer: str = Field(..., description="查询结果")
    question: str = Field(..., description="原始问题")
    session_id: str = Field(..., description="会话ID")
    success: bool = Field(..., description="查询是否成功")
    intermediate_steps: list = Field(default_factory=list, description="中间步骤")
    error: Optional[str] = Field(None, description="错误信息")


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


def get_sql_agent() -> SQLAgent:
    """获取SQL Agent实例"""
    if sql_agent is None:
        raise HTTPException(status_code=500, detail="SQL Agent未初始化")
    return sql_agent


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
        sql_status = sql_agent is not None
        langsmith_status = langsmith_manager.is_enabled
        
        return {
            "status": "healthy" if qa_status and vector_status else "unhealthy",
            "qa_chain": "ready" if qa_status else "not ready",
            "vector_store": "ready" if vector_status else "not ready",
            "sql_agent": "ready" if sql_status else "not ready",
            "langsmith": "enabled" if langsmith_status else "disabled"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/langsmith/config")
async def get_langsmith_config_endpoint():
    """获取 LangSmith 配置信息"""
    try:
        config = get_langsmith_config()
        return {
            "langsmith_config": config,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/langsmith/feedback")
async def submit_langsmith_feedback(
    run_id: str,
    key: str,
    score: float,
    comment: Optional[str] = None
):
    """提交 LangSmith 反馈"""
    try:
        if not langsmith_manager.is_enabled:
            raise HTTPException(status_code=400, detail="LangSmith 未启用")
        
        langsmith_manager.log_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment
        )
        
        return {
            "status": "success",
            "message": "反馈已提交"
        }
        
    except Exception as e:
        logger.error(f"提交 LangSmith 反馈失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


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


@app.get("/debug/config")
async def debug_config():
    """调试配置信息（仅用于开发调试）"""
    try:
        import os
        from ..config.settings import settings
        
        return {
            "settings": {
                "langchain_tracing_v2": settings.langchain_tracing_v2,
                "langchain_project": settings.langchain_project,
                "langchain_endpoint": settings.langchain_endpoint,
                "langchain_api_key": settings.langchain_api_key[:10] + "..." if settings.langchain_api_key else None
            },
            "environment_variables": {
                "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
                "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"), 
                "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
                "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY")[:10] + "..." if os.getenv("LANGCHAIN_API_KEY") else None
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ===============================
# 评估相关端点
# ===============================

@app.post("/evaluation/datasets/create")
async def create_evaluation_dataset(
    name: str,
    description: str = "",
    examples: List[Dict[str, Any]] = []
):
    """创建评估数据集"""
    try:
        from ..evaluation.datasets import EvaluationDataset, DatasetManager
        
        dataset = EvaluationDataset(name=name, description=description)
        if examples:
            dataset.add_examples_from_list(examples)
        
        dataset_manager = DatasetManager()
        file_path = dataset_manager.save_dataset(dataset)
        
        # 如果 LangSmith 启用，也上传到云端
        dataset_id = None
        if langsmith_manager.is_enabled:
            dataset_id = dataset.upload_to_langsmith()
        
        return {
            "message": "数据集创建成功",
            "dataset_name": name,
            "examples_count": len(dataset),
            "file_path": file_path,
            "langsmith_dataset_id": dataset_id
        }
        
    except Exception as e:
        logger.error(f"创建评估数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建数据集失败: {str(e)}")


@app.post("/evaluation/datasets/create-default")
async def create_default_evaluation_datasets():
    """创建默认评估数据集"""
    try:
        from ..evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        dataset_manager.create_default_datasets()
        
        datasets = dataset_manager.list_datasets()
        
        return {
            "message": "默认数据集创建成功",
            "datasets": datasets
        }
        
    except Exception as e:
        logger.error(f"创建默认数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建默认数据集失败: {str(e)}")


@app.get("/evaluation/datasets")
async def list_evaluation_datasets():
    """列出所有评估数据集"""
    try:
        from ..evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        datasets = dataset_manager.list_datasets()
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取数据集列表失败: {str(e)}")


@app.post("/evaluation/run")
async def run_evaluation(
    dataset_name: str,
    evaluator_types: List[str] = ["accuracy", "relevance", "helpfulness", "groundedness"],
    use_conversational: bool = False,
    max_concurrency: int = 3,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """运行模型评估"""
    try:
        from ..evaluation.datasets import DatasetManager
        from ..evaluation.runners import EvaluationRunner, EvaluationManager
        
        # 加载数据集
        dataset_manager = DatasetManager()
        dataset = dataset_manager.load_dataset(dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 '{dataset_name}' 不存在")
        
        # 创建评估运行器
        runner = EvaluationRunner(qa_chain=qa_chain, conversational_chain=conv_chain)
        
        # 运行评估
        logger.info(f"开始运行评估: {dataset_name}")
        report = await runner.run_evaluation(
            dataset=dataset,
            evaluator_types=evaluator_types,
            use_conversational=use_conversational,
            max_concurrency=max_concurrency
        )
        
        # 保存报告
        eval_manager = EvaluationManager()
        report_file = eval_manager.save_report(report)
        
        return {
            "message": "评估完成",
            "dataset_name": dataset_name,
            "total_examples": report.total_examples,
            "avg_scores": report.avg_scores,
            "execution_time": report.execution_time,
            "report_file": report_file
        }
        
    except Exception as e:
        logger.error(f"运行评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@app.post("/evaluation/run-langsmith")
async def run_langsmith_evaluation(
    dataset_name: str,
    experiment_name: str = None,
    evaluator_types: List[str] = ["accuracy", "relevance", "helpfulness"],
    use_conversational: bool = False,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """在 LangSmith 上运行评估"""
    try:
        if not langsmith_manager.is_enabled:
            raise HTTPException(status_code=400, detail="LangSmith 未启用")
        
        from ..evaluation.runners import EvaluationRunner
        
        # 创建评估运行器
        runner = EvaluationRunner(qa_chain=qa_chain, conversational_chain=conv_chain)
        
        # 在 LangSmith 上运行评估
        experiment_name = runner.run_langsmith_evaluation(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            evaluator_types=evaluator_types,
            use_conversational=use_conversational
        )
        
        if not experiment_name:
            raise HTTPException(status_code=500, detail="LangSmith 评估启动失败")
        
        return {
            "message": "LangSmith 评估已启动",
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "evaluator_types": evaluator_types
        }
        
    except Exception as e:
        logger.error(f"LangSmith 评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LangSmith 评估失败: {str(e)}")


@app.get("/evaluation/reports")
async def list_evaluation_reports():
    """列出所有评估报告"""
    try:
        from ..evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        reports = eval_manager.list_reports()
        
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"获取评估报告列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报告列表失败: {str(e)}")


@app.get("/evaluation/reports/{report_file:path}")
async def get_evaluation_report(report_file: str):
    """获取评估报告详情"""
    try:
        from ..evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        report = eval_manager.load_report(report_file)
        
        if not report:
            raise HTTPException(status_code=404, detail="报告文件不存在")
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"获取评估报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取报告失败: {str(e)}")


@app.get("/evaluation/summary")
async def get_evaluation_summary():
    """获取评估汇总报告"""
    try:
        from ..evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        report_files = eval_manager.list_reports()
        
        reports = []
        for report_file in report_files:
            report = eval_manager.load_report(report_file)
            if report:
                reports.append(report)
        
        summary = eval_manager.generate_summary_report(reports)
        
        return {
            "summary": summary,
            "total_report_files": len(report_files)
        }
        
    except Exception as e:
        logger.error(f"获取评估汇总失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取汇总失败: {str(e)}")


# ===============================
# SQL Agent相关端点
# ===============================

@app.post("/sql/query", response_model=SQLQueryResponse)
async def sql_query(
    request: SQLQueryRequest,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """执行SQL查询"""
    try:
        logger.info(f"收到SQL查询请求: {request.question[:100]}...")
        
        result = agent.query(
            question=request.question,
            session_id=request.session_id
        )
        
        return SQLQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"SQL查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL查询失败: {str(e)}")


@app.get("/sql/database/info")
async def get_database_info(
    agent: SQLAgent = Depends(get_sql_agent)
):
    """获取数据库信息"""
    try:
        info = agent.get_database_info()
        return {
            "status": "success",
            "database_info": info
        }
        
    except Exception as e:
        logger.error(f"获取数据库信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取数据库信息失败: {str(e)}")


@app.get("/sql/tables/{table_name}/sample")
async def get_table_sample(
    table_name: str,
    limit: int = 5,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """获取表的样例数据"""
    try:
        sample = agent.get_sample_data(table_name, limit)
        return {
            "status": "success",
            "sample_data": sample
        }
        
    except Exception as e:
        logger.error(f"获取表样例数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取表样例数据失败: {str(e)}")


@app.delete("/sql/memory/{session_id}")
async def clear_sql_memory(
    session_id: str,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """清除SQL会话记忆"""
    try:
        success = agent.clear_memory(session_id)
        return {
            "status": "success" if success else "failed",
            "message": f"会话 {session_id} 记忆已清除" if success else f"清除会话 {session_id} 记忆失败"
        }
        
    except Exception as e:
        logger.error(f"清除SQL记忆失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清除记忆失败: {str(e)}")


@app.get("/sql/memory/{session_id}/stats")
async def get_sql_memory_stats(
    session_id: str,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """获取SQL会话记忆统计"""
    try:
        stats = agent.get_memory_stats(session_id)
        return {
            "status": "success",
            "memory_stats": stats
        }
        
    except Exception as e:
        logger.error(f"获取SQL记忆统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取记忆统计失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 