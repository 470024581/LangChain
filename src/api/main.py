import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langserve import add_routes

from ..chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
from ..agents.sql_agent import SQLAgent
from ..workflows.multi_agent_workflow import MultiAgentWorkflow
from ..vectorstores.vector_store import VectorStoreManager
from ..memory.conversation_memory import SessionManager
from ..config.settings import settings
from ..utils.langsmith_utils import langsmith_manager, get_langsmith_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
vector_store_manager: Optional[VectorStoreManager] = None
qa_chain: Optional[DocumentQAChain] = None
conversational_chain: Optional[ConversationalRetrievalChain] = None
sql_agent: Optional[SQLAgent] = None
multi_agent_workflow: Optional[MultiAgentWorkflow] = None
session_manager: SessionManager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global vector_store_manager, qa_chain, conversational_chain, sql_agent, multi_agent_workflow
    
    logger.info("Initializing application...")
    
    try:
        # Initialize vector store manager
        vector_store_manager = VectorStoreManager(use_openai_embeddings=False)
        vector_store_manager.get_or_create_vector_store()
        
        # Initialize Q&A chain
        qa_chain = DocumentQAChain(
            vector_store_manager=vector_store_manager,
            use_memory=True
        )
        
        # Initialize conversational retrieval chain
        conversational_chain = ConversationalRetrievalChain(
            vector_store_manager=vector_store_manager
        )
        
        # Initialize SQL Agent
        try:
            sql_agent = SQLAgent(use_memory=True, verbose=False)
            logger.info("SQL Agent initialization completed")
        except Exception as e:
            logger.warning(f"SQL Agent initialization failed: {str(e)}")
            sql_agent = None
        
        # Initialize multi-agent workflow
        try:
            multi_agent_workflow = MultiAgentWorkflow(
                vector_store_manager=vector_store_manager,
                max_iterations=2
            )
            logger.info("Multi-agent workflow initialization completed")
        except Exception as e:
            logger.warning(f"Multi-agent workflow initialization failed: {str(e)}")
            multi_agent_workflow = None
        
        # Add LangServe routes
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
        
        logger.info("Application initialization completed")
        logger.info("LangServe routes added:")
        logger.info("  - /langserve/qa (Standard Q&A chain)")
        logger.info("  - /langserve/conversational (Conversational retrieval chain)")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        raise
    
    yield
    
    logger.info("Application shutdown")


# Create FastAPI application
app = FastAPI(
    title="Document Q&A System",
    description="Document Q&A API system built with LangChain",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QuestionRequest(BaseModel):
    """Question request model"""
    question: str = Field(..., description="User question")
    session_id: str = Field("default", description="Session ID")
    use_conversational: bool = Field(False, description="Whether to use conversational retrieval")


class QuestionResponse(BaseModel):
    """Question response model"""
    answer: str = Field(..., description="Answer")
    question: str = Field(..., description="Original question")
    session_id: str = Field(..., description="Session ID")
    relevant_documents: list = Field(default_factory=list, description="Relevant documents")
    standalone_question: Optional[str] = Field(None, description="Standalone question (conversational retrieval only)")


class MemoryStatsResponse(BaseModel):
    """Memory statistics response model"""
    session_id: str
    total_messages: int
    user_messages: int
    ai_messages: int
    total_characters: int
    memory_type: str


class SQLQueryRequest(BaseModel):
    """SQL query request model"""
    question: str = Field(..., description="Natural language question")
    session_id: str = Field("default", description="Session ID")


class SQLQueryResponse(BaseModel):
    """SQL query response model"""
    answer: str = Field(..., description="Query result")
    question: str = Field(..., description="Original question")
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether query was successful")
    intermediate_steps: list = Field(default_factory=list, description="Intermediate steps")
    error: Optional[str] = Field(None, description="Error message")


class WorkflowRequest(BaseModel):
    """Workflow request model"""
    question: str = Field(..., description="User question")
    session_id: str = Field(default="default", description="Session ID")


class WorkflowResponse(BaseModel):
    """Workflow response model"""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    query_type: str = Field(..., description="Query type (sql/rag)")
    router_reasoning: str = Field(..., description="Routing decision reasoning")
    review_score: float = Field(..., description="Review score")
    review_feedback: str = Field(..., description="Review feedback")
    review_approved: bool = Field(..., description="Whether review was approved")
    iteration_count: int = Field(..., description="Iteration count")
    retrieved_documents: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved documents")
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether execution was successful")
    error: Optional[str] = Field(None, description="Error message")


def get_qa_chain() -> DocumentQAChain:
    """Get Q&A chain instance"""
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="Q&A chain not initialized")
    return qa_chain


def get_conversational_chain() -> ConversationalRetrievalChain:
    """Get conversational retrieval chain instance"""
    if conversational_chain is None:
        raise HTTPException(status_code=500, detail="Conversational retrieval chain not initialized")
    return conversational_chain


def get_vector_store_manager() -> VectorStoreManager:
    """Get vector store manager instance"""
    if vector_store_manager is None:
        raise HTTPException(status_code=500, detail="Vector store manager not initialized")
    return vector_store_manager


def get_sql_agent() -> SQLAgent:
    """Get SQL Agent instance"""
    if sql_agent is None:
        raise HTTPException(status_code=500, detail="SQL Agent not initialized")
    return sql_agent


def get_multi_agent_workflow() -> MultiAgentWorkflow:
    """Get multi-agent workflow instance"""
    if multi_agent_workflow is None:
        raise HTTPException(status_code=500, detail="Multi-agent workflow not initialized")
    return multi_agent_workflow


# API Routes
@app.get("/")
async def root():
    """Root path"""
    return {"message": "Document Q&A System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check"""
    try:
        # Check component status
        qa_status = qa_chain is not None
        vector_status = vector_store_manager is not None and vector_store_manager.vector_store is not None
        sql_status = sql_agent is not None
        workflow_status = multi_agent_workflow is not None
        langsmith_status = langsmith_manager.is_enabled
        
        return {
            "status": "healthy" if qa_status and vector_status else "unhealthy",
            "qa_chain": "ready" if qa_status else "not ready",
            "vector_store": "ready" if vector_status else "not ready",
            "sql_agent": "ready" if sql_status else "not ready",
            "multi_agent_workflow": "ready" if workflow_status else "not ready",
            "langsmith": "enabled" if langsmith_status else "disabled"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/langsmith/config")
async def get_langsmith_config_endpoint():
    """Get LangSmith configuration information"""
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
    """Submit LangSmith feedback"""
    try:
        if not langsmith_manager.is_enabled:
            raise HTTPException(status_code=400, detail="LangSmith not enabled")
        
        langsmith_manager.log_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment
        )
        
        return {
            "status": "success",
            "message": "Feedback submitted"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit LangSmith feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """Handle Q&A request"""
    try:
        logger.info(f"Received question: {request.question[:100]}...")
        
        if request.use_conversational:
            # Use conversational retrieval chain
            result = conv_chain.invoke(
                question=request.question,
                session_id=request.session_id
            )
        else:
            # Use standard Q&A chain
            result = qa_chain.invoke(
                question=request.question,
                session_id=request.session_id
            )
        
        return QuestionResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to process question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@app.get("/documents/{question}", response_model=List[Dict])
async def get_relevant_documents(
    question: str, 
    k: int = 4, 
    vector_manager: VectorStoreManager = Depends(get_vector_store_manager)
):
    """Get relevant documents based on question"""
    try:
        retriever = vector_manager.get_retriever(k=k)
        documents = await retriever.ainvoke(question)
        return [doc.dict() for doc in documents]
    except Exception as e:
        logger.error(f"Failed to get relevant documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get relevant documents: {str(e)}")


@app.get("/memory/{session_id}/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(
    session_id: str,
    qa_chain: DocumentQAChain = Depends(get_qa_chain)
):
    """Get memory statistics"""
    try:
        stats = qa_chain.get_memory_stats(session_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Session does not exist")
        
        return MemoryStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory statistics: {str(e)}")


@app.delete("/memory/{session_id}")
async def clear_memory(
    session_id: str,
    qa_chain: DocumentQAChain = Depends(get_qa_chain)
):
    """Clear memory for specified session"""
    try:
        qa_chain.clear_memory(session_id)
        return {"message": f"Memory for session {session_id} has been cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@app.get("/memory/sessions")
async def get_sessions():
    """Get all session list"""
    try:
        sessions = session_manager.get_all_sessions()
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"Failed to get session list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session list: {str(e)}")


@app.post("/vector_store/rebuild")
async def rebuild_vector_store(
    force: bool = False,
    vector_manager: VectorStoreManager = Depends(get_vector_store_manager)
):
    """Rebuild vector store"""
    try:
        logger.info("Starting vector store rebuild...")
        await vector_manager.rebuild_vector_store(force=force)
        logger.info("Vector store rebuild completed")
        
        return {"message": "Vector store rebuild successful"}
        
    except Exception as e:
        logger.error(f"Failed to rebuild vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild vector store: {str(e)}")


@app.get("/debug/config")
async def debug_config():
    """Debug configuration information (for development debugging only)"""
    try:
        import os
        from ..config.settings import settings
        
        return {
            "settings": {
                "langchain_tracing_v2": settings.langchain_tracing_v2,
                "langchain_project": settings.langchain_project,
                "langchain_endpoint": settings.langchain_endpoint,
                "langsmith_api_key": settings.langsmith_api_key[:10] + "..." if settings.langsmith_api_key else None
            },
            "environment_variables": {
                "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
                "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"), 
                "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
                "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY")[:10] + "..." if os.getenv("LANGSMITH_API_KEY") else None
            }
        }
    except Exception as e:
        return {"error": str(e)}


# ===============================
# Evaluation Related Endpoints
# ===============================

@app.post("/evaluation/datasets/create")
async def create_evaluation_dataset(
    name: str,
    description: str = "",
    examples: List[Dict[str, Any]] = []
):
    """Create evaluation dataset"""
    try:
        from ..evaluation.datasets import EvaluationDataset, DatasetManager
        
        dataset = EvaluationDataset(name=name, description=description)
        if examples:
            dataset.add_examples_from_list(examples)
        
        dataset_manager = DatasetManager()
        file_path = dataset_manager.save_dataset(dataset)
        
        # If LangSmith is enabled, also upload to cloud
        dataset_id = None
        if langsmith_manager.is_enabled:
            dataset_id = dataset.upload_to_langsmith()
        
        return {
            "message": "Dataset created successfully",
            "dataset_name": name,
            "examples_count": len(dataset),
            "file_path": file_path,
            "langsmith_dataset_id": dataset_id
        }
        
    except Exception as e:
        logger.error(f"Failed to create evaluation dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")


@app.post("/evaluation/datasets/create-default")
async def create_default_evaluation_datasets():
    """Create default evaluation datasets"""
    try:
        from ..evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        dataset_manager.create_default_datasets()
        
        datasets = dataset_manager.list_datasets()
        
        return {
            "message": "Default datasets created successfully",
            "datasets": datasets
        }
        
    except Exception as e:
        logger.error(f"Failed to create default dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create default dataset: {str(e)}")


@app.get("/evaluation/datasets")
async def list_evaluation_datasets():
    """List all evaluation datasets"""
    try:
        from ..evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        datasets = dataset_manager.list_datasets()
        
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"Failed to get dataset list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset list: {str(e)}")


@app.post("/evaluation/run", response_model=WorkflowResponse)
async def run_evaluation(
    dataset_name: str,
    evaluator_types: List[str] = ["accuracy", "relevance", "helpfulness", "groundedness"],
    use_conversational: bool = False,
    max_concurrency: int = 3,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """Run model evaluation"""
    try:
        from ..evaluation.datasets import DatasetManager
        from ..evaluation.runners import EvaluationRunner, EvaluationManager
        
        # Load dataset
        dataset_manager = DatasetManager()
        dataset = dataset_manager.load_dataset(dataset_name)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' does not exist")
        
        # Create evaluation runner
        runner = EvaluationRunner(qa_chain=qa_chain, conversational_chain=conv_chain)
        
        # Run evaluation
        logger.info(f"Starting evaluation run: {dataset_name}")
        report = await runner.run_evaluation(
            dataset=dataset,
            evaluator_types=evaluator_types,
            use_conversational=use_conversational,
            max_concurrency=max_concurrency
        )
        
        # Save report
        eval_manager = EvaluationManager()
        report_file = eval_manager.save_report(report)
        
        return {
            "message": "Evaluation completed",
            "dataset_name": dataset_name,
            "total_examples": report.total_examples,
            "avg_scores": report.avg_scores,
            "execution_time": report.execution_time,
            "report_file": report_file
        }
        
    except Exception as e:
        logger.error(f"Failed to run evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/evaluation/run-langsmith")
async def run_langsmith_evaluation(
    dataset_name: str,
    experiment_name: str = None,
    evaluator_types: List[str] = ["accuracy", "relevance", "helpfulness"],
    use_conversational: bool = False,
    qa_chain: DocumentQAChain = Depends(get_qa_chain),
    conv_chain: ConversationalRetrievalChain = Depends(get_conversational_chain)
):
    """Run evaluation on LangSmith"""
    try:
        if not langsmith_manager.is_enabled:
            raise HTTPException(status_code=400, detail="LangSmith not enabled")
        
        from ..evaluation.runners import EvaluationRunner
        
        # Create evaluation runner
        runner = EvaluationRunner(qa_chain=qa_chain, conversational_chain=conv_chain)
        
        # Run evaluation on LangSmith
        experiment_name = runner.run_langsmith_evaluation(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            evaluator_types=evaluator_types,
            use_conversational=use_conversational
        )
        
        if not experiment_name:
            raise HTTPException(status_code=500, detail="LangSmith evaluation startup failed")
        
        return {
            "message": "LangSmith evaluation started",
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "evaluator_types": evaluator_types
        }
        
    except Exception as e:
        logger.error(f"LangSmith evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LangSmith evaluation failed: {str(e)}")


@app.get("/evaluation/reports")
async def list_evaluation_reports():
    """List all evaluation reports"""
    try:
        from ..evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        reports = eval_manager.list_reports()
        
        return {"reports": reports}
        
    except Exception as e:
        logger.error(f"Failed to get evaluation report list: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get report list: {str(e)}")


@app.get("/evaluation/reports/{report_file:path}")
async def get_evaluation_report(report_file: str):
    """Get evaluation report details"""
    try:
        from ..evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        report = eval_manager.load_report(report_file)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report file does not exist")
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get evaluation report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@app.get("/evaluation/summary")
async def get_evaluation_summary():
    """Get evaluation summary report"""
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
        logger.error(f"Failed to get evaluation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


# ===============================
# SQL Agent Related Endpoints
# ===============================

@app.post("/sql/query", response_model=SQLQueryResponse)
async def sql_query(
    request: SQLQueryRequest,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """Execute SQL query"""
    try:
        logger.info(f"Received SQL query request: {request.question[:100]}...")
        
        result = agent.query(
            question=request.question,
            session_id=request.session_id
        )
        
        return SQLQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"SQL query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQL query failed: {str(e)}")


@app.get("/sql/database/info")
async def get_database_info(
    agent: SQLAgent = Depends(get_sql_agent)
):
    """Get database information"""
    try:
        info = agent.get_database_info()
        return {
            "status": "success",
            "database_info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to get database information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get database information: {str(e)}")


@app.get("/sql/tables/{table_name}/sample")
async def get_table_sample(
    table_name: str,
    limit: int = 5,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """Get sample data from table"""
    try:
        sample = agent.get_sample_data(table_name, limit)
        return {
            "status": "success",
            "sample_data": sample
        }
        
    except Exception as e:
        logger.error(f"Failed to get table sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get table sample data: {str(e)}")


@app.delete("/sql/memory/{session_id}")
async def clear_sql_memory(
    session_id: str,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """Clear SQL session memory"""
    try:
        success = agent.clear_memory(session_id)
        return {
            "status": "success" if success else "failed",
            "message": f"Session {session_id} memory cleared" if success else f"Failed to clear session {session_id} memory"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear SQL memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")


@app.get("/sql/memory/{session_id}/stats")
async def get_sql_memory_stats(
    session_id: str,
    agent: SQLAgent = Depends(get_sql_agent)
):
    """Get SQL session memory statistics"""
    try:
        stats = agent.get_memory_stats(session_id)
        return {
            "status": "success",
            "memory_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get SQL memory statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory statistics: {str(e)}")


# ===============================
# Multi-Agent Workflow Related Endpoints
# ===============================

@app.post("/workflow/run", response_model=WorkflowResponse)
async def run_workflow(
    request: WorkflowRequest,
    workflow: MultiAgentWorkflow = Depends(get_multi_agent_workflow)
):
    """Run multi-agent workflow"""
    try:
        logger.info(f"Received workflow request: {request.question[:100]}...")
        
        result = workflow.run(
            question=request.question,
            session_id=request.session_id
        )
        
        return WorkflowResponse(**result)
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@app.get("/workflow/info")
async def get_workflow_info(
    workflow: MultiAgentWorkflow = Depends(get_multi_agent_workflow)
):
    """Get workflow information"""
    try:
        info = workflow.get_workflow_info()
        return {
            "status": "success",
            "workflow_info": info
        }
        
    except Exception as e:
        logger.error(f"Failed to get workflow information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow information: {str(e)}")


@app.delete("/workflow/memory/{session_id}")
async def clear_workflow_memory(
    session_id: str,
    workflow: MultiAgentWorkflow = Depends(get_multi_agent_workflow)
):
    """Clear workflow memory"""
    try:
        # Clear memory of RAG and SQL agents
        if workflow.rag_agent:
            workflow.rag_agent.clear_memory(session_id)
        if workflow.sql_agent:
            workflow.sql_agent.clear_memory(session_id)
        
        return {"message": f"Workflow session {session_id} memory has been cleared"}
        
    except Exception as e:
        logger.error(f"Failed to clear workflow memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear workflow memory: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    ) 