"""
Multi-Agent Collaborative Workflow Based on LangGraph

Implements the following flow:
User Input → Router Agent → SQL/RAG Agent → Answer Agent → Review Agent → Output
"""

import logging
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from ..agents.rag_agent import DocumentQAAgent
from ..agents.sql_agent import SQLAgent
from ..models.llm_factory import LLMFactory
from ..vectorstores.vector_store import VectorStoreManager
from ..config.settings import settings
from ..utils.langsmith_utils import langsmith_manager, with_langsmith_tracing

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type enumeration"""
    SQL = "sql"
    RAG = "rag"
    UNKNOWN = "unknown"


class WorkflowState(TypedDict):
    """Workflow state definition"""
    # Basic information
    user_question: str
    session_id: str
    
    # Routing information
    query_type: QueryType
    router_reasoning: str
    
    # Retrieval results
    retrieval_result: Optional[Dict[str, Any]]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    
    # Answer generation
    generated_answer: str
    answer_reasoning: str
    
    # Review results
    review_score: float
    review_feedback: str
    review_approved: bool
    
    # Message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Iteration control
    iteration_count: int
    max_iterations: int


class RouterAgent:
    """Router Agent - Determines query type"""
    
    def __init__(self, llm=None):
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router responsible for determining whether a user question should be answered using SQL queries or document retrieval (RAG).

**Important Note**: Please carefully analyze the semantics and context of the question, especially paying attention to homonyms and polysemous words.

Decision Rules:
1. **SQL Query** - If the question involves:
   - Data statistics, calculations, aggregations (e.g., total, average, maximum, sorting)
   - Specific numerical queries (e.g., price, quantity, date range)
   - Structured data queries from database tables
   - Keywords: how many, statistics, calculate, ranking, compare, filter, find records

2. **Document Retrieval (RAG)** - If the question involves:
   - Concept explanations, definitions (e.g., what is..., introduce...)
   - Person introductions, entity descriptions (e.g., do you know..., who is...)
   - Process descriptions, method introductions
   - Unstructured text content
   - Knowledge-based Q&A
   - Keywords: what is, how to, why, explain, introduce, know, understand

**Special Note**:
- Prioritize the main intent of the question rather than literal meaning

Please analyze the user question and return in JSON format:
{{
    "query_type": "sql" or "rag",
    "reasoning": "Detailed reasoning including semantic analysis of the question",
    "confidence": confidence score from 0.0-1.0
}}"""),
            ("human", "User question: {question}")
        ])
    
    def route(self, question: str) -> Dict[str, Any]:
        """Routing decision"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})
            
            # Parse JSON result
            import json
            try:
                parsed = json.loads(result)
                query_type = QueryType(parsed.get("query_type", "unknown"))
                reasoning = parsed.get("reasoning", "")
                confidence = parsed.get("confidence", 0.5)
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, use keyword matching as fallback
                query_type, reasoning = self._fallback_routing(question)
                confidence = 0.6
            
            logger.info(f"Routing decision: {query_type.value} (confidence: {confidence})")
            return {
                "query_type": query_type,
                "reasoning": reasoning,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Routing decision failed: {str(e)}")
            # Use fallback routing
            query_type, reasoning = self._fallback_routing(question)
            return {
                "query_type": query_type,
                "reasoning": reasoning,
                "confidence": 0.3
            }
    
    def _fallback_routing(self, question: str) -> tuple[QueryType, str]:
        """Fallback routing logic"""
        sql_keywords = ["how many", "statistics", "calculate", "ranking", "compare", "filter", "find", "count", "price", "total", "average", "maximum", "minimum", "how much", "statistic", "calculation", "rank", "comparison", "screen", "search", "quantity", "price", "total number", "mean", "max", "min"]
        rag_keywords = ["what is", "how to", "why", "explain", "introduce", "definition", "concept", "method", "process", "know", "understand", "who is", "what", "how to do", "why", "explanation", "introduction", "define", "concept", "approach", "procedure", "recognize", "learn", "who"]
        
        question_lower = question.lower()
        
        sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
        rag_score = sum(1 for keyword in rag_keywords if keyword in question_lower)
        
        if sql_score > rag_score:
            return QueryType.SQL, f"Detected SQL-related keywords: {sql_score} matches"
        elif rag_score > sql_score:
            return QueryType.RAG, f"Detected RAG-related keywords: {rag_score} matches"
        else:
            return QueryType.RAG, "Default to RAG retrieval"


class AnswerAgent:
    """Answer Generation Agent"""
    
    def __init__(self, llm=None):
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional answer generation assistant. Your task is to generate or optimize final answers based on retrieved information.

**Important Principles**:
1. **Analyze Retrieval Results**: Carefully analyze the type and content of retrieval results
2. **Accurately Understand Question Intent**: Understand what the user really wants to know
3. **Answer Based on Facts**: Strictly base answers on specific retrieved information, do not fabricate content
4. **Intelligently Process Results**: 
   - If it's SQL query results, use the query results directly, no need to regenerate
   - If it's RAG retrieval results, optimize answers based on document content
   - If retrieval fails, clearly explain the reason

Processing Strategy:
- **SQL Query Results**: If retrieval information comes from SQL queries, use query results directly, just format appropriately
- **RAG Retrieval Results**: If retrieval information comes from document retrieval, generate or optimize answers based on actual content
- **Error Handling**: If retrieval fails, clearly explain the failure reason

Please generate a final answer based on the following information:"""),
            ("human", """User Question: {question}

Retrieval Information: {retrieval_info}

Please carefully analyze the type of retrieval information. If it's SQL query results, use them directly; if it's document retrieval results, generate an answer based on actual content.""")
        ])
    
    def generate_answer(self, question: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer"""
        try:
            # Format retrieval information
            retrieval_info = ""
            
            # Handle RAG Agent return results
            if "answer" in retrieval_result and "relevant_documents" in retrieval_result:
                # RAG Agent return structure
                rag_answer = retrieval_result.get("answer", "")
                relevant_docs = retrieval_result.get("relevant_documents", [])
                
                # Build retrieval information including RAG Agent answer and document content
                retrieval_info = f"RAG Retrieval Results:\n{rag_answer}\n\n"
                
                if relevant_docs:
                    docs_info = "\n".join([
                        f"Document {i+1}: {doc.get('content', '')[:300]}..."
                        for i, doc in enumerate(relevant_docs[:3])
                    ])
                    retrieval_info += f"Relevant Document Snippets:\n{docs_info}"
                    
            # Handle SQL Agent return results
            elif "answer" in retrieval_result and "success" in retrieval_result:
                # SQL Agent return structure
                sql_answer = retrieval_result.get("answer", "")
                success = retrieval_result.get("success", False)
                
                if success:
                    # SQL query successful, directly return SQL Agent answer without LLM reprocessing
                    logger.info(f"SQL query successful, directly returning SQL Agent answer: {sql_answer[:100]}...")
                    return {
                        "answer": sql_answer,
                        "reasoning": "SQL query successful, using SQL Agent answer directly",
                        "success": True
                    }
                else:
                    # SQL query failed
                    error_msg = retrieval_result.get("error", "Unknown error")
                    retrieval_info = f"SQL query failed: {error_msg}"
                    
            # Handle other cases
            elif "error" in retrieval_result:
                retrieval_info = f"Retrieval failed: {retrieval_result.get('error', 'Unknown error')}"
            else:
                retrieval_info = "No relevant retrieval information found"
            
            # If retrieval failed, directly return error information without LLM processing
            if "Retrieval failed" in retrieval_info or "SQL query failed" in retrieval_info:
                return {
                    "answer": retrieval_info,
                    "reasoning": "Retrieval failed, returning error information directly",
                    "success": False
                }
            
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "question": question,
                "retrieval_info": retrieval_info
            })
            
            return {
                "answer": answer,
                "reasoning": "Generated answer based on retrieval information",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                "answer": f"Sorry, an error occurred while generating the answer: {str(e)}",
                "reasoning": "Exception occurred during generation process",
                "success": False
            }


class ReviewAgent:
    """Review Agent - Evaluates answer quality"""
    
    def __init__(self, llm=None):
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional answer quality reviewer. You need to evaluate the quality of generated answers and provide improvement suggestions.

Evaluation Dimensions:
1. **Accuracy** (0-10 points): Is the answer accurate, are there factual errors, is it consistent with retrieval information
2. **Completeness** (0-10 points): Does the answer completely address the user's question
3. **Relevance** (0-10 points): Is the answer highly relevant to the question, does it address what the user really wants to know
4. **Clarity** (0-10 points): Is the answer clearly expressed and easy to understand
5. **Usefulness** (0-10 points): Is the answer practically helpful to the user

**Special Focus**:
- Does the answer correctly understand the user's question intent
- Is the answer based on actual retrieved information
- Are there cases where the answer doesn't match the retrieval information
- Are there concept confusions or entity recognition errors

Scoring Standards:
- 8-10 points: Excellent, answer is accurate and completely addresses the question
- 6-7 points: Good, basically answers the question but may have minor issues
- 4-5 points: Fair, partially answers the question but has obvious shortcomings
- 0-3 points: Poor, answer is incorrect or completely fails to address the question

Please return in JSON format:
{{
    "overall_score": total score (0-10),
    "dimension_scores": {{
        "accuracy": accuracy score,
        "completeness": completeness score,
        "relevance": relevance score,
        "clarity": clarity score,
        "usefulness": usefulness score
    }},
    "approved": true/false,
    "feedback": "Specific improvement suggestions, particularly pointing out issues",
    "strengths": "Strengths of the answer",
    "weaknesses": "Weaknesses and areas for improvement"
}}"""),
            ("human", """User Question: {question}

Generated Answer: {answer}

Please conduct a comprehensive evaluation of this answer, particularly focusing on whether the answer correctly understands and addresses the user's question.""")
        ])
    
    def review_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """Review answer"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "question": question,
                "answer": answer
            })
            
            # Parse JSON result
            import json
            try:
                parsed = json.loads(result)
                overall_score = parsed.get("overall_score", 5.0)
                approved = parsed.get("approved", overall_score >= 8.0)
                feedback = parsed.get("feedback", "")
                
                return {
                    "score": overall_score,
                    "approved": approved,
                    "feedback": feedback,
                    "detailed_scores": parsed.get("dimension_scores", {}),
                    "strengths": parsed.get("strengths", ""),
                    "weaknesses": parsed.get("weaknesses", ""),
                    "success": True
                }
                
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, use simple scoring
                return self._simple_review(answer)
                
        except Exception as e:
            logger.error(f"Answer review failed: {str(e)}")
            return self._simple_review(answer)
    
    def _simple_review(self, answer: str) -> Dict[str, Any]:
        """Simple review logic"""
        # Simple scoring based on answer length and keywords
        score = 5.0
        if len(answer) > 100:
            score += 1.0
        if len(answer) > 300:
            score += 1.0
        if "sorry" in answer.lower() or "error" in answer.lower() or "sorry" in answer or "error" in answer:
            score -= 2.0
        
        score = max(0.0, min(10.0, score))
        
        return {
            "score": score,
            "approved": score >= 8.0,
            "feedback": "Scoring based on simple rules",
            "detailed_scores": {},
            "strengths": "",
            "weaknesses": "",
            "success": True
        }


class MultiAgentWorkflow:
    """Multi-Agent Collaborative Workflow"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager = None,
        sql_db_path: str = None,
        model_name: str = None,
        max_iterations: int = 2
    ):
        """
        Initialize multi-agent workflow
        
        Args:
            vector_store_manager: Vector store manager
            sql_db_path: SQL database path
            model_name: Model name
            max_iterations: Maximum number of iterations
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.router_agent = RouterAgent(LLMFactory.create_llm(model_name))
        self.answer_agent = AnswerAgent(LLMFactory.create_llm(model_name))
        self.review_agent = ReviewAgent(LLMFactory.create_llm(model_name))
        
        # Initialize RAG and SQL agents
        if vector_store_manager is None:
            vector_store_manager = VectorStoreManager()
            vector_store_manager.get_or_create_vector_store()
        
        self.rag_agent = DocumentQAAgent(
            vector_store_manager=vector_store_manager,
            model_name=model_name,
            use_memory=True
        )
        
        
        # Initialize SQL Agent
        try:
            # Ensure correct database path is used
            if sql_db_path is None:
                from pathlib import Path
                base_path = Path(__file__).parent.parent.parent
                sql_db_path = str(base_path / "data" / "database" / "erp.db")
                logger.info(f"Using default database path: {sql_db_path}")
            
            self.sql_agent = SQLAgent(
                db_path=sql_db_path,
                model_name=model_name,
                use_memory=True,
                verbose=True
            )
            self.sql_available = True
            logger.info(f"SQLAgent initialized successfully in workflow, database path: {sql_db_path}")
        except Exception as e:
            logger.error(f"SQLAgent initialization failed in workflow: {str(e)}")
            self.sql_agent = None
            self.sql_available = False
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Multi-agent workflow initialization completed")
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("sql_retrieval", self._sql_retrieval_node)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("review", self._review_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "sql": "sql_retrieval",
                "rag": "rag_retrieval",
                "unknown": "rag_retrieval"  # Default to RAG
            }
        )
        
        # Retrieval to answer generation
        workflow.add_edge("sql_retrieval", "answer_generation")
        workflow.add_edge("rag_retrieval", "answer_generation")
        
        # Answer generation to review
        workflow.add_edge("answer_generation", "review")
        
        # Review conditional edges
        workflow.add_conditional_edges(
            "review",
            self._review_decision,
            {
                "approved": END,
                "retry": "answer_generation",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _router_node(self, state: WorkflowState) -> WorkflowState:
        """Router node"""
        logger.info("Executing routing decision...")
        
        routing_result = self.router_agent.route(state["user_question"])
        
        state["query_type"] = routing_result["query_type"]
        state["router_reasoning"] = routing_result["reasoning"]
        state["messages"].append(AIMessage(content=f"Routing decision: {routing_result['query_type'].value} - {routing_result['reasoning']}"))
        
        return state
    
    def _sql_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """SQL retrieval node"""
        logger.info("Executing SQL retrieval...")
        
        if not self.sql_available:
            state["retrieval_result"] = {
                "success": False,
                "error": "SQL Agent unavailable",
                "answer": "Sorry, SQL query functionality is currently unavailable."
            }
        else:
            # Use independent session ID for SQL Agent to avoid memory pollution
            sql_session_id = f"sql_{state['session_id']}"
            result = self.sql_agent.query(
                question=state["user_question"],
                session_id=sql_session_id
            )
            
            # Add debug logs
            logger.debug(f"SQL retrieval result structure: {list(result.keys())}")
            logger.debug(f"SQL query success: {result.get('success', False)}")
            logger.debug(f"SQL answer length: {len(result.get('answer', ''))}")
            if result.get('answer'):
                logger.debug(f"SQL answer first 200 characters: {result.get('answer', '')[:200]}...")
            
            state["retrieval_result"] = result
        
        state["messages"].append(AIMessage(content="SQL retrieval completed"))
        return state
    
    def _rag_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """RAG retrieval node"""
        logger.info("Executing RAG retrieval...")
        
        result = self.rag_agent.invoke(
            question=state["user_question"],
            session_id=state["session_id"]
        )
        
        # Add debug logs
        logger.debug(f"RAG retrieval result structure: {list(result.keys())}")
        logger.debug(f"RAG answer length: {len(result.get('answer', ''))}")
        logger.debug(f"Relevant documents count: {len(result.get('relevant_documents', []))}")
        
        state["retrieval_result"] = result
        state["retrieved_documents"] = result.get("retrieved_documents", [])
        state["messages"].append(AIMessage(content="RAG retrieval completed"))
        
        return state
    
    def _answer_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Answer generation node"""
        logger.info("Generating answer...")
        
        answer_result = self.answer_agent.generate_answer(
            question=state["user_question"],
            retrieval_result=state["retrieval_result"]
        )
        
        # Update state with the generated answer and reasoning
        state["generated_answer"] = answer_result.get("answer", "No answer could be generated.")
        state["answer_reasoning"] = answer_result.get("reasoning", "")
        
        logger.info(f"Generated answer: {state['generated_answer'][:100]}...")
        
        return state
    
    def _review_node(self, state: WorkflowState) -> WorkflowState:
        """Review the generated answer"""
        logger.info("Reviewing answer quality...")
        
        review_result = self.review_agent.review_answer(
            question=state["user_question"],
            answer=state["generated_answer"]
        )
        
        state["review_score"] = review_result["score"]
        state["review_feedback"] = review_result["feedback"]
        state["review_approved"] = review_result["approved"]
        state["messages"].append(AIMessage(content=f"Review completed - Score: {review_result['score']:.1f}"))
        
        return state
    
    def _route_decision(self, state: WorkflowState) -> str:
        """Route decision function"""
        query_type = state.get("query_type", QueryType.UNKNOWN)
        if query_type == QueryType.SQL and self.sql_available:
            return "sql"
        elif query_type == QueryType.RAG:
            return "rag"
        else:
            return "unknown"
    
    def _review_decision(self, state: WorkflowState) -> str:
        """Review decision function"""
        # If review passes, end process
        if state.get("review_approved", False):
            return "approved"
        
        # If maximum iterations reached, force end
        if state.get("iteration_count", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        
        # Otherwise regenerate answer
        return "retry"
    
    @with_langsmith_tracing(name="MultiAgentWorkflow.run", tags=["workflow", "multi-agent"])
    def run(
        self,
        question: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run multi-agent workflow
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing complete results
        """
        try:
            logger.info(f"Starting multi-agent workflow to process question: {question[:100]}...")
            
            # Initialize state
            initial_state = WorkflowState(
                user_question=question,
                session_id=session_id,
                query_type=QueryType.UNKNOWN,
                router_reasoning="",
                retrieval_result=None,
                retrieved_documents=None,
                generated_answer="",
                answer_reasoning="",
                review_score=0.0,
                review_feedback="",
                review_approved=False,
                messages=[HumanMessage(content=question)],
                iteration_count=0,
                max_iterations=self.max_iterations
            )
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Build return result
            result = {
                "question": question,
                "answer": final_state["generated_answer"],
                "query_type": final_state["query_type"].value,
                "router_reasoning": final_state["router_reasoning"],
                "review_score": final_state["review_score"],
                "review_feedback": final_state["review_feedback"],
                "review_approved": final_state["review_approved"],
                "iteration_count": final_state["iteration_count"],
                "retrieved_documents": final_state.get("retrieved_documents", []),
                "session_id": session_id,
                "success": True
            }
            
            logger.info(f"Workflow completed - Query type: {final_state['query_type'].value}, Review score: {final_state['review_score']:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent workflow execution failed: {str(e)}")
            return {
                "question": question,
                "answer": f"Workflow execution failed: {str(e)}",
                "query_type": "error",
                "router_reasoning": "",
                "review_score": 0.0,
                "review_feedback": "Exception occurred during execution",
                "review_approved": False,
                "iteration_count": 0,
                "retrieved_documents": [],
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    async def arun(
        self,
        question: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronously run multi-agent workflow
        
        Args:
            question: User question
            session_id: Session ID
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing complete results
        """
        try:
            logger.info(f"Starting asynchronous multi-agent workflow to process question: {question[:100]}...")
            
            # Initialize state
            initial_state = WorkflowState(
                user_question=question,
                session_id=session_id,
                query_type=QueryType.UNKNOWN,
                router_reasoning="",
                retrieval_result=None,
                retrieved_documents=None,
                generated_answer="",
                answer_reasoning="",
                review_score=0.0,
                review_feedback="",
                review_approved=False,
                messages=[HumanMessage(content=question)],
                iteration_count=0,
                max_iterations=self.max_iterations
            )
            
            # Run workflow asynchronously
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Build return result
            result = {
                "question": question,
                "answer": final_state["generated_answer"],
                "query_type": final_state["query_type"].value,
                "router_reasoning": final_state["router_reasoning"],
                "review_score": final_state["review_score"],
                "review_feedback": final_state["review_feedback"],
                "review_approved": final_state["review_approved"],
                "iteration_count": final_state["iteration_count"],
                "retrieved_documents": final_state.get("retrieved_documents", []),
                "session_id": session_id,
                "success": True
            }
            
            logger.info(f"Asynchronous workflow completed - Query type: {final_state['query_type'].value}, Review score: {final_state['review_score']:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Asynchronous multi-agent workflow execution failed: {str(e)}")
            return {
                "question": question,
                "answer": f"Workflow execution failed: {str(e)}",
                "query_type": "error",
                "router_reasoning": "",
                "review_score": 0.0,
                "review_feedback": "Exception occurred during execution",
                "review_approved": False,
                "iteration_count": 0,
                "retrieved_documents": [],
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get workflow information"""
        return {
            "workflow_type": "multi_agent",
            "max_iterations": self.max_iterations,
            "sql_available": self.sql_available,
            "agents": {
                "router": "RouterAgent",
                "rag": "DocumentQAAgent", 
                "sql": "SQLAgent" if self.sql_available else None,
                "answer": "AnswerAgent",
                "review": "ReviewAgent"
            },
            "model_name": self.model_name
        } 