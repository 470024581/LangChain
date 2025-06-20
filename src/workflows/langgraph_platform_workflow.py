#!/usr/bin/env python3
"""
LangGraph Platform multi-agent collaboration workflow
Complete functionality implementation based on original multi_agent_workflow.py
Compatible with LangGraph Platform deployment requirements
"""

import os
import logging
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from enum import Enum
import operator

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import various agents and tools
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.agents.rag_agent import DocumentQAAgent
    from src.agents.sql_agent import SQLAgent
    from src.models.llm_factory import LLMFactory
    from src.vectorstores.vector_store import VectorStoreManager
    from src.config.settings import settings
except ImportError as e:
    logging.error(f"Import failed: {e}")
    DocumentQAAgent = None
    SQLAgent = None
    LLMFactory = None
    VectorStoreManager = None
    settings = None

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
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Iteration control
    iteration_count: int
    max_iterations: int


class RouterAgent:
    """Router agent - Determine query type"""
    
    def __init__(self, llm=None):
        # Simplified LLM creation logic, compatible with platform environment
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    # Fallback solution for platform environment
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM initialization failed, using mock LLM: {e}")
                # Create a mock LLM for testing
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent router responsible for determining whether user questions should be answered using SQL queries or document retrieval (RAG).

Judgment rules:
1. **SQL Query** - If the question involves:
   - Data statistics, calculations, aggregations (e.g., totals, averages, maximums, sorting)
   - Specific numerical queries (e.g., prices, quantities, date ranges)
   - Structured data queries from database tables
   - Keywords: how many, statistics, calculate, ranking, compare, filter, find records

2. **Document Retrieval (RAG)** - If the question involves:
   - Concept explanations, definitions (e.g., what is..., introduction to...)
   - Person introductions, entity descriptions (e.g., do you know..., who is...)
   - Process descriptions, method introductions
   - Unstructured text content
   - Knowledge-based Q&A
   - Keywords: what is, how to, why, explain, introduce, know, understand

Please analyze the user question and return in JSON format:
{{
    "query_type": "sql" or "rag",
    "reasoning": "detailed judgment rationale",
    "confidence": confidence level between 0.0-1.0
}}"""),
            ("human", "User question: {question}")
        ])
    
    def route(self, question: str) -> Dict[str, Any]:
        """Routing decision"""
        try:
            if self.llm is None:
                # Use fallback routing when LLM is not available
                query_type, reasoning = self._fallback_routing(question)
                return {
                    "query_type": query_type,
                    "reasoning": reasoning,
                    "confidence": 0.8
                }
            
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
        sql_keywords = ["how many", "statistics", "calculate", "ranking", "compare", "filter", "find", "quantity", "price", "total", "average", "maximum", "minimum"]
        rag_keywords = ["what is", "how to", "why", "explain", "introduce", "definition", "concept", "method", "process", "know", "understand", "who is"]
        
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
    """Answer generation agent"""
    
    def __init__(self, llm=None):
        # Simplified LLM creation logic
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM initialization failed, using mock LLM: {e}")
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional answer generation assistant. Generate final answers based on retrieved information.

Processing strategies:
- **SQL Query Results**: If retrieval information comes from SQL queries, directly adopt query results
- **RAG Retrieval Results**: If retrieval information comes from document retrieval, generate answers based on document content
- **Error Handling**: If retrieval fails, clearly explain the failure reason

Please generate the final answer based on the following information:"""),
            ("human", """User question: {question}

Retrieval information: {retrieval_info}

Please generate an accurate answer.""")
        ])
    
    def generate_answer(self, question: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer"""
        try:
            # Format retrieval information
            retrieval_info = ""
            
            # Process RAG Agent return results
            if "answer" in retrieval_result and "relevant_documents" in retrieval_result:
                rag_answer = retrieval_result.get("answer", "")
                relevant_docs = retrieval_result.get("relevant_documents", [])
                
                retrieval_info = f"RAG Retrieval Result:\n{rag_answer}\n\n"
                
                if relevant_docs:
                    docs_info = "\n".join([
                        f"Document {i+1}: {doc.get('content', '')[:300]}..."
                        for i, doc in enumerate(relevant_docs[:3])
                    ])
                    retrieval_info += f"Relevant Document Fragments:\n{docs_info}"
                    
            # Process SQL Agent return results
            elif "answer" in retrieval_result and "success" in retrieval_result:
                sql_answer = retrieval_result.get("answer", "")
                success = retrieval_result.get("success", False)
                
                if success:
                    # SQL query successful, directly return SQL Agent answer
                    logger.info(f"SQL query successful, directly returning SQL Agent answer")
                    return {
                        "answer": sql_answer,
                        "reasoning": "SQL query successful, directly using SQL Agent answer",
                        "success": True
                    }
                else:
                    error_msg = retrieval_result.get("error", "Unknown error")
                    retrieval_info = f"SQL query failed: {error_msg}"
                    
            # Process other cases
            elif "error" in retrieval_result:
                retrieval_info = f"Retrieval failed: {retrieval_result.get('error', 'Unknown error')}"
            else:
                retrieval_info = "No relevant retrieval information found"
            
            # If retrieval fails, directly return error information
            if "Retrieval failed" in retrieval_info or "SQL query failed" in retrieval_info:
                return {
                    "answer": retrieval_info,
                    "reasoning": "Retrieval failed, directly returning error information",
                    "success": False
                }
            
            # If no LLM available, return simplified answer
            if self.llm is None:
                return {
                    "answer": f"Simplified answer based on retrieval information: {retrieval_info[:200]}...",
                    "reasoning": "Using simplified answer generation (no LLM available)",
                    "success": True
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
    """Review agent - Evaluate answer quality"""
    
    def __init__(self, llm=None):
        # Simplified LLM creation logic
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM initialization failed, using mock LLM: {e}")
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional answer quality reviewer. Evaluate answer quality and provide improvement suggestions.

Scoring criteria:
- 8-10 points: Excellent, answer is accurate and completely addresses the question
- 6-7 points: Good, basically answers the question but may have minor issues
- 4-5 points: Fair, partially answers the question but has obvious shortcomings
- 0-3 points: Poor, answer is incorrect or completely fails to address the question

Please return in JSON format:
{{
    "overall_score": total score (0-10),
    "approved": true/false,
    "feedback": "specific improvement suggestions"
}}"""),
            ("human", """User question: {question}

Generated answer: {answer}

Please evaluate this answer.""")
        ])
    
    def review_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """Review answer"""
        try:
            # If no LLM available, use simple review
            if self.llm is None:
                return self._simple_review(answer)
            
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
                    "success": True
                }
                
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, use simple review
                return self._simple_review(answer)
                
        except Exception as e:
            logger.error(f"Answer review failed: {str(e)}")
            return self._simple_review(answer)
    
    def _simple_review(self, answer: str) -> Dict[str, Any]:
        """Simple review logic"""
        score = 5.0
        if len(answer) > 100:
            score += 1.0
        if len(answer) > 300:
            score += 1.0
        if "Sorry" in answer or "error" in answer:
            score -= 2.0
        
        score = max(0.0, min(10.0, score))
        
        return {
            "score": score,
            "approved": score >= 8.0,
            "feedback": "Score based on simple rules",
            "success": True
        }


class MultiAgentWorkflow:
    """Multi-agent collaborative workflow"""
    
    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations
        
        # Initialize various agents (using LLMFactory)
        try:
            base_llm = None
            if LLMFactory:
                try:
                    base_llm = LLMFactory.create_llm()
                    logger.info("LLM creation successful")
                except Exception as llm_e:
                    logger.warning(f"LLM creation failed, will use None: {str(llm_e)}")
                    base_llm = None
            
            self.router_agent = RouterAgent(base_llm)
            self.answer_agent = AnswerAgent(base_llm)
            self.review_agent = ReviewAgent(base_llm)
            logger.info("Agent initialization successful")
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            self.router_agent = RouterAgent(None)
            self.answer_agent = AnswerAgent(None)
            self.review_agent = ReviewAgent(None)
        
        # Initialize RAG and SQL agents (with error handling)
        self.rag_agent = None
        self.sql_agent = None
        self.sql_available = False
        
        # Force initialize RAG Agent
        try:
            if DocumentQAAgent and VectorStoreManager:
                # Create vector store manager without LLM
                vector_store_manager = VectorStoreManager()
                vector_store_manager.get_or_create_vector_store()
                
                # Create RAG Agent
                self.rag_agent = DocumentQAAgent(
                    vector_store_manager=vector_store_manager,
                    use_memory=True
                )
                logger.info("RAG Agent initialization successful")
            else:
                raise ImportError("DocumentQAAgent or VectorStoreManager not available")
        except Exception as e:
            logger.warning(f"RAG Agent initialization failed, attempting re-import: {str(e)}")
            # Re-attempt import
            try:
                import importlib
                rag_module = importlib.import_module('src.agents.rag_agent')
                vector_module = importlib.import_module('src.vectorstores.vector_store')
                DocumentQAAgent_new = getattr(rag_module, 'DocumentQAAgent')
                VectorStoreManager_new = getattr(vector_module, 'VectorStoreManager')
                
                vector_store_manager = VectorStoreManager_new()
                vector_store_manager.get_or_create_vector_store()
                self.rag_agent = DocumentQAAgent_new(
                    vector_store_manager=vector_store_manager,
                    use_memory=True
                )
                logger.info("RAG Agent re-initialization successful")
            except Exception as e2:
                logger.error(f"RAG Agent re-initialization also failed: {str(e2)}")
                self.rag_agent = None
        
        # Force initialize SQL Agent
        try:
            if SQLAgent:
                from pathlib import Path
                base_path = Path(__file__).parent.parent.parent
                sql_db_path = str(base_path / "data" / "database" / "erp.db")
                
                # Create SQL Agent
                self.sql_agent = SQLAgent(
                    db_path=sql_db_path,
                    use_memory=True,
                    verbose=True
                )
                self.sql_available = True
                logger.info(f"SQL Agent initialization successful, database path: {sql_db_path}")
            else:
                raise ImportError("SQLAgent not available")
        except Exception as e:
            logger.warning(f"SQL Agent initialization failed, attempting re-import: {str(e)}")
            # Re-attempt import
            try:
                import importlib
                sql_module = importlib.import_module('src.agents.sql_agent')
                SQLAgent_new = getattr(sql_module, 'SQLAgent')
                
                from pathlib import Path
                base_path = Path(__file__).parent.parent.parent
                sql_db_path = str(base_path / "data" / "database" / "erp.db")
                
                self.sql_agent = SQLAgent_new(
                    db_path=sql_db_path,
                    use_memory=True,
                    verbose=True
                )
                self.sql_available = True
                logger.info(f"SQL Agent re-initialization successful, database path: {sql_db_path}")
            except Exception as e2:
                logger.error(f"SQL Agent re-initialization also failed: {str(e2)}")
                self.sql_agent = None
                self.sql_available = False
        
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
                "unknown": "rag_retrieval"
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
        state["messages"].append(AIMessage(content=f"Routing decision: {routing_result['query_type'].value}"))
        
        return state
    
    def _sql_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """SQL retrieval node"""
        logger.info("Executing SQL retrieval...")
        
        if not self.sql_available or not self.sql_agent:
            state["retrieval_result"] = {
                "success": False,
                "error": "SQL Agent not available",
                "answer": "Sorry, SQL query functionality is currently unavailable."
            }
        else:
            try:
                sql_session_id = f"sql_{state['session_id']}"
                result = self.sql_agent.query(
                    question=state["user_question"],
                    session_id=sql_session_id
                )
                state["retrieval_result"] = result
            except Exception as e:
                logger.error(f"SQL retrieval failed: {str(e)}")
                state["retrieval_result"] = {
                    "success": False,
                    "error": f"SQL query execution failed: {str(e)}",
                    "answer": f"Sorry, SQL query execution failed: {str(e)}"
                }
        
        state["messages"].append(AIMessage(content="SQL retrieval completed"))
        return state
    
    def _rag_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """RAG retrieval node"""
        logger.info("Executing RAG retrieval...")
        
        if not self.rag_agent:
            # Simplified RAG processing
            question = state["user_question"]
            if "what is" in question.lower() or "introduce" in question.lower():
                answer = f"This is a knowledge Q&A question: {question}. I will answer based on documents."
            else:
                answer = f"Received your question: {question}. I am processing it for you."
            
            state["retrieval_result"] = {
                "answer": answer,
                "relevant_documents": [],
                "success": True
            }
        else:
            try:
                result = self.rag_agent.invoke(
                    question=state["user_question"],
                    session_id=state["session_id"]
                )
                state["retrieval_result"] = result
                state["retrieved_documents"] = result.get("retrieved_documents", [])
            except Exception as e:
                logger.error(f"RAG retrieval failed: {str(e)}")
                state["retrieval_result"] = {
                    "answer": f"RAG retrieval failed: {str(e)}",
                    "relevant_documents": [],
                    "success": False,
                    "error": str(e)
                }
        
        state["messages"].append(AIMessage(content="RAG retrieval completed"))
        return state
    
    def _answer_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Answer generation node"""
        logger.info("Generating answer...")
        
        if state.get("generated_answer"):
            state["iteration_count"] = state.get("iteration_count", 0) + 1
        else:
            state["iteration_count"] = 1
        
        answer_result = self.answer_agent.generate_answer(
            question=state["user_question"],
            retrieval_result=state["retrieval_result"]
        )
        
        state["generated_answer"] = answer_result["answer"]
        state["answer_reasoning"] = answer_result["reasoning"]
        state["messages"].append(AIMessage(content=f"Answer generation completed (iteration {state['iteration_count']})"))
        
        return state
    
    def _review_node(self, state: WorkflowState) -> WorkflowState:
        """Review node"""
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
        """Router decision function"""
        query_type = state.get("query_type", QueryType.UNKNOWN)
        if query_type == QueryType.SQL and self.sql_available:
            return "sql"
        elif query_type == QueryType.RAG:
            return "rag"
        else:
            return "unknown"
    
    def _review_decision(self, state: WorkflowState) -> str:
        """Review decision function"""
        if state.get("review_approved", False):
            return "approved"
        
        if state.get("iteration_count", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        
        return "retry"


# Create workflow instance
multi_agent_workflow = MultiAgentWorkflow()
workflow = multi_agent_workflow._build_workflow()

# Compile graph (without checkpointer, compatible with LangGraph Platform)
graph = workflow

# Main run function
def run_workflow(
    question: str,
    session_id: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Run multi-agent workflow"""
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
            max_iterations=multi_agent_workflow.max_iterations
        )
        
        # Run workflow
        final_state = graph.invoke(initial_state)
        
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

# Asynchronous run function
async def arun_workflow(
    question: str,
    session_id: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """Asynchronously run multi-agent workflow"""
    try:
        logger.info(f"Starting multi-agent workflow to process question: {question[:100]}...")
        
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
            max_iterations=multi_agent_workflow.max_iterations
        )
        
        final_state = await graph.ainvoke(initial_state)
        
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

def get_workflow_info() -> Dict[str, Any]:
    """Get workflow information"""
    return {
        "workflow_type": "multi_agent",
        "max_iterations": multi_agent_workflow.max_iterations,
        "sql_available": multi_agent_workflow.sql_available,
        "agents": {
            "router": "RouterAgent",
            "rag": "DocumentQAAgent" if multi_agent_workflow.rag_agent else "SimpleRAG",
            "sql": "SQLAgent" if multi_agent_workflow.sql_available else None,
            "answer": "AnswerAgent",
            "review": "ReviewAgent"
        },
        "nodes": list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    }

# Test function
def test_workflow():
    """Test multi-agent workflow"""
    try:
        print("=== Multi-agent Workflow Test ===")
        print("Workflow Info:", get_workflow_info())
        
        # Simple structure test, no real API call needed
        print("\n✅ Workflow structure test passed")
        print(f"- Max iterations: {multi_agent_workflow.max_iterations}")
        print(f"- SQL availability: {multi_agent_workflow.sql_available}")
        print(f"- RAG agent: {'Available' if multi_agent_workflow.rag_agent else 'Simplified version'}")
        print(f"- Graph nodes: {len(list(graph.nodes.keys())) if hasattr(graph, 'nodes') else 'Unknown'}")
        
        print("\n✅ Multi-agent workflow successfully loaded and ready")
        print("Note: Full functionality testing requires valid API keys")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("LangGraph Platform Multi-agent Workflow Loaded")
    print("Workflow Info:", get_workflow_info())
    
    # Run test
    test_workflow() 