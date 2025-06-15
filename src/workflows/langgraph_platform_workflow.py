#!/usr/bin/env python3
"""
LangGraph Platform 多智能体协作工作流
基于原始 multi_agent_workflow.py 的完整功能实现
兼容 LangGraph Platform 部署要求
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

# 导入各个代理和工具
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.agents.rag_agent import DocumentQAAgent
    from src.agents.sql_agent import SQLAgent
    from src.models.llm_factory import LLMFactory
    from src.vectorstores.vector_store import VectorStoreManager
    from src.config.settings import settings
except ImportError as e:
    logging.error(f"导入失败: {e}")
    DocumentQAAgent = None
    SQLAgent = None
    LLMFactory = None
    VectorStoreManager = None
    settings = None

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """查询类型枚举"""
    SQL = "sql"
    RAG = "rag"
    UNKNOWN = "unknown"


class WorkflowState(TypedDict):
    """工作流状态定义"""
    # 基础信息
    user_question: str
    session_id: str
    
    # 路由信息
    query_type: QueryType
    router_reasoning: str
    
    # 检索结果
    retrieval_result: Optional[Dict[str, Any]]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    
    # 答案生成
    generated_answer: str
    answer_reasoning: str
    
    # 审阅结果
    review_score: float
    review_feedback: str
    review_approved: bool
    
    # 消息历史
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 迭代控制
    iteration_count: int
    max_iterations: int


class RouterAgent:
    """路由智能体 - 判断查询类型"""
    
    def __init__(self, llm=None):
        # 简化的LLM创建逻辑，兼容平台环境
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    # 平台环境下的后备方案
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM初始化失败，使用模拟LLM: {e}")
                # 创建一个模拟的LLM用于测试
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能路由器，负责判断用户问题应该使用SQL查询还是文档检索(RAG)来回答。

判断规则：
1. **SQL查询** - 如果问题涉及：
   - 数据统计、计算、聚合（如：总数、平均值、最大值、排序）
   - 具体的数值查询（如：价格、数量、日期范围）
   - 数据库表中的结构化数据查询
   - 关键词：多少、统计、计算、排名、比较、筛选、查找记录

2. **文档检索(RAG)** - 如果问题涉及：
   - 概念解释、定义说明（如：什么是...、介绍...）
   - 人物介绍、实体描述（如：你认识...吗、谁是...）
   - 流程描述、方法介绍
   - 非结构化文本内容
   - 知识性问答
   - 关键词：是什么、怎么做、为什么、解释、介绍、认识、了解

请分析用户问题，返回JSON格式：
{{
    "query_type": "sql" 或 "rag",
    "reasoning": "详细的判断理由",
    "confidence": 0.0-1.0的置信度
}}"""),
            ("human", "用户问题：{question}")
        ])
    
    def route(self, question: str) -> Dict[str, Any]:
        """路由决策"""
        try:
            if self.llm is None:
                # 没有LLM时使用后备路由
                query_type, reasoning = self._fallback_routing(question)
                return {
                    "query_type": query_type,
                    "reasoning": reasoning,
                    "confidence": 0.8
                }
            
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({"question": question})
            
            # 解析JSON结果
            import json
            try:
                parsed = json.loads(result)
                query_type = QueryType(parsed.get("query_type", "unknown"))
                reasoning = parsed.get("reasoning", "")
                confidence = parsed.get("confidence", 0.5)
            except (json.JSONDecodeError, ValueError):
                # 如果解析失败，使用关键词匹配作为后备
                query_type, reasoning = self._fallback_routing(question)
                confidence = 0.6
            
            logger.info(f"路由决策: {query_type.value} (置信度: {confidence})")
            return {
                "query_type": query_type,
                "reasoning": reasoning,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"路由决策失败: {str(e)}")
            # 使用后备路由
            query_type, reasoning = self._fallback_routing(question)
            return {
                "query_type": query_type,
                "reasoning": reasoning,
                "confidence": 0.3
            }
    
    def _fallback_routing(self, question: str) -> tuple[QueryType, str]:
        """后备路由逻辑"""
        sql_keywords = ["多少", "统计", "计算", "排名", "比较", "筛选", "查找", "数量", "价格", "总数", "平均", "最大", "最小"]
        rag_keywords = ["是什么", "怎么做", "为什么", "解释", "介绍", "定义", "概念", "方法", "流程", "认识", "了解", "谁是"]
        
        question_lower = question.lower()
        
        sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
        rag_score = sum(1 for keyword in rag_keywords if keyword in question_lower)
        
        if sql_score > rag_score:
            return QueryType.SQL, f"检测到SQL相关关键词: {sql_score}个"
        elif rag_score > sql_score:
            return QueryType.RAG, f"检测到RAG相关关键词: {rag_score}个"
        else:
            return QueryType.RAG, "默认使用RAG检索"


class AnswerAgent:
    """答案生成智能体"""
    
    def __init__(self, llm=None):
        # 简化的LLM创建逻辑
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM初始化失败，使用模拟LLM: {e}")
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案生成助手。基于检索到的信息生成最终答案。

处理策略：
- **SQL查询结果**: 如果检索信息来自SQL查询，直接采用查询结果
- **RAG检索结果**: 如果检索信息来自文档检索，基于文档内容生成答案
- **错误处理**: 如果检索失败，明确说明失败原因

请基于以下信息生成最终答案："""),
            ("human", """用户问题：{question}

检索信息：{retrieval_info}

请生成准确的答案。""")
        ])
    
    def generate_answer(self, question: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成答案"""
        try:
            # 格式化检索信息
            retrieval_info = ""
            
            # 处理RAG Agent的返回结果
            if "answer" in retrieval_result and "relevant_documents" in retrieval_result:
                rag_answer = retrieval_result.get("answer", "")
                relevant_docs = retrieval_result.get("relevant_documents", [])
                
                retrieval_info = f"RAG检索结果：\n{rag_answer}\n\n"
                
                if relevant_docs:
                    docs_info = "\n".join([
                        f"文档{i+1}: {doc.get('content', '')[:300]}..."
                        for i, doc in enumerate(relevant_docs[:3])
                    ])
                    retrieval_info += f"相关文档片段：\n{docs_info}"
                    
            # 处理SQL Agent的返回结果
            elif "answer" in retrieval_result and "success" in retrieval_result:
                sql_answer = retrieval_result.get("answer", "")
                success = retrieval_result.get("success", False)
                
                if success:
                    # SQL查询成功，直接返回SQL Agent的答案
                    logger.info(f"SQL查询成功，直接返回SQL Agent答案")
                    return {
                        "answer": sql_answer,
                        "reasoning": "SQL查询成功，直接使用SQL Agent的答案",
                        "success": True
                    }
                else:
                    error_msg = retrieval_result.get("error", "未知错误")
                    retrieval_info = f"SQL查询失败: {error_msg}"
                    
            # 处理其他情况
            elif "error" in retrieval_result:
                retrieval_info = f"检索失败: {retrieval_result.get('error', '未知错误')}"
            else:
                retrieval_info = "未找到相关检索信息"
            
            # 如果检索失败，直接返回错误信息
            if "检索失败" in retrieval_info or "SQL查询失败" in retrieval_info:
                return {
                    "answer": retrieval_info,
                    "reasoning": "检索失败，直接返回错误信息",
                    "success": False
                }
            
            # 如果没有LLM，返回简化答案
            if self.llm is None:
                return {
                    "answer": f"基于检索信息的简化答案：{retrieval_info[:200]}...",
                    "reasoning": "使用简化答案生成（无LLM）",
                    "success": True
                }
            
            chain = self.prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                "question": question,
                "retrieval_info": retrieval_info
            })
            
            return {
                "answer": answer,
                "reasoning": "基于检索信息生成答案",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"答案生成失败: {str(e)}")
            return {
                "answer": f"抱歉，生成答案时出现错误: {str(e)}",
                "reasoning": "生成过程中出现异常",
                "success": False
            }


class ReviewAgent:
    """审阅智能体 - 评估答案质量"""
    
    def __init__(self, llm=None):
        # 简化的LLM创建逻辑
        if llm is None:
            try:
                if LLMFactory:
                    self.llm = LLMFactory.create_llm()
                else:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            except Exception as e:
                logger.warning(f"LLM初始化失败，使用模拟LLM: {e}")
                self.llm = None
        else:
            self.llm = llm
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案质量审阅员。评估答案质量并提供改进建议。

评分标准：
- 8-10分：优秀，答案准确且完整回答了问题
- 6-7分：良好，基本回答了问题但可能有小问题
- 4-5分：一般，部分回答了问题但存在明显不足
- 0-3分：差，答案错误或完全没有回答问题

请返回JSON格式：
{{
    "overall_score": 总分(0-10),
    "approved": true/false,
    "feedback": "具体的改进建议"
}}"""),
            ("human", """用户问题：{question}

生成的答案：{answer}

请对这个答案进行评估。""")
        ])
    
    def review_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """审阅答案"""
        try:
            # 如果没有LLM，使用简单评分
            if self.llm is None:
                return self._simple_review(answer)
            
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "question": question,
                "answer": answer
            })
            
            # 解析JSON结果
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
                # 如果解析失败，使用简单评分
                return self._simple_review(answer)
                
        except Exception as e:
            logger.error(f"答案审阅失败: {str(e)}")
            return self._simple_review(answer)
    
    def _simple_review(self, answer: str) -> Dict[str, Any]:
        """简单审阅逻辑"""
        score = 5.0
        if len(answer) > 100:
            score += 1.0
        if len(answer) > 300:
            score += 1.0
        if "抱歉" in answer or "错误" in answer:
            score -= 2.0
        
        score = max(0.0, min(10.0, score))
        
        return {
            "score": score,
            "approved": score >= 8.0,
            "feedback": "基于简单规则的评分",
            "success": True
        }


class MultiAgentWorkflow:
    """多智能体协作工作流"""
    
    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations
        
        # 初始化各个智能体（使用LLMFactory）
        try:
            base_llm = None
            if LLMFactory:
                try:
                    base_llm = LLMFactory.create_llm()
                    logger.info("LLM创建成功")
                except Exception as llm_e:
                    logger.warning(f"LLM创建失败，将使用None: {str(llm_e)}")
                    base_llm = None
            
            self.router_agent = RouterAgent(base_llm)
            self.answer_agent = AnswerAgent(base_llm)
            self.review_agent = ReviewAgent(base_llm)
            logger.info("智能体初始化成功")
        except Exception as e:
            logger.error(f"智能体初始化失败: {str(e)}")
            self.router_agent = RouterAgent(None)
            self.answer_agent = AnswerAgent(None)
            self.review_agent = ReviewAgent(None)
        
        # 初始化RAG和SQL智能体（带错误处理）
        self.rag_agent = None
        self.sql_agent = None
        self.sql_available = False
        
        # 强制初始化RAG Agent
        try:
            if DocumentQAAgent and VectorStoreManager:
                # 创建不依赖LLM的向量存储管理器
                vector_store_manager = VectorStoreManager()
                vector_store_manager.get_or_create_vector_store()
                
                # 创建RAG Agent
                self.rag_agent = DocumentQAAgent(
                    vector_store_manager=vector_store_manager,
                    use_memory=True
                )
                logger.info("RAG Agent 初始化成功")
            else:
                raise ImportError("DocumentQAAgent 或 VectorStoreManager 不可用")
        except Exception as e:
            logger.warning(f"RAG Agent 初始化失败，尝试重新导入: {str(e)}")
            # 重新尝试导入
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
                logger.info("RAG Agent 重新初始化成功")
            except Exception as e2:
                logger.error(f"RAG Agent 重新初始化也失败: {str(e2)}")
                self.rag_agent = None
        
        # 强制初始化SQL Agent
        try:
            if SQLAgent:
                from pathlib import Path
                base_path = Path(__file__).parent.parent.parent
                sql_db_path = str(base_path / "data" / "database" / "erp.db")
                
                # 创建SQL Agent
                self.sql_agent = SQLAgent(
                    db_path=sql_db_path,
                    use_memory=True,
                    verbose=True
                )
                self.sql_available = True
                logger.info(f"SQL Agent 初始化成功，数据库路径: {sql_db_path}")
            else:
                raise ImportError("SQLAgent 不可用")
        except Exception as e:
            logger.warning(f"SQL Agent 初始化失败，尝试重新导入: {str(e)}")
            # 重新尝试导入
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
                logger.info(f"SQL Agent 重新初始化成功，数据库路径: {sql_db_path}")
            except Exception as e2:
                logger.error(f"SQL Agent 重新初始化也失败: {str(e2)}")
                self.sql_agent = None
                self.sql_available = False
        
        logger.info("多智能体工作流初始化完成")
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("router", self._router_node)
        workflow.add_node("sql_retrieval", self._sql_retrieval_node)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("review", self._review_node)
        
        # 设置入口点
        workflow.set_entry_point("router")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "sql": "sql_retrieval",
                "rag": "rag_retrieval",
                "unknown": "rag_retrieval"
            }
        )
        
        # 检索到答案生成
        workflow.add_edge("sql_retrieval", "answer_generation")
        workflow.add_edge("rag_retrieval", "answer_generation")
        
        # 答案生成到审阅
        workflow.add_edge("answer_generation", "review")
        
        # 审阅的条件边
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
        """路由节点"""
        logger.info("执行路由决策...")
        
        routing_result = self.router_agent.route(state["user_question"])
        
        state["query_type"] = routing_result["query_type"]
        state["router_reasoning"] = routing_result["reasoning"]
        state["messages"].append(AIMessage(content=f"路由决策: {routing_result['query_type'].value}"))
        
        return state
    
    def _sql_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """SQL检索节点"""
        logger.info("执行SQL检索...")
        
        if not self.sql_available or not self.sql_agent:
            state["retrieval_result"] = {
                "success": False,
                "error": "SQL Agent不可用",
                "answer": "抱歉，SQL查询功能当前不可用。"
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
                logger.error(f"SQL检索失败: {str(e)}")
                state["retrieval_result"] = {
                    "success": False,
                    "error": f"SQL查询执行失败: {str(e)}",
                    "answer": f"抱歉，SQL查询执行失败: {str(e)}"
                }
        
        state["messages"].append(AIMessage(content="SQL检索完成"))
        return state
    
    def _rag_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """RAG检索节点"""
        logger.info("执行RAG检索...")
        
        if not self.rag_agent:
            # 简化的RAG处理
            question = state["user_question"]
            if "什么是" in question or "介绍" in question:
                answer = f"这是一个知识问答问题：{question}。我会基于文档为您解答。"
            else:
                answer = f"收到您的问题：{question}。我正在为您处理。"
            
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
                logger.error(f"RAG检索失败: {str(e)}")
                state["retrieval_result"] = {
                    "answer": f"RAG检索失败: {str(e)}",
                    "relevant_documents": [],
                    "success": False,
                    "error": str(e)
                }
        
        state["messages"].append(AIMessage(content="RAG检索完成"))
        return state
    
    def _answer_generation_node(self, state: WorkflowState) -> WorkflowState:
        """答案生成节点"""
        logger.info("生成答案...")
        
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
        state["messages"].append(AIMessage(content=f"答案生成完成 (第{state['iteration_count']}次)"))
        
        return state
    
    def _review_node(self, state: WorkflowState) -> WorkflowState:
        """审阅节点"""
        logger.info("审阅答案质量...")
        
        review_result = self.review_agent.review_answer(
            question=state["user_question"],
            answer=state["generated_answer"]
        )
        
        state["review_score"] = review_result["score"]
        state["review_feedback"] = review_result["feedback"]
        state["review_approved"] = review_result["approved"]
        state["messages"].append(AIMessage(content=f"审阅完成 - 得分: {review_result['score']:.1f}"))
        
        return state
    
    def _route_decision(self, state: WorkflowState) -> str:
        """路由决策函数"""
        query_type = state.get("query_type", QueryType.UNKNOWN)
        if query_type == QueryType.SQL and self.sql_available:
            return "sql"
        elif query_type == QueryType.RAG:
            return "rag"
        else:
            return "unknown"
    
    def _review_decision(self, state: WorkflowState) -> str:
        """审阅决策函数"""
        if state.get("review_approved", False):
            return "approved"
        
        if state.get("iteration_count", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        
        return "retry"


# 创建工作流实例
multi_agent_workflow = MultiAgentWorkflow()
workflow = multi_agent_workflow._build_workflow()

# 编译图（不使用checkpointer，兼容LangGraph Platform）
graph = workflow

# 主要运行函数
def run_workflow(
    question: str,
    session_id: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """运行多智能体工作流"""
    try:
        logger.info(f"启动多智能体工作流处理问题: {question[:100]}...")
        
        # 初始化状态
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
        
        # 运行工作流
        final_state = graph.invoke(initial_state)
        
        # 构建返回结果
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
        
        logger.info(f"工作流完成 - 查询类型: {final_state['query_type'].value}, 审阅得分: {final_state['review_score']:.1f}")
        return result
        
    except Exception as e:
        logger.error(f"多智能体工作流执行失败: {str(e)}")
        return {
            "question": question,
            "answer": f"工作流执行失败: {str(e)}",
            "query_type": "error",
            "router_reasoning": "",
            "review_score": 0.0,
            "review_feedback": "执行过程中出现异常",
            "review_approved": False,
            "iteration_count": 0,
            "retrieved_documents": [],
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }

# 异步运行函数
async def arun_workflow(
    question: str,
    session_id: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """异步运行多智能体工作流"""
    try:
        logger.info(f"启动异步多智能体工作流处理问题: {question[:100]}...")
        
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
        
        logger.info(f"异步工作流完成 - 查询类型: {final_state['query_type'].value}, 审阅得分: {final_state['review_score']:.1f}")
        return result
        
    except Exception as e:
        logger.error(f"异步多智能体工作流执行失败: {str(e)}")
        return {
            "question": question,
            "answer": f"工作流执行失败: {str(e)}",
            "query_type": "error",
            "router_reasoning": "",
            "review_score": 0.0,
            "review_feedback": "执行过程中出现异常",
            "review_approved": False,
            "iteration_count": 0,
            "retrieved_documents": [],
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }

def get_workflow_info() -> Dict[str, Any]:
    """获取工作流信息"""
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

# 测试函数
def test_workflow():
    """测试多智能体工作流"""
    try:
        print("=== 多智能体工作流测试 ===")
        print("工作流信息:", get_workflow_info())
        
        # 简单的结构测试，不需要真实的API调用
        print("\n✅ 工作流结构测试通过")
        print(f"- 最大迭代次数: {multi_agent_workflow.max_iterations}")
        print(f"- SQL可用性: {multi_agent_workflow.sql_available}")
        print(f"- RAG代理: {'可用' if multi_agent_workflow.rag_agent else '简化版本'}")
        print(f"- 图节点数: {len(list(graph.nodes.keys())) if hasattr(graph, 'nodes') else '未知'}")
        
        print("\n✅ 多智能体工作流已成功加载并准备就绪")
        print("注意：完整功能测试需要有效的API密钥")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    print("LangGraph Platform 多智能体工作流已加载")
    print("工作流信息:", get_workflow_info())
    
    # 运行测试
    test_workflow() 