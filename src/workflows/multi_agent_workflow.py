"""
基于LangGraph的多智能体协作工作流

实现以下流程：
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
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 迭代控制
    iteration_count: int
    max_iterations: int


class RouterAgent:
    """路由智能体 - 判断查询类型"""
    
    def __init__(self, llm=None):
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个智能路由器，负责判断用户问题应该使用SQL查询还是文档检索(RAG)来回答。

**重要提示**: 请仔细分析问题的语义和上下文，特别注意中文的同音词和多义词。

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

**特别注意**:
- 优先考虑问题的主要意图而非字面意思

请分析用户问题，返回JSON格式：
{{
    "query_type": "sql" 或 "rag",
    "reasoning": "详细的判断理由，包括对问题语义的分析",
    "confidence": 0.0-1.0的置信度
}}"""),
            ("human", "用户问题：{question}")
        ])
    
    def route(self, question: str) -> Dict[str, Any]:
        """路由决策"""
        try:
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
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案生成助手。你的任务是基于检索到的信息，生成或优化最终答案。

**重要原则**:
1. **分析检索结果**: 仔细分析检索结果的类型和内容
2. **准确理解问题意图**: 理解用户真正想了解什么
3. **基于事实回答**: 严格基于检索到的具体信息，不要编造内容
4. **智能处理结果**: 
   - 如果是SQL查询结果，直接使用查询结果，不要重新生成
   - 如果是RAG检索结果，可以基于文档内容优化答案
   - 如果检索失败，明确说明原因

处理策略：
- **SQL查询结果**: 如果检索信息来自SQL查询，直接采用查询结果，只需要适当格式化
- **RAG检索结果**: 如果检索信息来自文档检索，可以基于文档内容生成或优化答案
- **错误处理**: 如果检索失败，明确说明失败原因

请基于以下信息生成最终答案："""),
            ("human", """用户问题：{question}

检索信息：{retrieval_info}

请仔细分析检索信息的类型，如果是SQL查询结果，请直接使用；如果是文档检索结果，请基于实际内容生成答案。""")
        ])
    
    def generate_answer(self, question: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成答案"""
        try:
            # 格式化检索信息
            retrieval_info = ""
            
            # 处理RAG Agent的返回结果
            if "answer" in retrieval_result and "relevant_documents" in retrieval_result:
                # RAG Agent返回的结构
                rag_answer = retrieval_result.get("answer", "")
                relevant_docs = retrieval_result.get("relevant_documents", [])
                
                # 构建检索信息，包含RAG Agent的答案和文档内容
                retrieval_info = f"RAG检索结果：\n{rag_answer}\n\n"
                
                if relevant_docs:
                    docs_info = "\n".join([
                        f"文档{i+1}: {doc.get('content', '')[:300]}..."
                        for i, doc in enumerate(relevant_docs[:3])
                    ])
                    retrieval_info += f"相关文档片段：\n{docs_info}"
                    
            # 处理SQL Agent的返回结果
            elif "answer" in retrieval_result and "success" in retrieval_result:
                # SQL Agent返回的结构
                sql_answer = retrieval_result.get("answer", "")
                success = retrieval_result.get("success", False)
                
                if success:
                    # SQL查询成功，直接返回SQL Agent的答案，不再通过LLM重新处理
                    logger.info(f"SQL查询成功，直接返回SQL Agent答案: {sql_answer[:100]}...")
                    return {
                        "answer": sql_answer,
                        "reasoning": "SQL查询成功，直接使用SQL Agent的答案",
                        "success": True
                    }
                else:
                    # SQL查询失败
                    error_msg = retrieval_result.get("error", "未知错误")
                    retrieval_info = f"SQL查询失败: {error_msg}"
                    
            # 处理其他情况
            elif "error" in retrieval_result:
                retrieval_info = f"检索失败: {retrieval_result.get('error', '未知错误')}"
            else:
                retrieval_info = "未找到相关检索信息"
            
            # 如果检索失败，直接返回错误信息，不再通过LLM处理
            if "检索失败" in retrieval_info or "SQL查询失败" in retrieval_info:
                return {
                    "answer": retrieval_info,
                    "reasoning": "检索失败，直接返回错误信息",
                    "success": False
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
        self.llm = llm or LLMFactory.create_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的答案质量审阅员。你需要评估生成的答案质量并提供改进建议。

评估维度：
1. **准确性** (0-10分): 答案是否准确，有无事实错误，是否与检索信息一致
2. **完整性** (0-10分): 答案是否完整回答了用户问题
3. **相关性** (0-10分): 答案是否与问题高度相关，是否回答了用户真正想了解的内容
4. **清晰度** (0-10分): 答案是否表达清晰，易于理解
5. **有用性** (0-10分): 答案是否对用户有实际帮助

**特别关注**:
- 答案是否正确理解了用户问题的意图
- 答案是否基于检索到的实际信息
- 是否存在答案与检索信息不符的情况
- 是否存在概念混淆或实体识别错误

评分标准：
- 8-10分：优秀，答案准确且完整回答了问题
- 6-7分：良好，基本回答了问题但可能有小问题
- 4-5分：一般，部分回答了问题但存在明显不足
- 0-3分：差，答案错误或完全没有回答问题

请返回JSON格式：
{{
    "overall_score": 总分(0-10),
    "dimension_scores": {{
        "accuracy": 准确性得分,
        "completeness": 完整性得分,
        "relevance": 相关性得分,
        "clarity": 清晰度得分,
        "usefulness": 有用性得分
    }},
    "approved": true/false,
    "feedback": "具体的改进建议，特别指出问题所在",
    "strengths": "答案的优点",
    "weaknesses": "答案的不足和需要改进的地方"
}}"""),
            ("human", """用户问题：{question}

生成的答案：{answer}

请对这个答案进行全面评估，特别关注答案是否正确理解和回答了用户问题。""")
        ])
    
    def review_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """审阅答案"""
        try:
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
                    "detailed_scores": parsed.get("dimension_scores", {}),
                    "strengths": parsed.get("strengths", ""),
                    "weaknesses": parsed.get("weaknesses", ""),
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
        # 基于答案长度和关键词的简单评分
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
            "detailed_scores": {},
            "strengths": "",
            "weaknesses": "",
            "success": True
        }


class MultiAgentWorkflow:
    """多智能体协作工作流"""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager = None,
        sql_db_path: str = None,
        model_name: str = None,
        max_iterations: int = 2
    ):
        """
        初始化多智能体工作流
        
        Args:
            vector_store_manager: 向量存储管理器
            sql_db_path: SQL数据库路径
            model_name: 模型名称
            max_iterations: 最大迭代次数
        """
        self.model_name = model_name
        self.max_iterations = max_iterations
        
        # 初始化各个智能体
        self.router_agent = RouterAgent(LLMFactory.create_llm(model_name))
        self.answer_agent = AnswerAgent(LLMFactory.create_llm(model_name))
        self.review_agent = ReviewAgent(LLMFactory.create_llm(model_name))
        
        # 初始化RAG和SQL智能体
        if vector_store_manager is None:
            vector_store_manager = VectorStoreManager()
            vector_store_manager.get_or_create_vector_store()
        
        self.rag_agent = DocumentQAAgent(
            vector_store_manager=vector_store_manager,
            model_name=model_name,
            use_memory=True
        )
        
        
        # 初始化SQL Agent
        try:
            # 确保使用正确的数据库路径
            if sql_db_path is None:
                from pathlib import Path
                base_path = Path(__file__).parent.parent.parent
                sql_db_path = str(base_path / "data" / "database" / "erp.db")
                logger.info(f"使用默认数据库路径: {sql_db_path}")
            
            self.sql_agent = SQLAgent(
                db_path=sql_db_path,
                model_name=model_name,
                use_memory=True,
                verbose=True
            )
            self.sql_available = True
            logger.info(f"SQLAgent 在工作流中初始化成功，数据库路径: {sql_db_path}")
        except Exception as e:
            logger.error(f"SQLAgent 在工作流中初始化失败: {str(e)}")
            self.sql_agent = None
            self.sql_available = False
        
        # 构建工作流图
        self.workflow = self._build_workflow()
        
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
                "unknown": "rag_retrieval"  # 默认使用RAG
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
        state["messages"].append(AIMessage(content=f"路由决策: {routing_result['query_type'].value} - {routing_result['reasoning']}"))
        
        return state
    
    def _sql_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """SQL检索节点"""
        logger.info("执行SQL检索...")
        
        if not self.sql_available:
            state["retrieval_result"] = {
                "success": False,
                "error": "SQL Agent不可用",
                "answer": "抱歉，SQL查询功能当前不可用。"
            }
        else:
            # 为SQL Agent使用独立的会话ID，避免记忆污染
            sql_session_id = f"sql_{state['session_id']}"
            result = self.sql_agent.query(
                question=state["user_question"],
                session_id=sql_session_id
            )
            
            # 添加调试日志
            logger.debug(f"SQL检索结果结构: {list(result.keys())}")
            logger.debug(f"SQL查询成功: {result.get('success', False)}")
            logger.debug(f"SQL答案长度: {len(result.get('answer', ''))}")
            if result.get('answer'):
                logger.debug(f"SQL答案前200字符: {result.get('answer', '')[:200]}...")
            
            state["retrieval_result"] = result
        
        state["messages"].append(AIMessage(content="SQL检索完成"))
        return state
    
    def _rag_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """RAG检索节点"""
        logger.info("执行RAG检索...")
        
        result = self.rag_agent.invoke(
            question=state["user_question"],
            session_id=state["session_id"]
        )
        
        # 添加调试日志
        logger.debug(f"RAG检索结果结构: {list(result.keys())}")
        logger.debug(f"RAG答案长度: {len(result.get('answer', ''))}")
        logger.debug(f"相关文档数量: {len(result.get('relevant_documents', []))}")
        
        state["retrieval_result"] = result
        state["retrieved_documents"] = result.get("retrieved_documents", [])
        state["messages"].append(AIMessage(content="RAG检索完成"))
        
        return state
    
    def _answer_generation_node(self, state: WorkflowState) -> WorkflowState:
        """答案生成节点"""
        logger.info("生成答案...")
        
        # 如果是重新生成，增加迭代计数
        if state.get("generated_answer"):
            state["iteration_count"] = state.get("iteration_count", 0) + 1
        else:
            state["iteration_count"] = 1
        
        # 添加调试日志
        retrieval_result = state["retrieval_result"]
        logger.debug(f"传递给AnswerAgent的数据结构: {list(retrieval_result.keys()) if retrieval_result else 'None'}")
        
        answer_result = self.answer_agent.generate_answer(
            question=state["user_question"],
            retrieval_result=state["retrieval_result"]
        )
        
        logger.debug(f"AnswerAgent生成的答案长度: {len(answer_result.get('answer', ''))}")
        
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
        query_type = state["query_type"]
        if query_type == QueryType.SQL and self.sql_available:
            return "sql"
        elif query_type == QueryType.RAG:
            return "rag"
        else:
            return "unknown"
    
    def _review_decision(self, state: WorkflowState) -> str:
        """审阅决策函数"""
        # 如果审阅通过，结束流程
        if state["review_approved"]:
            return "approved"
        
        # 如果达到最大迭代次数，强制结束
        if state.get("iteration_count", 0) >= state.get("max_iterations", self.max_iterations):
            return "end"
        
        # 否则重新生成答案
        return "retry"
    
    @with_langsmith_tracing(name="MultiAgentWorkflow.run", tags=["workflow", "multi-agent"])
    def run(
        self,
        question: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行多智能体工作流
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Returns:
            包含完整结果的字典
        """
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
                max_iterations=self.max_iterations
            )
            
            # 运行工作流
            final_state = self.workflow.invoke(initial_state)
            
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
    
    async def arun(
        self,
        question: str,
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        异步运行多智能体工作流
        
        Args:
            question: 用户问题
            session_id: 会话ID
            **kwargs: 其他参数
        
        Returns:
            包含完整结果的字典
        """
        try:
            logger.info(f"启动异步多智能体工作流处理问题: {question[:100]}...")
            
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
                max_iterations=self.max_iterations
            )
            
            # 异步运行工作流
            final_state = await self.workflow.ainvoke(initial_state)
            
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
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """获取工作流信息"""
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