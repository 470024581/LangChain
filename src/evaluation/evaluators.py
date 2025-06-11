"""
LLM 评估器模块
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from langsmith.evaluation import LangChainStringEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator
from langchain_core.language_models import BaseLanguageModel

from ..models.llm_factory import LLMFactory
from ..config.settings import settings

logger = logging.getLogger(__name__)


class EvaluatorType(Enum):
    """评估器类型枚举"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    COHERENCE = "coherence"
    GROUNDEDNESS = "groundedness"
    CUSTOM = "custom"


class AccuracyEvaluator(LangChainStringEvaluator):
    """准确性评估器"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "accuracy"
        self.description = "评估答案的准确性"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """评估字符串准确性"""
        
        prompt = f"""
请评估以下AI回答的准确性。

问题: {input or "未提供"}
参考答案: {reference or "未提供"}
AI回答: {prediction}

评估标准:
1. 信息是否准确无误
2. 是否回答了问题的核心
3. 是否存在错误或误导性信息

请给出1-5分的评分（5分最高）并简要说明理由。
格式: 分数: X, 理由: ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # 解析分数
            score = self._extract_score(response.content if hasattr(response, 'content') else str(response))
            
            return EvaluationResult(
                key="accuracy",
                score=score,
                comment=response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as e:
            logger.error(f"准确性评估失败: {str(e)}")
            return EvaluationResult(
                key="accuracy",
                score=0,
                comment=f"评估失败: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        try:
            # 查找分数模式
            import re
            match = re.search(r'分数[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0  # 转换为0-1范围
            
            # 如果没找到，尝试其他模式
            match = re.search(r'([1-5])分', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5  # 默认中等分数
        except:
            return 0.5


class RelevanceEvaluator(LangChainStringEvaluator):
    """相关性评估器"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "relevance"
        self.description = "评估答案与问题的相关性"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """评估相关性"""
        
        prompt = f"""
请评估AI回答与用户问题的相关性。

用户问题: {input or "未提供"}
AI回答: {prediction}

评估标准:
1. 回答是否直接针对问题
2. 是否包含问题相关的关键信息
3. 是否偏离主题或包含无关内容

请给出1-5分的评分（5分最高）并简要说明理由。
格式: 分数: X, 理由: ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            score = self._extract_score(response.content if hasattr(response, 'content') else str(response))
            
            return EvaluationResult(
                key="relevance",
                score=score,
                comment=response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as e:
            logger.error(f"相关性评估失败: {str(e)}")
            return EvaluationResult(
                key="relevance",
                score=0,
                comment=f"评估失败: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        try:
            import re
            match = re.search(r'分数[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])分', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class HelpfulnessEvaluator(LangChainStringEvaluator):
    """有用性评估器"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "helpfulness"
        self.description = "评估答案的有用性"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """评估有用性"""
        
        prompt = f"""
请评估AI回答对用户的有用性。

用户问题: {input or "未提供"}
AI回答: {prediction}

评估标准:
1. 是否提供了实用的信息
2. 是否解决了用户的问题
3. 是否给出了具体的建议或指导
4. 回答的完整性和深度

请给出1-5分的评分（5分最高）并简要说明理由。
格式: 分数: X, 理由: ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            score = self._extract_score(response.content if hasattr(response, 'content') else str(response))
            
            return EvaluationResult(
                key="helpfulness",
                score=score,
                comment=response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as e:
            logger.error(f"有用性评估失败: {str(e)}")
            return EvaluationResult(
                key="helpfulness",
                score=0,
                comment=f"评估失败: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        try:
            import re
            match = re.search(r'分数[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])分', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class GroundednessEvaluator(LangChainStringEvaluator):
    """基于事实的评估器"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "groundedness"
        self.description = "评估答案是否基于提供的文档内容"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """评估基于事实程度"""
        
        context = kwargs.get('context', '')
        
        prompt = f"""
请评估AI回答是否基于提供的文档内容，避免幻觉和编造。

用户问题: {input or "未提供"}
参考文档: {context or "未提供"}
AI回答: {prediction}

评估标准:
1. 回答中的信息是否来源于参考文档
2. 是否存在编造或无依据的信息
3. 是否正确引用了文档内容
4. 对于文档中没有的信息，是否明确说明

请给出1-5分的评分（5分最高）并简要说明理由。
格式: 分数: X, 理由: ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            score = self._extract_score(response.content if hasattr(response, 'content') else str(response))
            
            return EvaluationResult(
                key="groundedness",
                score=score,
                comment=response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as e:
            logger.error(f"基于事实评估失败: {str(e)}")
            return EvaluationResult(
                key="groundedness",
                score=0,
                comment=f"评估失败: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """从响应中提取分数"""
        try:
            import re
            match = re.search(r'分数[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])分', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class EvaluatorFactory:
    """评估器工厂"""
    
    @staticmethod
    def create_evaluator(
        evaluator_type: Union[EvaluatorType, str],
        llm: Optional[BaseLanguageModel] = None
    ) -> LangChainStringEvaluator:
        """创建评估器"""
        
        if isinstance(evaluator_type, str):
            evaluator_type = EvaluatorType(evaluator_type)
        
        evaluator_map = {
            EvaluatorType.ACCURACY: AccuracyEvaluator,
            EvaluatorType.RELEVANCE: RelevanceEvaluator,
            EvaluatorType.HELPFULNESS: HelpfulnessEvaluator,
            EvaluatorType.GROUNDEDNESS: GroundednessEvaluator,
        }
        
        evaluator_class = evaluator_map.get(evaluator_type)
        if not evaluator_class:
            raise ValueError(f"不支持的评估器类型: {evaluator_type}")
        
        return evaluator_class(llm=llm)
    
    @staticmethod
    def create_multiple_evaluators(
        evaluator_types: List[Union[EvaluatorType, str]],
        llm: Optional[BaseLanguageModel] = None
    ) -> List[LangChainStringEvaluator]:
        """创建多个评估器"""
        
        return [
            EvaluatorFactory.create_evaluator(eval_type, llm)
            for eval_type in evaluator_types
        ]
    
    @staticmethod
    def get_default_evaluators(llm: Optional[BaseLanguageModel] = None) -> List[LangChainStringEvaluator]:
        """获取默认评估器集合"""
        
        default_types = [
            EvaluatorType.ACCURACY,
            EvaluatorType.RELEVANCE,
            EvaluatorType.HELPFULNESS,
            EvaluatorType.GROUNDEDNESS
        ]
        
        return EvaluatorFactory.create_multiple_evaluators(default_types, llm) 