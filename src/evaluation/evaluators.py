"""
LLM Evaluator Module
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
    """Evaluator type enumeration"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    HELPFULNESS = "helpfulness"
    COHERENCE = "coherence"
    GROUNDEDNESS = "groundedness"
    CUSTOM = "custom"


class AccuracyEvaluator(LangChainStringEvaluator):
    """Accuracy evaluator"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "accuracy"
        self.description = "Evaluate answer accuracy"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate string accuracy"""
        
        prompt = f"""
Please evaluate the accuracy of the following AI answer.

Question: {input or "Not provided"}
Reference answer: {reference or "Not provided"}
AI answer: {prediction}

Evaluation criteria:
1. Is the information accurate and error-free
2. Does it address the core of the question
3. Are there any errors or misleading information

Please give a score of 1-5 (5 being the highest) and briefly explain the reason.
Format: Score: X, Reason: ...
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse score
            score = self._extract_score(response.content if hasattr(response, 'content') else str(response))
            
            return EvaluationResult(
                key="accuracy",
                score=score,
                comment=response.content if hasattr(response, 'content') else str(response)
            )
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {str(e)}")
            return EvaluationResult(
                key="accuracy",
                score=0,
                comment=f"Evaluation failed: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response"""
        try:
            # Look for score patterns
            import re
            match = re.search(r'[Ss]core[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0  # Convert to 0-1 range
            
            # If not found, try other patterns
            match = re.search(r'([1-5])/5', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5  # Default medium score
        except:
            return 0.5


class RelevanceEvaluator(LangChainStringEvaluator):
    """Relevance evaluator"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "relevance"
        self.description = "Evaluate answer relevance to question"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate relevance"""
        
        prompt = f"""
Please evaluate the relevance of the AI answer to the user question.

User question: {input or "Not provided"}
AI answer: {prediction}

Evaluation criteria:
1. Does the answer directly address the question
2. Does it contain key information related to the question
3. Does it deviate from the topic or contain irrelevant content

Please give a score of 1-5 (5 being the highest) and briefly explain the reason.
Format: Score: X, Reason: ...
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
            logger.error(f"Relevance evaluation failed: {str(e)}")
            return EvaluationResult(
                key="relevance",
                score=0,
                comment=f"Evaluation failed: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response"""
        try:
            import re
            match = re.search(r'[Ss]core[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])/5', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class HelpfulnessEvaluator(LangChainStringEvaluator):
    """Helpfulness evaluator"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "helpfulness"
        self.description = "Evaluate answer helpfulness"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate helpfulness"""
        
        prompt = f"""
Please evaluate the helpfulness of the AI answer to the user.

User question: {input or "Not provided"}
AI answer: {prediction}

Evaluation criteria:
1. Does it provide practical information
2. Does it solve the user's problem
3. Does it give specific suggestions or guidance
4. Completeness and depth of the answer

Please give a score of 1-5 (5 being the highest) and briefly explain the reason.
Format: Score: X, Reason: ...
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
            logger.error(f"Helpfulness evaluation failed: {str(e)}")
            return EvaluationResult(
                key="helpfulness",
                score=0,
                comment=f"Evaluation failed: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response"""
        try:
            import re
            match = re.search(r'[Ss]core[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])/5', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class GroundednessEvaluator(LangChainStringEvaluator):
    """Groundedness evaluator"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or LLMFactory.create_llm()
        self.name = "groundedness"
        self.description = "Evaluate whether answer is based on provided document content"
    
    def _evaluate_strings(
        self,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate groundedness level"""
        
        context = kwargs.get('context', '')
        
        prompt = f"""
Please evaluate whether the AI answer is based on the provided document content, avoiding hallucination and fabrication.

User question: {input or "Not provided"}
Reference document: {context or "Not provided"}
AI answer: {prediction}

Evaluation criteria:
1. Is the information in the answer sourced from the reference document
2. Is there any fabricated or unfounded information
3. Does it correctly cite document content
4. For information not in the document, is it clearly stated

Please give a score of 1-5 (5 being the highest) and briefly explain the reason.
Format: Score: X, Reason: ...
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
            logger.error(f"Groundedness evaluation failed: {str(e)}")
            return EvaluationResult(
                key="groundedness",
                score=0,
                comment=f"Evaluation failed: {str(e)}"
            )
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response"""
        try:
            import re
            match = re.search(r'[Ss]core[：:]\s*([1-5])', response)
            if match:
                return float(match.group(1)) / 5.0
            
            match = re.search(r'([1-5])/5', response)
            if match:
                return float(match.group(1)) / 5.0
                
            return 0.5
        except:
            return 0.5


class EvaluatorFactory:
    """Evaluator factory"""
    
    @staticmethod
    def create_evaluator(
        evaluator_type: Union[EvaluatorType, str],
        llm: Optional[BaseLanguageModel] = None
    ) -> LangChainStringEvaluator:
        """Create evaluator"""
        
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
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
        
        return evaluator_class(llm=llm)
    
    @staticmethod
    def create_multiple_evaluators(
        evaluator_types: List[Union[EvaluatorType, str]],
        llm: Optional[BaseLanguageModel] = None
    ) -> List[LangChainStringEvaluator]:
        """Create multiple evaluators"""
        
        return [
            EvaluatorFactory.create_evaluator(eval_type, llm)
            for eval_type in evaluator_types
        ]
    
    @staticmethod
    def get_default_evaluators(llm: Optional[BaseLanguageModel] = None) -> List[LangChainStringEvaluator]:
        """Get default evaluator set"""
        
        default_types = [
            EvaluatorType.ACCURACY,
            EvaluatorType.RELEVANCE,
            EvaluatorType.HELPFULNESS,
            EvaluatorType.GROUNDEDNESS
        ]
        
        return EvaluatorFactory.create_multiple_evaluators(default_types, llm) 