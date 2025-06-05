"""
评估运行器模块
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from langsmith.evaluation import evaluate
from langsmith import Client

from .datasets import EvaluationDataset, EvaluationExample
from .evaluators import EvaluatorFactory, EvaluatorType
from ..chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
from ..utils.langsmith_utils import langsmith_manager

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """单个评估结果"""
    example_id: str
    question: str
    prediction: str
    expected_answer: Optional[str]
    scores: Dict[str, float]
    comments: Dict[str, str]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class EvaluationReport:
    """评估报告"""
    dataset_name: str
    evaluator_names: List[str]
    total_examples: int
    avg_scores: Dict[str, float]
    individual_results: List[EvaluationResult]
    execution_time: float
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_to_file(self, file_path: str):
        """保存到文件"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"评估报告已保存到: {file_path}")


class EvaluationRunner:
    """评估运行器"""
    
    def __init__(
        self,
        qa_chain: Optional[DocumentQAChain] = None,
        conversational_chain: Optional[ConversationalRetrievalChain] = None
    ):
        self.qa_chain = qa_chain
        self.conversational_chain = conversational_chain
    
    async def run_evaluation(
        self,
        dataset: EvaluationDataset,
        evaluator_types: List[str] = None,
        use_conversational: bool = False,
        max_concurrency: int = 3
    ) -> EvaluationReport:
        """运行评估"""
        
        start_time = datetime.now()
        
        # 默认评估器
        if not evaluator_types:
            evaluator_types = ["accuracy", "relevance", "helpfulness", "groundedness"]
        
        # 创建评估器
        evaluators = EvaluatorFactory.create_multiple_evaluators(evaluator_types)
        
        # 选择使用的链
        chain = self.conversational_chain if use_conversational else self.qa_chain
        if not chain:
            raise ValueError("未提供有效的问答链")
        
        logger.info(f"开始评估数据集 '{dataset.name}'，包含 {len(dataset)} 个样例")
        
        # 运行评估
        individual_results = []
        semaphore = asyncio.Semaphore(max_concurrency)
        
        tasks = [
            self._evaluate_single_example(example, evaluators, chain, semaphore, i)
            for i, example in enumerate(dataset.examples)
        ]
        
        individual_results = await asyncio.gather(*tasks)
        
        # 计算平均分数
        avg_scores = self._calculate_average_scores(individual_results, evaluator_types)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # 创建评估报告
        report = EvaluationReport(
            dataset_name=dataset.name,
            evaluator_names=evaluator_types,
            total_examples=len(dataset),
            avg_scores=avg_scores,
            individual_results=individual_results,
            execution_time=execution_time,
            timestamp=start_time.isoformat(),
            metadata={
                "use_conversational": use_conversational,
                "max_concurrency": max_concurrency
            }
        )
        
        logger.info(f"评估完成，总耗时: {execution_time:.2f}秒")
        self._log_results(report)
        
        return report
    
    async def _evaluate_single_example(
        self,
        example: EvaluationExample,
        evaluators: List,
        chain,
        semaphore: asyncio.Semaphore,
        example_index: int
    ) -> EvaluationResult:
        """评估单个样例"""
        
        async with semaphore:
            start_time = datetime.now()
            
            try:
                # 获取预测结果
                if hasattr(chain, 'invoke'):
                    # 标准问答链
                    result = chain.invoke(
                        question=example.question,
                        session_id=example.session_id or f"eval_{example_index}"
                    )
                    prediction = result.get("answer", "")
                else:
                    # 其他类型的链
                    prediction = str(chain.invoke({"question": example.question}))
                
                # 运行所有评估器
                scores = {}
                comments = {}
                
                for evaluator in evaluators:
                    eval_result = evaluator._evaluate_strings(
                        prediction=prediction,
                        reference=example.expected_answer,
                        input=example.question,
                        context=example.context
                    )
                    scores[evaluator.name] = eval_result.score
                    comments[evaluator.name] = eval_result.comment
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return EvaluationResult(
                    example_id=f"example_{example_index}",
                    question=example.question,
                    prediction=prediction,
                    expected_answer=example.expected_answer,
                    scores=scores,
                    comments=comments,
                    execution_time=execution_time,
                    metadata=example.metadata or {}
                )
                
            except Exception as e:
                logger.error(f"评估样例 {example_index} 失败: {str(e)}")
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return EvaluationResult(
                    example_id=f"example_{example_index}",
                    question=example.question,
                    prediction=f"错误: {str(e)}",
                    expected_answer=example.expected_answer,
                    scores={evaluator.name: 0.0 for evaluator in evaluators},
                    comments={evaluator.name: f"评估失败: {str(e)}" for evaluator in evaluators},
                    execution_time=execution_time,
                    metadata=example.metadata or {}
                )
    
    def _calculate_average_scores(
        self,
        results: List[EvaluationResult],
        evaluator_names: List[str]
    ) -> Dict[str, float]:
        """计算平均分数"""
        
        avg_scores = {}
        
        for evaluator_name in evaluator_names:
            scores = [result.scores.get(evaluator_name, 0.0) for result in results]
            avg_scores[evaluator_name] = sum(scores) / len(scores) if scores else 0.0
        
        # 计算总体平均分
        all_scores = list(avg_scores.values())
        avg_scores["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return avg_scores
    
    def _log_results(self, report: EvaluationReport):
        """记录评估结果"""
        logger.info("=" * 50)
        logger.info(f"评估报告 - {report.dataset_name}")
        logger.info("=" * 50)
        logger.info(f"总样例数: {report.total_examples}")
        logger.info(f"执行时间: {report.execution_time:.2f}秒")
        logger.info("平均分数:")
        
        for evaluator_name, score in report.avg_scores.items():
            logger.info(f"  {evaluator_name}: {score:.3f}")
        
        logger.info("=" * 50)
    
    def run_langsmith_evaluation(
        self,
        dataset_name: str,
        experiment_name: str = None,
        evaluator_types: List[str] = None,
        use_conversational: bool = False
    ) -> Optional[str]:
        """在 LangSmith 上运行评估"""
        
        if not langsmith_manager.is_enabled:
            logger.warning("LangSmith 未启用，无法运行云端评估")
            return None
        
        try:
            # 选择使用的链
            chain = self.conversational_chain if use_conversational else self.qa_chain
            if not chain:
                raise ValueError("未提供有效的问答链")
            
            # 创建评估器
            if not evaluator_types:
                evaluator_types = ["accuracy", "relevance", "helpfulness"]
            
            evaluators = EvaluatorFactory.create_multiple_evaluators(evaluator_types)
            
            # 创建预测函数
            def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
                question = inputs.get("question", "")
                try:
                    if hasattr(chain, 'invoke'):
                        result = chain.invoke(question=question, session_id="langsmith_eval")
                        return {"answer": result.get("answer", "")}
                    else:
                        answer = str(chain.invoke({"question": question}))
                        return {"answer": answer}
                except Exception as e:
                    return {"answer": f"错误: {str(e)}"}
            
            # 运行评估
            experiment_name = experiment_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment = evaluate(
                predict,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=experiment_name,
                client=langsmith_manager.client
            )
            
            logger.info(f"LangSmith 评估已启动，实验名称: {experiment_name}")
            return experiment_name
            
        except Exception as e:
            logger.error(f"LangSmith 评估失败: {str(e)}")
            return None


class EvaluationManager:
    """评估管理器"""
    
    def __init__(self, reports_dir: str = "./evaluation_reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
    def save_report(self, report: EvaluationReport) -> str:
        """保存评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.dataset_name}_{timestamp}.json"
        file_path = self.reports_dir / filename
        
        report.save_to_file(str(file_path))
        return str(file_path)
    
    def load_report(self, file_path: str) -> Optional[EvaluationReport]:
        """加载评估报告"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重构 EvaluationResult 对象
            individual_results = []
            for result_data in data.get("individual_results", []):
                result = EvaluationResult(**result_data)
                individual_results.append(result)
            
            data["individual_results"] = individual_results
            
            return EvaluationReport(**data)
            
        except Exception as e:
            logger.error(f"加载评估报告失败: {str(e)}")
            return None
    
    def list_reports(self) -> List[str]:
        """列出所有评估报告"""
        report_files = list(self.reports_dir.glob("*.json"))
        return [str(f) for f in report_files]
    
    def generate_summary_report(self, reports: List[EvaluationReport]) -> Dict[str, Any]:
        """生成汇总报告"""
        if not reports:
            return {}
        
        summary = {
            "total_reports": len(reports),
            "datasets_evaluated": list(set(report.dataset_name for report in reports)),
            "avg_scores_by_evaluator": {},
            "performance_trends": [],
            "best_performing_dataset": None,
            "worst_performing_dataset": None
        }
        
        # 计算各评估器的平均分数
        all_evaluator_names = set()
        for report in reports:
            all_evaluator_names.update(report.avg_scores.keys())
        
        for evaluator_name in all_evaluator_names:
            scores = [
                report.avg_scores.get(evaluator_name, 0.0) 
                for report in reports 
                if evaluator_name in report.avg_scores
            ]
            summary["avg_scores_by_evaluator"][evaluator_name] = {
                "mean": sum(scores) / len(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0
            }
        
        # 找出表现最好和最差的数据集
        if "overall" in summary["avg_scores_by_evaluator"]:
            best_score = 0.0
            worst_score = 1.0
            
            for report in reports:
                overall_score = report.avg_scores.get("overall", 0.0)
                if overall_score > best_score:
                    best_score = overall_score
                    summary["best_performing_dataset"] = {
                        "name": report.dataset_name,
                        "score": overall_score,
                        "timestamp": report.timestamp
                    }
                if overall_score < worst_score:
                    worst_score = overall_score
                    summary["worst_performing_dataset"] = {
                        "name": report.dataset_name,
                        "score": overall_score,
                        "timestamp": report.timestamp
                    }
        
        return summary 