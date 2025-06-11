#!/usr/bin/env python3
"""
文档问答系统主入口文件
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstores.vector_store import VectorStoreManager
from src.chains.qa_chain import DocumentQAChain
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_vector_store(force_rebuild: bool = False, use_openai: bool = False):
    """构建向量存储"""
    logger.info("开始构建向量存储...")
    
    try:
        vector_manager = VectorStoreManager(use_openai_embeddings=use_openai)
        vector_manager.get_or_create_vector_store(force_recreate=force_rebuild)
        logger.info("向量存储构建完成")
        return vector_manager
    except Exception as e:
        logger.error(f"构建向量存储失败: {str(e)}")
        raise





def start_api_server():
    """启动API服务器"""
    logger.info("启动API服务器...")
    
    try:
        import uvicorn
        from src.api.main import app
        
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=False
        )
    except Exception as e:
        logger.error(f"启动API服务器失败: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档问答系统")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 交互模式
    chat_parser = subparsers.add_parser("chat", help="启动交互式问答")
    chat_parser.add_argument("--model", default=None, help="指定使用的模型")
    chat_parser.add_argument("--session", default="default", help="会话ID")
    chat_parser.add_argument("--no-memory", action="store_true", help="不使用记忆功能")
    chat_parser.add_argument("--conversational", action="store_true", help="使用对话式检索")
    chat_parser.add_argument("--use-agent", action="store_true", help="使用Agent架构而非Chain架构")
    
    # 服务器模式
    server_parser = subparsers.add_parser("server", help="启动API服务器")
    
    # 向量存储管理
    vector_parser = subparsers.add_parser("vector", help="向量存储管理")
    vector_subparsers = vector_parser.add_subparsers(dest="vector_action")
    
    rebuild_parser = vector_subparsers.add_parser("rebuild", help="重建向量存储")
    rebuild_parser.add_argument("--force", action="store_true", help="强制重建")
    
    info_parser = vector_subparsers.add_parser("info", help="显示向量存储信息")
    
    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="模型评估")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_action")
    
    # 创建数据集
    create_dataset_parser = eval_subparsers.add_parser("create-dataset", help="创建评估数据集")
    create_dataset_parser.add_argument("--name", help="数据集名称（创建自定义数据集时必需）")
    create_dataset_parser.add_argument("--description", default="", help="数据集描述")
    create_dataset_parser.add_argument("--default", action="store_true", help="创建默认数据集")
    
    # 列出数据集
    list_datasets_parser = eval_subparsers.add_parser("list-datasets", help="列出所有数据集")
    
    # 运行评估
    run_eval_parser = eval_subparsers.add_parser("run", help="运行评估")
    run_eval_parser.add_argument("--dataset", required=True, help="数据集名称")
    run_eval_parser.add_argument("--evaluators", nargs="+", 
                                default=["accuracy", "relevance", "helpfulness", "groundedness"],
                                help="评估器类型")
    run_eval_parser.add_argument("--conversational", action="store_true", help="使用对话式检索链")
    run_eval_parser.add_argument("--concurrency", type=int, default=3, help="并发数量")
    
    # 列出报告
    list_reports_parser = eval_subparsers.add_parser("list-reports", help="列出评估报告")
    
    # 查看报告
    view_report_parser = eval_subparsers.add_parser("view-report", help="查看评估报告")
    view_report_parser.add_argument("--file", required=True, help="报告文件路径")
    
    # 生成汇总
    summary_parser = eval_subparsers.add_parser("summary", help="生成评估汇总")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "chat":
            handle_chat_command(args)
        elif args.command == "server":
            handle_server_command()
        elif args.command == "vector":
            handle_vector_command(args)
        elif args.command == "eval":
            handle_eval_command(args)
        else:
            logger.error(f"未知命令: {args.command}")
            
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"执行命令失败: {str(e)}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()


def handle_eval_command(args):
    """处理评估命令"""
    if not args.eval_action:
        logger.error("请指定评估操作")
        return
    
    if args.eval_action == "create-dataset":
        handle_create_dataset(args)
    elif args.eval_action == "list-datasets":
        handle_list_datasets()
    elif args.eval_action == "run":
        handle_run_evaluation(args)
    elif args.eval_action == "list-reports":
        handle_list_reports()
    elif args.eval_action == "view-report":
        handle_view_report(args)
    elif args.eval_action == "summary":
        handle_evaluation_summary()
    else:
        logger.error(f"未知的评估操作: {args.eval_action}")


def handle_create_dataset(args):
    """处理创建数据集命令"""
    try:
        from src.evaluation.datasets import DatasetManager, DatasetBuilder
        
        dataset_manager = DatasetManager()
        
        if args.default:
            # 创建默认数据集
            dataset_manager.create_default_datasets()
            datasets = dataset_manager.list_datasets()
            logger.info(f"默认数据集已创建: {datasets}")
        else:
            # 创建自定义数据集 - 需要名称
            if not args.name:
                logger.error("创建自定义数据集时必须提供 --name 参数")
                return
            
            from src.evaluation.datasets import EvaluationDataset
            dataset = EvaluationDataset(name=args.name, description=args.description)
            file_path = dataset_manager.save_dataset(dataset)
            logger.info(f"数据集 '{args.name}' 已创建: {file_path}")
            
    except Exception as e:
        logger.error(f"创建数据集失败: {str(e)}")


def handle_list_datasets():
    """处理列出数据集命令"""
    try:
        from src.evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        datasets = dataset_manager.list_datasets()
        
        if datasets:
            logger.info("可用的评估数据集:")
            for dataset_name in datasets:
                logger.info(f"  - {dataset_name}")
        else:
            logger.info("没有找到评估数据集")
            
    except Exception as e:
        logger.error(f"列出数据集失败: {str(e)}")


def handle_run_evaluation(args):
    """处理运行评估命令"""
    try:
        import asyncio
        from src.evaluation.datasets import DatasetManager
        from src.evaluation.runners import EvaluationRunner, EvaluationManager
        from src.vectorstores.vector_store import VectorStoreManager
        from src.chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
        
        # 初始化组件
        logger.info("初始化组件...")
        vector_store_manager = VectorStoreManager(use_openai_embeddings=False)
        vector_store_manager.get_or_create_vector_store()
        
        qa_chain = DocumentQAChain(vector_store_manager=vector_store_manager, use_memory=True)
        conversational_chain = ConversationalRetrievalChain(vector_store_manager=vector_store_manager)
        
        # 加载数据集
        dataset_manager = DatasetManager()
        dataset = dataset_manager.load_dataset(args.dataset)
        if not dataset:
            logger.error(f"数据集 '{args.dataset}' 不存在")
            return
        
        # 创建评估运行器
        runner = EvaluationRunner(qa_chain=qa_chain, conversational_chain=conversational_chain)
        
        # 运行评估
        logger.info(f"开始评估数据集: {args.dataset}")
        
        async def run_async_evaluation():
            return await runner.run_evaluation(
                dataset=dataset,
                evaluator_types=args.evaluators,
                use_conversational=args.conversational,
                max_concurrency=args.concurrency
            )
        
        report = asyncio.run(run_async_evaluation())
        
        # 保存报告
        eval_manager = EvaluationManager()
        report_file = eval_manager.save_report(report)
        
        logger.info(f"评估完成，报告已保存: {report_file}")
        
    except Exception as e:
        logger.error(f"运行评估失败: {str(e)}")


def handle_list_reports():
    """处理列出报告命令"""
    try:
        from src.evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        reports = eval_manager.list_reports()
        
        if reports:
            logger.info("评估报告文件:")
            for report_file in reports:
                logger.info(f"  - {report_file}")
        else:
            logger.info("没有找到评估报告")
            
    except Exception as e:
        logger.error(f"列出报告失败: {str(e)}")


def handle_view_report(args):
    """处理查看报告命令"""
    try:
        from src.evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        report = eval_manager.load_report(args.file)
        
        if not report:
            logger.error(f"报告文件不存在: {args.file}")
            return
        
        # 显示报告摘要
        print("=" * 60)
        print(f"评估报告: {report.dataset_name}")
        print("=" * 60)
        print(f"时间戳: {report.timestamp}")
        print(f"总样例数: {report.total_examples}")
        print(f"执行时间: {report.execution_time:.2f}秒")
        print(f"评估器: {', '.join(report.evaluator_names)}")
        print("\n平均分数:")
        
        for evaluator_name, score in report.avg_scores.items():
            print(f"  {evaluator_name}: {score:.3f}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"查看报告失败: {str(e)}")


def handle_evaluation_summary():
    """处理评估汇总命令"""
    try:
        from src.evaluation.runners import EvaluationManager
        
        eval_manager = EvaluationManager()
        report_files = eval_manager.list_reports()
        
        reports = []
        for report_file in report_files:
            report = eval_manager.load_report(report_file)
            if report:
                reports.append(report)
        
        if not reports:
            logger.info("没有找到评估报告")
            return
        
        summary = eval_manager.generate_summary_report(reports)
        
        # 显示汇总信息
        print("=" * 60)
        print("评估汇总报告")
        print("=" * 60)
        print(f"总报告数: {summary.get('total_reports', 0)}")
        print(f"评估的数据集: {', '.join(summary.get('datasets_evaluated', []))}")
        
        print("\n各评估器平均分数:")
        for evaluator_name, stats in summary.get('avg_scores_by_evaluator', {}).items():
            print(f"  {evaluator_name}:")
            print(f"    均值: {stats['mean']:.3f}")
            print(f"    最小值: {stats['min']:.3f}")
            print(f"    最大值: {stats['max']:.3f}")
        
        best = summary.get('best_performing_dataset')
        if best:
            print(f"\n表现最好的数据集: {best['name']} (分数: {best['score']:.3f})")
        
        worst = summary.get('worst_performing_dataset')
        if worst:
            print(f"表现最差的数据集: {worst['name']} (分数: {worst['score']:.3f})")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"生成汇总报告失败: {str(e)}")


def handle_chat_command(args):
    """处理聊天命令"""
    try:
        from src.vectorstores.vector_store import VectorStoreManager
        from src.chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
        from src.agents.rag_agent import DocumentQAAgent, ConversationalRetrievalAgent
        from src.agents.sql_agent import SQLAgent
        
        # 初始化向量存储
        vector_manager = VectorStoreManager(use_openai_embeddings=False)
        vector_manager.get_or_create_vector_store()
        
        # 初始化问答系统（Chain或Agent）
        if args.use_agent:
            # 使用Agent架构
            if args.conversational:
                qa_system = ConversationalRetrievalAgent(vector_store_manager=vector_manager)
                print("使用对话式检索Agent 🤖")
            else:
                use_memory = not args.no_memory
                qa_system = DocumentQAAgent(
                    vector_store_manager=vector_manager,
                    model_name=args.model,
                    use_memory=use_memory
                )
                print(f"使用文档问答Agent 🤖，记忆功能: {'开启' if use_memory else '关闭'}")
        else:
            # 使用Chain架构
            if args.conversational:
                qa_system = ConversationalRetrievalChain(vector_store_manager=vector_manager)
                print("使用对话式检索Chain ⛓️")
            else:
                use_memory = not args.no_memory
                qa_system = DocumentQAChain(
                    vector_store_manager=vector_manager,
                    model_name=args.model,
                    use_memory=use_memory
                )
                print(f"使用标准问答Chain ⛓️，记忆功能: {'开启' if use_memory else '关闭'}")
        
        qa_chain = qa_system  # 保持变量名兼容性
        
        # 初始化SQL Agent
        try:
            sql_agent = SQLAgent(use_memory=True, verbose=False)
            sql_available = True
            print("SQL Agent 初始化成功")
        except Exception as e:
            sql_agent = None
            sql_available = False
            print(f"SQL Agent 初始化失败: {str(e)}")
        
        print("\n=== 智能问答系统 ===")
        print("📄 文档问答 | 🗃️ SQL查询")
        print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空记忆")
        print("输入 'sql:' 前缀进行SQL查询")
        print("输入 'mode doc' 切换到文档问答模式")
        print("输入 'mode sql' 切换到SQL查询模式")
        print("=" * 50)
        
        session_id = args.session
        current_mode = "doc"  # 默认文档模式
        print(f"当前模式: {'📄 文档问答' if current_mode == 'doc' else '🗃️ SQL查询'}")
        
        while True:
            try:
                mode_indicator = "📄" if current_mode == "doc" else "🗃️"
                question = input(f"\n{mode_indicator} 问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if question.lower() in ['clear', '清空']:
                    if current_mode == "doc":
                        if hasattr(qa_chain, 'clear_memory'):
                            qa_chain.clear_memory(session_id)
                        elif hasattr(qa_chain, 'memory_manager'):
                            qa_chain.memory_manager.clear_memory(session_id)
                    elif current_mode == "sql" and sql_available:
                        sql_agent.clear_memory(session_id)
                    print("记忆已清空")
                    continue
                
                if question.lower() == 'mode doc':
                    current_mode = "doc"
                    print("已切换到文档问答模式 📄")
                    continue
                
                if question.lower() == 'mode sql':
                    if sql_available:
                        current_mode = "sql"
                        print("已切换到SQL查询模式 🗃️")
                    else:
                        print("SQL Agent不可用")
                    continue
                
                if not question:
                    continue
                
                # 处理SQL查询（无论当前模式）
                if question.lower().startswith('sql:'):
                    if not sql_available:
                        print("SQL Agent不可用")
                        continue
                    
                    sql_question = question[4:].strip()
                    print("🗃️ SQL查询中...")
                    result = sql_agent.query(sql_question, session_id=session_id)
                    
                    print(f"\n答案: {result['answer']}")
                    if not result['success'] and result.get('error'):
                        print(f"错误: {result['error']}")
                    continue
                
                # 根据当前模式处理查询
                if current_mode == "doc":
                    print("📄 文档检索中...")
                    
                    if args.conversational:
                        if args.use_agent:
                            # Agent版本：对话式检索Agent
                            result = qa_chain.invoke(question=question, chat_history=[])
                        else:
                            # Chain版本：对话式检索链
                            result = qa_chain.invoke(question=question, session_id=session_id)
                    else:
                        # 标准问答（Agent或Chain）
                        result = qa_chain.invoke(question, session_id)
                    
                    print(f"\n答案: {result['answer']}")
                    
                    # 显示Agent的中间步骤（仅Agent版本）
                    if args.use_agent and result.get('intermediate_steps'):
                        print(f"\n🤖 Agent执行步骤({len(result['intermediate_steps'])}个):")
                        for i, step in enumerate(result['intermediate_steps'][:2], 1):
                            if hasattr(step, '__dict__'):
                                print(f"  {i}. {step}")
                            else:
                                print(f"  {i}. {str(step)[:100]}...")
                    
                    # 显示相关文档
                    if result.get('relevant_documents'):
                        print(f"\n📚 相关文档({len(result['relevant_documents'])}个):")
                        for i, doc in enumerate(result['relevant_documents'][:2], 1):
                            source = doc['metadata'].get('source_file', 'Unknown')
                            content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                            print(f"  {i}. 来源: {source}")
                            print(f"     内容: {content}")
                
                elif current_mode == "sql":
                    if not sql_available:
                        print("SQL Agent不可用，请切换到文档模式")
                        continue
                    
                    print("🗃️ SQL查询中...")
                    result = sql_agent.query(question, session_id=session_id)
                    
                    print(f"\n答案: {result['answer']}")
                    if not result['success'] and result.get('error'):
                        print(f"错误: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                print(f"错误: {str(e)}")
                
    except Exception as e:
        logger.error(f"初始化聊天模式失败: {str(e)}")


def handle_server_command():
    """处理服务器命令"""
    try:
        import uvicorn
        from src.api.main import app
        
        logger.info("启动API服务器...")
        uvicorn.run(
            app,
            host=settings.api_host,
            port=settings.api_port,
            reload=False
        )
    except Exception as e:
        logger.error(f"启动API服务器失败: {str(e)}")


def handle_vector_command(args):
    """处理向量存储命令"""
    if not args.vector_action:
        logger.error("请指定向量存储操作")
        return
    
    try:
        from src.vectorstores.vector_store import VectorStoreManager
        
        if args.vector_action == "rebuild":
            vector_manager = VectorStoreManager(use_openai_embeddings=False)
            vector_manager.get_or_create_vector_store(force_recreate=args.force)
            logger.info("向量存储重建完成")
            
        elif args.vector_action == "info":
            vector_manager = VectorStoreManager(use_openai_embeddings=False)
            vector_store = vector_manager.get_or_create_vector_store()
            
            print("=" * 50)
            print("向量存储信息")
            print("=" * 50)
            print(f"存储路径: {vector_manager.vector_store_path}")
            print(f"存储类型: {type(vector_store).__name__}")
            
            # 尝试获取文档数量
            try:
                if hasattr(vector_store, '_collection'):
                    count = vector_store._collection.count()
                    print(f"文档数量: {count}")
                else:
                    print("文档数量: 无法获取")
            except:
                print("文档数量: 无法获取")
            
            print("=" * 50)
            
        else:
            logger.error(f"未知的向量存储操作: {args.vector_action}")
            
    except Exception as e:
        logger.error(f"向量存储操作失败: {str(e)}")


if __name__ == "__main__":
    main() 