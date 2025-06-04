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


def interactive_qa():
    """交互式问答模式"""
    logger.info("启动交互式问答模式...")
    
    try:
        # 初始化向量存储
        vector_manager = VectorStoreManager(use_openai_embeddings=False)
        vector_manager.get_or_create_vector_store()
        
        # 初始化问答链
        qa_chain = DocumentQAChain(
            vector_store_manager=vector_manager,
            use_memory=True
        )
        
        print("\n=== 文档问答系统 ===")
        print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空记忆")
        print("=" * 50)
        
        session_id = "interactive"
        
        while True:
            try:
                question = input("\n问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if question.lower() in ['clear', '清空']:
                    qa_chain.clear_memory(session_id)
                    print("记忆已清空")
                    continue
                
                if not question:
                    continue
                
                print("正在思考...")
                result = qa_chain.invoke(question, session_id)
                
                print(f"\n答案: {result['answer']}")
                
                # 显示相关文档
                if result.get('relevant_documents'):
                    print(f"\n相关文档({len(result['relevant_documents'])}个):")
                    for i, doc in enumerate(result['relevant_documents'][:2], 1):
                        source = doc['metadata'].get('source_file', 'Unknown')
                        content = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                        print(f"  {i}. 来源: {source}")
                        print(f"     内容: {content}")
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                print(f"错误: {str(e)}")
                
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        print(f"初始化失败: {str(e)}")


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
    parser.add_argument(
        "command",
        choices=["build", "interactive", "server"],
        help="执行的命令: build=构建向量存储, interactive=交互式问答, server=启动API服务器"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重建向量存储"
    )
    parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="使用OpenAI embeddings（需要API密钥）"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "build":
            build_vector_store(
                force_rebuild=args.force_rebuild,
                use_openai=args.use_openai_embeddings
            )
            print("向量存储构建完成")
            
        elif args.command == "interactive":
            interactive_qa()
            
        elif args.command == "server":
            start_api_server()
            
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 