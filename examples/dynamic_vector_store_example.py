#!/usr/bin/env python3
"""
动态FAISS向量存储使用示例

演示如何使用基于watchdog和MCP的动态向量存储系统
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.vectorstores.dynamic_vector_store import DynamicVectorStoreManager
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """主函数演示动态向量存储的功能"""
    
    print("🚀 动态FAISS向量存储示例")
    print("=" * 50)
    
    # 1. 初始化动态向量存储管理器
    print("\n1️⃣ 初始化动态向量存储管理器...")
    
    vector_manager = DynamicVectorStoreManager(
        use_openai_embeddings=False,  # 使用免费的HuggingFace embeddings
        enable_file_watching=True,    # 启用文件监控
        enable_mcp=True              # 启用MCP支持
    )
    
    # 2. 初始化向量存储
    print("\n2️⃣ 初始化向量存储...")
    
    try:
        vector_store = await vector_manager.initialize()
        print("✅ 动态向量存储初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        return
    
    # 3. 显示初始状态
    print("\n3️⃣ 显示初始状态...")
    
    status = vector_manager.get_status()
    print(f"📊 状态信息:")
    print(f"   - 文件监控: {'开启' if status['file_watching_enabled'] else '关闭'}")
    print(f"   - MCP支持: {'开启' if status['mcp_enabled'] else '关闭'}")
    print(f"   - 跟踪文件数: {status['tracked_files_count']}")
    print(f"   - 文件监控运行: {'是' if status['file_watcher_running'] else '否'}")
    
    # 4. 显示跟踪的文件
    print("\n4️⃣ 当前跟踪的文件:")
    
    file_mapping = vector_manager.get_file_document_mapping()
    if file_mapping:
        for file_path, doc_ids in file_mapping.items():
            print(f"   📄 {file_path} ({len(doc_ids)} 个文档)")
    else:
        print("   📭 暂无跟踪的文件")
    
    # 5. 演示搜索功能
    print("\n5️⃣ 演示向量搜索功能...")
    
    if vector_store:
        try:
            # 执行相似性搜索
            query = "什么是人工智能？"
            results = vector_manager.similarity_search(query, k=3)
            
            print(f"🔍 搜索查询: {query}")
            print(f"📋 搜索结果 ({len(results)} 个):")
            
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"   {i}. 来源: {source}")
                print(f"      内容: {content}")
                
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
    
    # 6. 演示文件系统同步
    print("\n6️⃣ 演示文件系统同步...")
    
    try:
        await vector_manager.force_sync_with_filesystem()
        print("✅ 文件系统同步完成")
        
        # 显示同步后的状态
        updated_status = vector_manager.get_status()
        print(f"📊 同步后状态: 跟踪文件数 {updated_status['tracked_files_count']}")
        
    except Exception as e:
        print(f"❌ 同步失败: {str(e)}")
    
    # 7. 交互式演示
    print("\n7️⃣ 交互式演示")
    print("现在您可以:")
    print("- 在 data/document/ 目录下添加新文件")
    print("- 修改现有文件")
    print("- 删除文件")
    print("系统会自动检测变化并更新向量存储")
    print("\n输入 'quit' 退出演示")
    
    while True:
        try:
            command = input("\n💬 请输入命令 (search/status/files/sync/quit): ").strip().lower()
            
            if command == 'quit':
                break
            
            elif command == 'search':
                query = input("🔍 请输入搜索查询: ").strip()
                if query:
                    try:
                        results = vector_manager.similarity_search(query, k=3)
                        print(f"📋 搜索结果 ({len(results)} 个):")
                        
                        for i, doc in enumerate(results, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                            print(f"   {i}. 来源: {source}")
                            print(f"      内容: {content}\n")
                            
                    except Exception as e:
                        print(f"❌ 搜索失败: {str(e)}")
            
            elif command == 'status':
                status = vector_manager.get_status()
                print("📊 当前状态:")
                print(f"   - 跟踪文件数: {status['tracked_files_count']}")
                print(f"   - 处理中文件数: {status['processing_files_count']}")
                print(f"   - 文件监控: {'运行中' if status['file_watcher_running'] else '未运行'}")
                print(f"   - MCP可用: {'是' if status['mcp_available'] else '否'}")
            
            elif command == 'files':
                mapping = vector_manager.get_file_document_mapping()
                processing = vector_manager.get_processing_files()
                
                print(f"📁 跟踪的文件 ({len(mapping)} 个):")
                for file_path, doc_ids in mapping.items():
                    status_icon = "🔄" if file_path in processing else "✅"
                    print(f"   {status_icon} {file_path} ({len(doc_ids)} 个文档)")
                
                if processing:
                    print(f"\n🔄 正在处理的文件:")
                    for file_path in processing:
                        print(f"   - {file_path}")
            
            elif command == 'sync':
                print("🔄 开始同步文件系统...")
                try:
                    await vector_manager.force_sync_with_filesystem()
                    print("✅ 同步完成")
                except Exception as e:
                    print(f"❌ 同步失败: {str(e)}")
            
            else:
                print("❓ 未知命令，请输入: search/status/files/sync/quit")
                
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断")
            break
        except Exception as e:
            print(f"❌ 执行命令时出错: {str(e)}")
    
    # 8. 清理资源
    print("\n8️⃣ 清理资源...")
    
    try:
        await vector_manager.cleanup()
        print("✅ 资源清理完成")
    except Exception as e:
        print(f"❌ 清理失败: {str(e)}")
    
    print("\n🎉 动态向量存储演示完成！")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main()) 