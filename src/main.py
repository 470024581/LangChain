#!/usr/bin/env python3
"""
Document Q&A System Main Entry File
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import atexit
import asyncio

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vectorstores.vector_store import VectorStoreManager
from src.chains.qa_chain import DocumentQAChain
from src.workflows.multi_agent_workflow import MultiAgentWorkflow
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce log level for noisy background services to keep console clean
logging.getLogger("src.vectorstores.file_watcher").setLevel(logging.WARNING)
logging.getLogger("src.vectorstores.dynamic_vector_store").setLevel(logging.WARNING)

# Global list to hold background services that need cleanup
background_services = []
main_event_loop = None

def shutdown_services():
    """Shutdown all registered background services."""
    logger.info("Shutting down background services...")
    global main_event_loop

    async_tasks = []
    for service in background_services:
        if hasattr(service, 'cleanup'):
            if asyncio.iscoroutinefunction(service.cleanup):
                async_tasks.append(service.cleanup())
            else:
                try:
                    service.cleanup()
                    logger.info(f"Successfully shut down sync service {service.__class__.__name__}.")
                except Exception as e:
                    logger.error(f"Error shutting down sync service {service.__class__.__name__}: {e}", exc_info=True)

    if async_tasks and main_event_loop and not main_event_loop.is_closed():
        logger.info(f"Running {len(async_tasks)} async cleanup tasks...")
        try:
            # Set the loop as the current one for this context, in case it was unset.
            asyncio.set_event_loop(main_event_loop)
            # Create a single task to run all cleanup coroutines concurrently
            all_cleanup_tasks = asyncio.gather(*async_tasks, return_exceptions=True)
            results = main_event_loop.run_until_complete(all_cleanup_tasks)
            
            # Process results
            cleaned_services = [s for s in background_services if asyncio.iscoroutinefunction(s.cleanup)]
            for result, service in zip(results, cleaned_services):
                if isinstance(result, Exception):
                    logger.error(f"Error during async cleanup of {service.__class__.__name__}: {result}", exc_info=result)
                else:
                    logger.info(f"Successfully shut down async service {service.__class__.__name__}.")

        except Exception as e:
            logger.error(f"Failed to run async cleanup tasks: {e}", exc_info=True)


# Register the shutdown function to be called on exit
atexit.register(shutdown_services)

def build_vector_store(force_rebuild: bool = False, use_openai: bool = False):
    """Build vector store"""
    logger.info("Starting to build vector store...")
    
    try:
        vector_manager = VectorStoreManager(use_openai_embeddings=use_openai)
        vector_manager.get_or_create_vector_store(force_recreate=force_rebuild)
        logger.info("Vector store construction completed")
        if use_openai and hasattr(vector_manager, 'cleanup'):
            background_services.append(vector_manager)
        return vector_manager
    except Exception as e:
        logger.error(f"Failed to build vector store: {str(e)}")
        raise





def start_api_server():
    """Start API server"""
    logger.info("Starting API server...")
    
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
        logger.error(f"Failed to start API server: {str(e)}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Document Q&A System")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Interactive mode
    chat_parser = subparsers.add_parser("chat", help="Start interactive Q&A")
    chat_parser.add_argument("--model", default=None, help="Specify the model to use")
    chat_parser.add_argument("--session", default="default", help="Session ID")
    chat_parser.add_argument("--no-memory", action="store_true", help="Do not use memory function")
    chat_parser.add_argument("--conversational", action="store_true", help="Use conversational retrieval")
    chat_parser.add_argument("--use-agent", action="store_true", help="Use Agent architecture instead of Chain architecture")
    chat_parser.add_argument("--use-workflow", action="store_true", help="Use multi-agent workflow (LangGraph)")
    # Dynamic vector store enabled by default, only provide disable option
    chat_parser.add_argument(
        "--no-dynamic", 
        action="store_true", 
        help="Disable dynamic vector store, use traditional static mode"
    )
    
    # Server mode
    server_parser = subparsers.add_parser("server", help="Start API server")
    
    # Vector store management
    vector_parser = subparsers.add_parser("vector", help="Vector store management")
    vector_subparsers = vector_parser.add_subparsers(dest="vector_action")
    
    rebuild_parser = vector_subparsers.add_parser("rebuild", help="Rebuild vector store")
    rebuild_parser.add_argument("--force", action="store_true", help="Force rebuild")
    
    info_parser = vector_subparsers.add_parser("info", help="Show vector store information")
    
    # Evaluation commands
    eval_parser = subparsers.add_parser("eval", help="Model evaluation")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_action")
    
    # Create dataset
    create_dataset_parser = eval_subparsers.add_parser("create-dataset", help="Create evaluation dataset")
    create_dataset_parser.add_argument("--name", help="Dataset name (required when creating custom dataset)")
    create_dataset_parser.add_argument("--description", default="", help="Dataset description")
    create_dataset_parser.add_argument("--default", action="store_true", help="Create default dataset")
    
    # List datasets
    list_datasets_parser = eval_subparsers.add_parser("list-datasets", help="List all datasets")
    
    # Run evaluation
    run_eval_parser = eval_subparsers.add_parser("run", help="Run evaluation")
    run_eval_parser.add_argument("--dataset", required=True, help="Dataset name")
    run_eval_parser.add_argument("--evaluators", nargs="+", 
                                default=["accuracy", "relevance", "helpfulness", "groundedness"],
                                help="Evaluator types")
    run_eval_parser.add_argument("--conversational", action="store_true", help="Use conversational retrieval chain")
    run_eval_parser.add_argument("--concurrency", type=int, default=3, help="Concurrency count")
    
    # List reports
    list_reports_parser = eval_subparsers.add_parser("list-reports", help="List evaluation reports")
    
    # View report
    view_report_parser = eval_subparsers.add_parser("view-report", help="View evaluation report")
    view_report_parser.add_argument("--file", required=True, help="Report file path")
    
    # Generate summary
    summary_parser = eval_subparsers.add_parser("summary", help="Generate evaluation summary")
    
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
            logger.error(f"Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        if logger.level == logging.DEBUG:
            import traceback
            traceback.print_exc()


def handle_eval_command(args):
    """Handle evaluation command"""
    if not args.eval_action:
        logger.error("Please specify evaluation action")
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
        logger.error(f"Unknown evaluation action: {args.eval_action}")


def handle_create_dataset(args):
    """Handle create dataset command"""
    try:
        from src.evaluation.datasets import DatasetManager, DatasetBuilder
        
        dataset_manager = DatasetManager()
        
        if args.default:
            # Create default dataset
            dataset_manager.create_default_datasets()
            datasets = dataset_manager.list_datasets()
            logger.info(f"Default datasets created: {datasets}")
        else:
            # Create custom dataset - name required
            if not args.name:
                logger.error("Must provide --name parameter when creating custom dataset")
                return
            
            from src.evaluation.datasets import EvaluationDataset
            dataset = EvaluationDataset(name=args.name, description=args.description)
            file_path = dataset_manager.save_dataset(dataset)
            logger.info(f"Custom dataset '{args.name}' created: {file_path}")
            
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")


def handle_list_datasets():
    """Handle list datasets command"""
    try:
        from src.evaluation.datasets import DatasetManager
        
        dataset_manager = DatasetManager()
        datasets = dataset_manager.list_datasets()
        
        if not datasets:
            logger.info("No datasets found")
        else:
            logger.info("Available datasets:")
            for dataset in datasets:
                logger.info(f"  - {dataset}")
            
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")


def handle_run_evaluation(args):
    """Handle run evaluation command"""
    try:
        from src.evaluation.runners import EvaluationRunner
        
        # Validate dataset exists
        from src.evaluation.datasets import DatasetManager
        dataset_manager = DatasetManager()
        available_datasets = dataset_manager.list_datasets()
        
        if args.dataset not in available_datasets:
            logger.error(f"Dataset '{args.dataset}' not found. Available: {available_datasets}")
            return
        
        # Create evaluation runner
        runner = EvaluationRunner(
            dataset_name=args.dataset,
            evaluators=args.evaluators,
            conversational_retrieval=args.conversational,
            concurrency=args.concurrency
        )
        
        logger.info(f"Starting evaluation on dataset '{args.dataset}'...")
        logger.info(f"Evaluators: {args.evaluators}")
        logger.info(f"Concurrency: {args.concurrency}")
        logger.info(f"Conversational: {args.conversational}")
        
        # Run evaluation asynchronously
        async def run_async_evaluation():
            try:
                results = await runner.run_evaluation()
                logger.info(f"Evaluation completed successfully")
                logger.info(f"Results saved to: {results['report_file']}")
                return results
            except Exception as e:
                logger.error(f"Evaluation failed: {str(e)}")
                raise
        
        # Run in event loop
        asyncio.run(run_async_evaluation())
        
    except Exception as e:
        logger.error(f"Failed to run evaluation: {str(e)}")


def handle_list_reports():
    """Handle list reports command"""
    try:
        import glob
        
        reports_dir = Path("evaluation_reports")
        if not reports_dir.exists():
            logger.info("No evaluation reports directory found")
            return
        
        report_files = list(reports_dir.glob("*.json"))
        
        if not report_files:
            logger.info("No evaluation reports found")
        else:
            logger.info("Evaluation reports:")
            for report_file in sorted(report_files, reverse=True):
                logger.info(f"  - {report_file.name}")
            
    except Exception as e:
        logger.error(f"Failed to list reports: {str(e)}")


def handle_view_report(args):
    """Handle view report command"""
    try:
        import json
        from pathlib import Path
        
        report_path = Path(args.file)
        
        if not report_path.exists():
            logger.error(f"Report file not found: {args.file}")
            return
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        logger.info(f"Report: {report_path.name}")
        logger.info("=" * 50)
        
        # Print basic information
        logger.info(f"Dataset: {report.get('dataset', 'Unknown')}")
        logger.info(f"Timestamp: {report.get('timestamp', 'Unknown')}")
        logger.info(f"Total Questions: {report.get('total_questions', 0)}")
        
        # Print overall scores
        overall_scores = report.get('overall_scores', {})
        if overall_scores:
            logger.info("\nOverall Scores:")
            for metric, score in overall_scores.items():
                logger.info(f"  {metric}: {score:.4f}")
        
        # Print individual results summary
        individual_results = report.get('individual_results', [])
        if individual_results:
            logger.info(f"\nSample Results (first 3):")
            for i, result in enumerate(individual_results[:3]):
                logger.info(f"\nQuestion {i+1}: {result.get('question', '')[:100]}...")
                logger.info(f"Answer: {result.get('answer', '')[:100]}...")
                scores = result.get('scores', {})
                for metric, score in scores.items():
                    logger.info(f"  {metric}: {score}")
        
    except Exception as e:
        logger.error(f"Failed to view report: {str(e)}")


def handle_evaluation_summary():
    """Handle evaluation summary command"""
    try:
        import glob
        import json
        from pathlib import Path
        from collections import defaultdict
        
        reports_dir = Path("evaluation_reports")
        if not reports_dir.exists():
            logger.info("No evaluation reports directory found")
            return
        
        report_files = list(reports_dir.glob("*.json"))
        
        if not report_files:
            logger.info("No evaluation reports found")
            return
        
        # Aggregate statistics
        dataset_stats = defaultdict(list)
        
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                dataset = report.get('dataset', 'Unknown')
                overall_scores = report.get('overall_scores', {})
                
                dataset_stats[dataset].append({
                    'file': report_file.name,
                    'timestamp': report.get('timestamp', ''),
                    'scores': overall_scores
                })
                
            except Exception as e:
                logger.warning(f"Failed to read report {report_file}: {e}")
        
        # Print summary
        logger.info("Evaluation Summary")
        logger.info("=" * 50)
        
        for dataset, reports in dataset_stats.items():
            logger.info(f"\nDataset: {dataset}")
            logger.info(f"Reports count: {len(reports)}")
            
            # Calculate average scores
            if reports:
                all_metrics = set()
                for report in reports:
                    all_metrics.update(report['scores'].keys())
                
                avg_scores = {}
                for metric in all_metrics:
                    scores = [r['scores'].get(metric, 0) for r in reports if metric in r['scores']]
                    if scores:
                        avg_scores[metric] = sum(scores) / len(scores)
                
                logger.info("Average scores:")
                for metric, score in avg_scores.items():
                    logger.info(f"  {metric}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")


def handle_chat_command(args):
    """Handle chat command"""
    logger.info("Starting interactive chat mode...")
    global main_event_loop
    
    try:
        use_dynamic = not args.no_dynamic
        vector_manager = None

        # Initialize VectorStoreManager (dynamic or static)
        if use_dynamic:
            logger.info("Attempting to start dynamic vector store...")
            try:
                from src.vectorstores.dynamic_vector_store import DynamicVectorStoreManager
                
                async def init_dynamic_store():
                    dynamic_manager = DynamicVectorStoreManager(enable_file_watching=True)
                    await dynamic_manager.initialize()
                    return dynamic_manager

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        raise RuntimeError("Event loop is closed")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                main_event_loop = loop  # Store the loop
                vector_manager = loop.run_until_complete(init_dynamic_store())
                logger.info("Dynamic vector store started successfully.")
                if use_dynamic and hasattr(vector_manager, 'cleanup'):
                    background_services.append(vector_manager)
            except Exception as e:
                logger.error(f"Failed to start dynamic vector store: {e}")
                logger.info("Falling back to static vector store.")
                use_dynamic = False # Disable dynamic features if it failed
        
        if not use_dynamic:
            logger.info("Using static vector store.")
            vector_manager = VectorStoreManager()
            vector_manager.get_or_create_vector_store()

        # Setup chat based on architecture
        if args.use_workflow:
            logger.info("Using LangGraph multi-agent workflow...")
            from src.workflows.multi_agent_workflow import MultiAgentWorkflow
            workflow = MultiAgentWorkflow(
                vector_store_manager=vector_manager,
                model_name=args.model
            )
            start_workflow_chat(workflow, args.session, use_dynamic, vector_manager)
            
        elif args.use_agent:
            logger.info("Using Agent architecture...")
            from src.agents.rag_agent import DocumentQAAgent
            from src.agents.sql_agent import SQLAgent
            
            doc_agent = DocumentQAAgent(
                vector_store_manager=vector_manager,
                model_name=args.model,
                use_memory=not args.no_memory
            )
            sql_agent = SQLAgent(
                model_name=args.model,
                use_memory=not args.no_memory
            )
            start_agent_chat(doc_agent, sql_agent, args.session, use_dynamic, vector_manager)
            
        else:
            # Default to Chain architecture
            logger.info("Using Chain architecture...")
            from src.chains.qa_chain import DocumentQAChain, ConversationalRetrievalChain
            from src.agents.sql_agent import SQLAgent
            
            if args.conversational:
                doc_chain = ConversationalRetrievalChain(
                    vector_store_manager=vector_manager,
                    model_name=args.model
                )
            else:
                doc_chain = DocumentQAChain(
                    vector_store_manager=vector_manager,
                    model_name=args.model,
                    use_memory=not args.no_memory
                )
            
            sql_agent = SQLAgent(
                model_name=args.model,
                use_memory=not args.no_memory
            )
            start_chain_chat(doc_chain, sql_agent, args.session, use_dynamic, vector_manager)
            
    except Exception as e:
        logger.error(f"Failed to start chat mode: {str(e)}")
        import traceback
        traceback.print_exc()


def start_workflow_chat(workflow, session_id, use_dynamic, vector_manager):
    """Start LangGraph workflow chat session"""
    logger.info("🚀 Workflow system started! Type 'help' for available commands.")
    if use_dynamic:
        logger.info("📁 Dynamic vector store is active.")
        logger.info("💡 Commands: help, status, sync, files, quit")
    else:
        logger.info("📚 Static vector store mode.")
        logger.info("💡 Commands: help, quit")
    print()

    while True:
        try:
            user_input = input("👤 You> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help           - Show this help message")
                if use_dynamic:
                    print("  status         - Show vector store status")
                    print("  sync           - Force sync with filesystem")
                    print("  files          - List tracked files")
                print("  quit/exit      - Exit the program")
                continue
            
            if use_dynamic and handle_dynamic_commands(user_input, vector_manager):
                continue

            print("🤔 Processing with workflow...")
            try:
                # Check if the workflow has an async invoke method on its graph attribute
                if hasattr(workflow.workflow, 'ainvoke'):
                    result = asyncio.run(workflow.workflow.ainvoke({
                        "user_question": user_input,
                        "session_id": session_id
                    }))
                else: # Fallback to sync invoke on the graph attribute
                    result = workflow.workflow.invoke({
                        "user_question": user_input,
                        "session_id": session_id
                    })

                answer = result.get("generated_answer", "No answer generated")
                print(f"🤖 {answer}")
                if "query_type" in result:
                    print(f"🔍 Type: {result['query_type']}, Review Score: {result.get('review_score', 'N/A')}")

            except Exception as e:
                print(f"❌ Error during workflow execution: {str(e)}")
                logger.error(f"Workflow error: {str(e)}", exc_info=True)
            
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {str(e)}")
            logger.error(f"Chat loop error: {str(e)}", exc_info=True)


def start_agent_chat(doc_agent, sql_agent, session_id, use_dynamic, vector_manager):
    """Start Agent chat session"""
    start_interactive_chat(
        doc_handler=doc_agent, 
        sql_handler=sql_agent, 
        session_id=session_id, 
        use_dynamic=use_dynamic, 
        vector_manager=vector_manager,
        system_name="Agent"
    )


def start_chain_chat(doc_chain, sql_agent, session_id, use_dynamic, vector_manager):
    """Start Chain chat session"""
    start_interactive_chat(
        doc_handler=doc_chain, 
        sql_handler=sql_agent, 
        session_id=session_id, 
        use_dynamic=use_dynamic, 
        vector_manager=vector_manager,
        system_name="Chain"
    )


def start_interactive_chat(doc_handler, sql_handler, session_id, use_dynamic, vector_manager, system_name=""):
    """Generic interactive chat loop for Agent and Chain modes."""
    current_mode = "doc"
    logger.info(f"🚀 {system_name} system started! Type 'help' for available commands.")
    if use_dynamic:
        logger.info("📁 Dynamic vector store is active.")
        logger.info("💡 Commands: mode doc, mode sql, help, status, sync, files, clear, quit")
    else:
        logger.info("📚 Static vector store mode.")
        logger.info("💡 Commands: mode doc, mode sql, help, clear, quit")
    print()

    while True:
        try:
            mode_indicator = "📄" if current_mode == "doc" else "🗃️"
            user_input = input(f"{mode_indicator} {current_mode.upper()}> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  mode doc       - Switch to document Q&A mode")
                print("  mode sql       - Switch to SQL query mode")
                print("  help           - Show this help message")
                if use_dynamic:
                    print("  status         - Show vector store status")
                    print("  sync           - Force sync with filesystem")
                    print("  files          - List tracked files")
                print("  clear          - Clear current mode memory")
                print("  quit/exit      - Exit the program")
                print(f"\nCurrent mode: {current_mode.upper()}")
                continue
                
            elif user_input.lower() == 'mode doc':
                current_mode = "doc"
                print("📄 Switched to document Q&A mode")
                continue
                
            elif user_input.lower() == 'mode sql':
                current_mode = "sql"
                print("🗃️ Switched to SQL query mode")
                continue
                
            elif user_input.lower() == 'clear':
                handler = doc_handler if current_mode == "doc" else sql_handler
                if hasattr(handler, 'clear_memory'):
                    handler.clear_memory(session_id)
                    print(f"💭 {current_mode.upper()} mode memory cleared")
                else:
                    print("This mode does not support clearing memory.")
                continue

            if use_dynamic and handle_dynamic_commands(user_input, vector_manager):
                continue
            
            print("🤔 Processing...")
            try:
                if current_mode == "doc":
                    result = doc_handler.invoke(user_input, session_id=session_id)
                    answer = result.get("answer", result.get("output", "No answer generated"))
                    print(f"📄 {answer}")
                    if "relevant_documents" in result and result["relevant_documents"] is not None:
                        doc_count = len(result["relevant_documents"])
                        print(f"📚 Found {doc_count} relevant documents")
                else: # SQL mode
                    result = sql_handler.query(user_input, session_id=session_id)
                    answer = result.get("answer", result.get("output", "No answer generated"))
                    print(f"🗃️ {answer}")
                    if result.get("success"):
                        print("✅ Query executed successfully")
                    elif "error" in result:
                        print(f"❌ Error: {result['error']}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"{system_name} error: {str(e)}", exc_info=True)
            
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred: {str(e)}")
            logger.error(f"Chat loop error: {str(e)}", exc_info=True)


def handle_dynamic_commands(user_input, vector_manager):
    """Handles commands specific to dynamic vector store mode. Returns True if command was handled."""
    command = user_input.lower()

    if command == 'status':
        if hasattr(vector_manager, 'get_status'):
            status = vector_manager.get_status()
            print("📊 Vector Store Status:")
            print(f"   - Files tracked: {status.get('files_tracked', 'N/A')}")
            print(f"   - Documents count: {status.get('documents_count', 'N/A')}")
            print(f"   - Last sync: {status.get('last_sync', 'N/A')}")
            processing = status.get('processing_files', [])
            if processing:
                print(f"   - Currently processing: {', '.join(processing)}")
        else:
            print("📊 Status not available for this vector store manager.")
        return True

    elif command == 'sync':
        if hasattr(vector_manager, 'force_sync_with_filesystem'):
            print("🔄 Syncing with filesystem...")
            try:
                asyncio.run(vector_manager.force_sync_with_filesystem())
                print("✅ Filesystem sync completed")
            except Exception as e:
                print(f"❌ Sync failed: {e}")
        else:
            print("🔄 Sync functionality not available.")
        return True

    elif command == 'files':
        if hasattr(vector_manager, 'get_file_document_mapping'):
            files = vector_manager.get_file_document_mapping()
            if files:
                print("📁 Tracked files:")
                for file_path, doc_ids in files.items():
                    print(f"   - {file_path}: {len(doc_ids)} document chunks")
            else:
                print("📁 No files currently tracked.")
        else:
            print("📁 File listing not available.")
        return True
    
    return False


def handle_server_command():
    """Wrapper to start API server"""
    start_api_server()

def handle_vector_command(args):
    """Wrapper for vector store commands"""
    if not args.vector_action:
        logger.error("Please specify a vector store action (e.g., 'rebuild', 'info')")
        return

    use_openai_emb = settings.embeddings_provider == "openai"
    
    if args.vector_action == "rebuild":
        logger.info(f"Rebuilding vector store (force={args.force})...")
        build_vector_store(force_rebuild=args.force, use_openai=use_openai_emb)
        logger.info("Vector store rebuild completed.")
    
    elif args.vector_action == "info":
        logger.info("Vector store information:")
        vector_store = VectorStoreManager(use_openai_embeddings=use_openai_emb)
        vector_store.load_vector_store()
        if vector_store.vector_store:
            num_docs = vector_store.vector_store.index.ntotal
            logger.info(f"  - Number of documents: {num_docs}")
            logger.info(f"  - Store location: {vector_store.get_store_path('default')}")
        else:
            logger.warning("Could not load vector store to get info.")

if __name__ == "__main__":
    main() 