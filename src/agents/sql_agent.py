"""
SQL AI Agent Module
Intelligent SQL query agent built with create_sql_agent
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from langchain.agents import AgentType

from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import PromptTemplate

from ..models.llm_factory import LLMFactory
from ..memory.conversation_memory import ConversationMemoryManager
from ..utils.langsmith_utils import langsmith_manager, with_langsmith_tracing
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SQLAgent:
    """SQL AI Agent - Built with create_sql_agent"""
    
    def __init__(
        self,
        db_path: str = None,
        model_name: str = None,
        use_memory: bool = True,
        verbose: bool = True
    ):
        """
        Initialize SQL Agent
        
        Args:
            db_path: SQLite database file path
            model_name: Model name to use
            use_memory: Whether to use memory functionality
            verbose: Whether to show detailed output
        """
        self.db_path = db_path or self._get_default_db_path()
        self.model_name = model_name
        self.use_memory = use_memory
        self.verbose = verbose
        
        # Validate database file
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file does not exist: {self.db_path}")
        
        # Initialize components
        self.llm = LLMFactory.create_llm(model_name)
        self.db = self._create_database_connection()
        self.memory_manager = ConversationMemoryManager() if use_memory else None
        
        # Create SQL Agent
        self.agent_executor = self._create_sql_agent()
        
        # Configure LangSmith
        self._configure_langsmith()
        
        logger.info(f"SQL Agent initialization completed, database: {self.db_path}")
        if langsmith_manager.is_enabled:
            logger.info("LangSmith tracing enabled")
    
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        # Find database file from project root directory
        base_path = Path(__file__).parent.parent.parent
        db_path = base_path / "data" / "database" / "erp.db"
        return str(db_path)
    
    def _create_database_connection(self) -> SQLDatabase:
        """Create database connection"""
        try:
            # Build SQLite connection URI
            db_uri = f"sqlite:///{self.db_path}"
            
            # Create SQLDatabase instance
            db = SQLDatabase.from_uri(
                db_uri,
                # Set sample row count for providing table data examples
                sample_rows_in_table_info=3,
                # Set included tables (if restriction needed)
                include_tables=['products', 'inventory', 'sales']
            )
            
            logger.info(f"Database connection successful: {db_uri}")
            return db
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def _create_sql_agent(self) -> AgentExecutor:
        """Create SQL Agent (using custom prompt)"""
        try:
            # Create a simpler, more direct system prompt
            system_message = """You are an agent designed to interact with a SQLite database.
Given an input question, you must correctly generate and execute a SQL query to answer the question.

Here are some rules to follow:
1. You MUST use the `sql_db_query` tool to execute queries.
2. Your SQL queries must be syntactically correct for SQLite.
3. DO NOT end your SQL queries with a semicolon (`;`).
4. Always use the `sql_db_list_tables` tool first to see what tables are available.
5. After running a query and getting a result, if you have the answer, you MUST output it in the format: `Final Answer: <your answer>`.
6. Do not make up an answer. If you cannot find the answer, say "I could not find the answer."

If the question does not seem related to the database, just return "I don't know" as the answer."""

            # Create custom prompt template
            prompt = PromptTemplate.from_template(f"""{system_message}

Question: {{input}}
{{agent_scratchpad}}""")

            # Create SQL database toolkit and filter out the confusing 'sql_db_query_checker' tool
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            all_tools = toolkit.get_tools()
            tools = [tool for tool in all_tools if tool.name != "sql_db_query_checker"]

            # Use initialize_agent to create agent
            agent_executor = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                prompt=prompt,
                verbose=self.verbose,
                handle_parsing_errors=True
            )
            
            logger.info("SQL Agent created successfully")
            return agent_executor
            
        except Exception as e:
            logger.error(f"SQL Agent creation failed: {str(e)}")
            raise
    
    def _configure_langsmith(self):
        """Configure LangSmith tracing"""
        if langsmith_manager.is_enabled:
            callbacks = langsmith_manager.get_callbacks()
            if callbacks:
                self.langsmith_config = RunnableConfig(
                    callbacks=callbacks,
                    tags=["SQLAgent", f"memory_{self.use_memory}"],
                    metadata={
                        "db_path": self.db_path,
                        "use_memory": self.use_memory,
                        "model": getattr(self.llm, 'model_name', 'unknown')
                    }
                )
            else:
                self.langsmith_config = None
        else:
            self.langsmith_config = None
    
    @with_langsmith_tracing(name="SQLAgent.query", tags=["sql", "query"])
    def query(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute SQL query (synchronous method, compatible with existing code)
        
        Args:
            question: User's natural language question
            session_id: Session ID (if using memory)
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing query results
        """
        logger.info(f"Processing SQL query: {question[:100]}...")

        if not self.llm:
            logger.error("LLM not initialized, SQL Agent query cannot be executed")
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent functionality cannot be executed because LLM is not initialized.",
                "data": {"session_id": session_id},
                "error": "LLM not initialized, SQL Agent cannot be used."
            }

        try:
            # Check if it's a simple database structure query, which doesn't need historical context
            simple_queries = [
                "table", "database", "structure", "schema", 
                "column", "field", "what tables", "show tables"
            ]
            is_simple_query = any(keyword in question.lower() for keyword in simple_queries)
            
            # If using memory and not a simple query, get context from memory
            chat_history = ""
            if self.use_memory and self.memory_manager and not is_simple_query:
                history = self.memory_manager.get_chat_history(session_id)
                if history:
                    # Format chat history
                    history_texts = []
                    for msg in history[-4:]:  # Only take the last 4 rounds of conversation
                        if hasattr(msg, 'content'):
                            role = "User" if msg.type == "human" else "Assistant"
                            history_texts.append(f"{role}: {msg.content}")
                    if history_texts:
                        chat_history = "\n".join(history_texts)
                        question = f"Chat history:\n{chat_history}\n\nCurrent question: {question}"
                        logger.info("Using chat history context")
                else:
                    logger.info("No chat history found")
            elif is_simple_query:
                logger.info("Detected simple database query, skipping historical context")
            
            logger.info(f"Executing SQL Agent query: {question}")
            # Reference example code: Agent's invoke method expects a dictionary with "input" key
            if self.langsmith_config:
                result = self.agent_executor.invoke(
                    {"input": question}, 
                    config=self.langsmith_config
                )
            else:
                result = self.agent_executor.invoke({"input": question})
            
            # Extract answer
            answer = result.get("output", "Unable to get answer from SQL Agent.")
            logger.info(f"SQL Agent execution completed. Answer: {answer}")
            
            # Save to memory
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question.split("Current question: ")[-1] if "Current question: " in question else question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            # Build return result (reference example code return format)
            response = {
                "query": question,
                "query_type": "sql_agent",
                "success": True,
                "answer": answer,
                "data": {
                    "session_id": session_id,
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            }
            
            logger.info("SQL query processing completed")
            return response
            
        except Exception as e:
            error_msg = f"SQL query failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"Error occurred while executing SQL query: {str(e)}",
                "data": {"session_id": session_id},
                "error": str(e)
            }
    
    def query_sync(
        self, 
        question: str, 
        session_id: str = "default",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute SQL query synchronously (backward compatibility)
        
        Args:
            question: User's natural language question
            session_id: Session ID (if using memory)
            **kwargs: Other parameters
        
        Returns:
            Dictionary containing query results
        """
        logger.info(f"Processing synchronous SQL query: {question[:100]}...")

        if not self.llm:
            logger.error("LLM not initialized, SQL Agent query cannot be executed")
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent functionality cannot be executed because LLM is not initialized.",
                "data": {"session_id": session_id},
                "error": "LLM not initialized."
            }

        try:
            # If using memory, get context from memory
            chat_history = ""
            if self.use_memory and self.memory_manager:
                history = self.memory_manager.get_chat_history(session_id)
                if history:
                    # Format chat history
                    history_texts = []
                    for msg in history[-4:]:  # Only take the last 4 rounds of conversation
                        if hasattr(msg, 'content'):
                            role = "User" if msg.type == "human" else "Assistant"
                            history_texts.append(f"{role}: {msg.content}")
                    if history_texts:
                        chat_history = "\n".join(history_texts)
                        question = f"Chat history:\n{chat_history}\n\nCurrent question: {question}"
            
            # Execute query synchronously
            if self.langsmith_config:
                result = self.agent_executor.invoke(
                    {"input": question}, 
                    config=self.langsmith_config
                )
            else:
                result = self.agent_executor.invoke({"input": question})
            
            # Extract answer
            answer = result.get("output", "Unable to get answer from SQL Agent.")
            logger.info(f"Synchronous SQL Agent execution completed. Answer: {answer}")
            
            # Save to memory
            if self.use_memory and self.memory_manager:
                self.memory_manager.add_message_pair(
                    user_message=question.split("Current question: ")[-1] if "Current question: " in question else question,
                    ai_message=answer,
                    session_id=session_id
                )
            
            # Build return result
            response = {
                "query": question,
                "query_type": "sql_agent",
                "success": True,
                "answer": answer,
                "data": {
                    "session_id": session_id,
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            }
            
            logger.info("Synchronous SQL query processing completed")
            return response
            
        except Exception as e:
            error_msg = f"Synchronous SQL query failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "query": question, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"Error occurred while executing synchronous SQL query: {str(e)}",
                "data": {"session_id": session_id},
                "error": str(e)
            }
    
    async def query_table(
        self, 
        query: str, 
        table_name: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Query specified table (reference example code implementation)
        
        Args:
            query: User query
            table_name: Specified table name
            session_id: Session ID
        
        Returns:
            Query result dictionary
        """
        logger.info(f"Querying specified table {table_name}: {query}")

        if not self.llm:
            logger.error("LLM not initialized, SQL Agent query cannot be executed")
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": "SQL Agent functionality cannot be executed because LLM is not initialized.",
                "data": {"queried_table": table_name, "session_id": session_id},
                "error": "LLM not initialized."
            }

        try:
            logger.info(f"Initializing SQLDatabase for table {table_name}, using database: {self.db_path}")
            # Reference example code: SQLDatabase connects to main database but only includes specified table
            db_uri = f"sqlite:///{self.db_path}"
            db = SQLDatabase.from_uri(db_uri, include_tables=[table_name])
            
            logger.info(f"Creating SQL Agent for table {table_name}")
            # Reference example code: Create specialized SQL Agent
            sql_agent_executor = create_sql_agent(
                llm=self.llm, 
                db=db, 
                verbose=True, 
                handle_parsing_errors=True
            )
            
            logger.info(f"Executing SQL Agent with query: {query}")
            # Reference example code: Agent's ainvoke method expects a dictionary with "input" key
            response = await sql_agent_executor.ainvoke({"input": query})
            
            answer = response.get("output", "Unable to get answer from SQL Agent.")
            logger.info(f"SQL Agent execution for table {table_name} completed. Answer: {answer}")
            
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": True,
                "answer": answer,
                "data": {
                    "queried_table": table_name,
                    "session_id": session_id,
                    "intermediate_steps": response.get("intermediate_steps", [])
                }
            }

        except Exception as e:
            logger.error(f"SQL Agent query failed for table {table_name}: {e}", exc_info=True)
            return {
                "query": query, 
                "query_type": "sql_agent", 
                "success": False,
                "answer": f"Error occurred while executing SQL query on table '{table_name}'.",
                "data": {"queried_table": table_name, "session_id": session_id},
                "error": str(e)
            }

    async def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            # Get table list
            tables = self.db.get_usable_table_names()
            
            # Get table structure
            table_info = {}
            for table in tables:
                table_info[table] = self.db.get_table_info([table])
            
            return {
                "database_path": self.db_path,
                "tables": tables,
                "table_info": table_info,
                "dialect": self.db.dialect
            }
            
        except Exception as e:
            logger.error(f"Failed to get database information: {str(e)}")
            return {"error": str(e)}
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """Get sample data from table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = self.db.run(query)
            
            return {
                "table": table_name,
                "sample_data": result,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Failed to get sample data: {str(e)}")
            return {"error": str(e)}
    
    def clear_memory(self, session_id: str = "default") -> bool:
        """Clear session memory"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.clear_memory(session_id)
        return True
    
    def get_memory_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """Get memory statistics"""
        if self.use_memory and self.memory_manager:
            return self.memory_manager.get_memory_stats(session_id)
        return {}


# Convenience functions
def create_sql_agent_simple(
    db_path: str = None,
    model_name: str = None,
    **kwargs
) -> SQLAgent:
    """
    Create simple SQL Agent instance
    
    Args:
        db_path: Database file path
        model_name: Model name
        **kwargs: Other parameters
    
    Returns:
        SQLAgent instance
    """
    return SQLAgent(
        db_path=db_path,
        model_name=model_name,
        **kwargs
    )


# Convenience function implemented according to example code
async def get_sql_agent_response(
    query: str,
    db_path: str = None,
    model_name: str = None,
    table_name: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to get SQL Agent response following example code implementation
    
    Args:
        query: User query
        db_path: Database path
        model_name: Model name  
        table_name: Specific table name (for single table query)
        **kwargs: Other parameters
    
    Returns:
        Query result dictionary
    """
    try:
        # Create SQL Agent instance
        agent = SQLAgent(
            db_path=db_path,
            model_name=model_name,
            use_memory=False,  # Convenience function doesn't use memory
            verbose=False
        )
        
        # If table name is specified, use specific table query method
        if table_name:
            return await agent.query_table(query, table_name, session_id="convenience_function")
        else:
            # Otherwise use regular query (synchronous)
            return agent.query(query, session_id="convenience_function")
            
    except Exception as e:
        logger.error(f"Convenience function SQL Agent query failed: {str(e)}")
        return {
            "query": query,
            "query_type": "sql_agent",
            "success": False,
            "answer": f"SQL query execution failed: {str(e)}",
            "data": {"queried_table": table_name} if table_name else {},
            "error": str(e)
        } 