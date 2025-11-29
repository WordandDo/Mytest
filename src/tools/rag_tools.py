"""
RAG Tools - Query tool for RAG index

This module contains only the query tool. All index-related classes 
(RAGIndexLocal, RAGIndexLocal_faiss, get_rag_index_class) are defined in
envs/RAG_environment.py for better separation of concerns.
"""
    

class QueryRAGIndexTool:
    """
    Tool for querying a pre-built RAG index.
    
    This tool provides a simple interface for searching through indexed documents
    and retrieving relevant context based on a query.
    """
    name = "local_search"
    description = (
        "Search the pre-built RAG index for the most relevant text chunks. "
        "Use this tool to answer questions related to the indexed knowledge."
    )
    parameters = [
        {
            'name': 'query', 
            'type': 'string', 
            'description': 'Query text to look up in the index', 
            'required': True
        },
        {
            'name': 'top_k', 
            'type': 'integer', 
            'description': 'Number of top relevant text chunks to retrieve (default: 10)', 
            'required': False
        }
    ]

    def __init__(self, rag_index):
        """
        Initialize the query tool with a RAG index.
        
        Args:
            rag_index: 已经加载完成的本地 RAG 索引实例
                      (RAGIndexLocal 或 RAGIndexLocal_faiss)
        """
        self.rag_index = rag_index

    def call(self, params: dict, **kwargs) -> str:
        """
        Execute the query against the RAG index.
        
        Args:
            params: Dictionary containing:
                - query (str): The search query
                - top_k (int, optional): Number of results to return (default: 10)
            **kwargs: Additional keyword arguments (unused)
            
        Returns:
            str: Retrieved context from the index
        """
        query = params.get("query")
        top_k = params.get("top_k", 10)

        if not query:
            return "[查询工具错误] 'query' 参数是必需的"

        try:
            return self.rag_index.query(query, top_k=top_k)
        except Exception as e:
            return f"[查询工具错误] 搜索过程中发生异常: {str(e)}"
