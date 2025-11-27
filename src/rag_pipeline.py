import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
from src import config

# Setup Global Settings 
embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_ID)
Settings.embed_model = embed_model

def get_rag_engine():
    """Initialize and return RAG Query Engine with Hybrid Retrieval.
    
    This function creates a query engine that uses:
    - Hybrid Retrieval: Combines Vector Search and BM25 (Keyword Search)
    - Reranking: Uses SentenceTransformer to rerank results
    - Vietnamese Response: Prompt optimized for Vietnamese language answers
    
    Returns:
        RetrieverQueryEngine: Configured query engine ready to use
    """
    with st.spinner("STARTING..."):
        # Setup Embedding Model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        
        # Setup Reranker
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", 
            top_n=3 
        )
        
        # Kết nối Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name = "advanced-rag-project"
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # Load Indexes
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # BM25 Index 
        documents = SimpleDirectoryReader("./data").load_data()
        nodes = Settings.text_splitter.get_nodes_from_documents(documents)
        
        # Define Retrievers
        vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
        
        # Define Hybrid Retriever Class
        class HybridRetriever(BaseRetriever):
            """Hybrid Retriever combining Vector Search and BM25 Keyword Search.
            
            This class merges results from both vector retriever and BM25 retriever
            to improve search accuracy.
            """
            
            def __init__(self, vector_retriever, bm25_retriever):
                """Initialize Hybrid Retriever.
                
                Args:
                    vector_retriever: Vector-based retriever using embeddings
                    bm25_retriever: Keyword-based retriever using BM25 algorithm
                """
                self.vector_retriever = vector_retriever
                self.bm25_retriever = bm25_retriever
                super().__init__()

            def _retrieve(self, query_bundle):
                """Perform retrieval from both sources and merge results.
                
                Args:
                    query_bundle: Query bundle containing the search query
                    
                Returns:
                    list: List of merged nodes (duplicates removed)
                """
                vector_nodes = self.vector_retriever.retrieve(query_bundle)
                bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
                
                all_nodes = {}
                for node in vector_nodes: all_nodes[node.node.node_id] = node
                for node in bm25_nodes:
                    if node.node.node_id not in all_nodes:
                        all_nodes[node.node.node_id] = node
                return list(all_nodes.values())
        
        hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
        
        # Create Query Engine with Vietnamese response and strict relevance control
        from llama_index.core import PromptTemplate
        
        # QA Prompt Template for Vietnamese academic regulations chatbot
        # Ensures responses are strictly based on document context
        qa_prompt_template = PromptTemplate(
            """You are an AI assistant specializing in Can Tho University's academic regulations.
Your task is to answer the question based COMPLETELY on the information provided in the document.

MANDATORY RULES:
1. Only use information in the context below to answer.
2. If the question is NOT related to the content in the document, answer:
   "Thông tin không được đề cập trong tài liệu, vui lòng liên hệ trang dsa.ctu.edu.vn"
3. If unsure or the information is unclear, say:
   "Thông tin không được đề cập trong tài liệu, vui lòng liên hệ trang dsa.ctu.edu.vn"
4. DO NOT fabricate, infer, or add information outside the document.
5. Answer COMPLETELY in Vietnamese.
6. Answer briefly, concisely, and easily understood.

Context from the document:
{context_str}

Question: {query_str}

Answer in Vietnamese:"""
        )
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            text_qa_template=qa_prompt_template
        )
        
        return query_engine

def rewrite_query(query, llm):
    """Rewrite user's question into standardized form.
    
    Uses LLM to convert casual/slang questions into formal questions
    with administrative/legal terminology to improve search accuracy
    in regulatory documents.
    
    Args:
        query (str): Original question from user
        llm: LLM instance to perform rewrite
        
    Returns:
        str: Standardized question
        
    Example:
        >>> rewrite_query("Rớt môn có bị đuổi khỏi ktx ko?", llm)
        "Sinh viên nợ môn có bị chấm dứt hợp đồng ký túc xá không?"
    """
    # Query Rewrite Prompt: Converts casual Vietnamese to formal administrative language
    # Improves retrieval accuracy by standardizing terminology
    prompt = f"""You are an AI assistant specializing in Vietnamese semantics.
Your task is to rewrite the user's question in a formal, clear way, 
using administrative/legal terminology for easy searching in the regulations document.

Requirements:
1. Keep the original intention of the question.
2. Replace slang and local dialect with standard/administrative terminology.
3. Only return the rewritten question, do not explain further.
4. If the question is not related to the academic/student regulations, return the original question.

Example:
- Input: "Nợ môn có bị tống cổ ra khỏi ktx ko"
- Output: "Sinh viên nợ môn hoặc kết quả học tập kém có bị chấm dứt hợp đồng ký túc xá không?"

Question: "{query}"
Standardized question:"""
    
    # Gọi LLM để sinh câu hỏi mới 
    response = llm.complete(prompt)
    return str(response).strip()