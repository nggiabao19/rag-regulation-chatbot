import sys
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from src import config

def init_settings():
    """Khởi tạo các model và settings global cho LlamaIndex"""
    # 1. Setup Embedding Model
    print(f">> Loading Embedding Model: {config.EMBED_MODEL_ID}...")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_ID)
    Settings.embed_model = embed_model

    # 2. Setup Chunking
    text_splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    Settings.text_splitter = text_splitter

def get_vector_store():
    """Kết nối và trả về PineconeVectorStore"""
    # Khởi tạo client Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Kiểm tra Index, nếu chưa có thì tạo 
    if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f">> Creating new Pinecone Index: {config.PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=384, # Dimension của all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store

def get_index():
    """Trả về VectorStoreIndex đã kết nối với Database"""
    # Đảm bảo Settings đã được load
    init_settings()
    
    vector_store = get_vector_store()
    
    # Kết nối Storage Context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load Index từ Vector Store (không tạo mới data)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index