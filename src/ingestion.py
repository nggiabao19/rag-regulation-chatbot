import os
import json
import hashlib
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from src import config
from src.database import get_vector_store, init_settings

def calculate_file_hash(filepath):
    """Tạo mã MD5 hash cho nội dung file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_registry():
    if os.path.exists(config.REGISTRY_FILE):
        with open(config.REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(config.REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)

def run_ingestion():
    """Hàm chính để quét thư mục và nạp dữ liệu"""
    # 1. Khởi tạo Settings & DB
    init_settings()
    vector_store = get_vector_store()
    
    registry = load_registry()
    files_to_process = []
    
    # 2. Quét thư mục Data
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Data directory '{config.DATA_DIR}' not found.")
        return

    print(f">> Scanning directory: {config.DATA_DIR}")
    for filename in os.listdir(config.DATA_DIR):
        if filename.startswith("."): continue
        
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.isfile(filepath):
            current_hash = calculate_file_hash(filepath)
            
            # Logic kiểm tra thay đổi
            if filename not in registry:
                print(f"   [NEW] Found new file: {filename}")
                files_to_process.append(filepath)
                registry[filename] = current_hash
            elif registry[filename] != current_hash:
                print(f"   [MODIFIED] File changed: {filename}")
                files_to_process.append(filepath)
                registry[filename] = current_hash
    
    if not files_to_process:
        print("System is up-to-date. No new files to ingest.")
        return

    # 3. Thực hiện Ingestion
    print(f"\n>> Processing {len(files_to_process)} files...")
    documents = SimpleDirectoryReader(input_files=files_to_process).load_data()
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Kết nối index cũ để insert thêm
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    
    for doc in documents:
        print(f"   -> Indexing: {doc.metadata.get('file_name')}")
        index.insert(doc)

    # 4. Lưu trạng thái
    save_registry(registry)
    print("\nIngestion Complete! Registry updated.")