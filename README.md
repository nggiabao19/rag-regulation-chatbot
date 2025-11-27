# Advanced RAG: Regulation Assistant Chatbot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-v0.10-purple)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

Hệ thống **Retrieval-Augmented Generation (RAG)** nâng cao, hỗ trợ tra cứu quy định/pháp lý với độ chính xác cao nhờ kỹ thuật **Hybrid Search** và **Re-ranking**.

## Tính năng nổi bật (Key Features)

* **Hybrid Search (BM25 + Vector):** Kết hợp tìm kiếm từ khóa chính xác và tìm kiếm ngữ nghĩa để không bỏ sót thông tin.
* **Re-ranking (Cross-Encoder):** Sử dụng mô hình `BAAI/bge-reranker` để chấm điểm lại kết quả, đảm bảo đoạn văn phù hợp nhất nằm trên cùng.
* **Query Rewriting:** Tự động viết lại câu hỏi của người dùng (xử lý từ lóng, viết tắt) trước khi tìm kiếm.
* **Auto-Ingestion Pipeline:** Tự động phát hiện và cập nhật tài liệu mới dựa trên mã Hash (MD5), tránh trùng lặp dữ liệu.
* **Citations:** Trích dẫn chính xác tên file và số trang của nguồn thông tin.

## Kiến trúc hệ thống (Architecture)



## Cấu trúc dự án

```text
advanced_rag_project/
├── data/                  # Thư mục chứa tài liệu PDF gốc
├── src/
│   ├── ingestion.py       # Logic xử lý dữ liệu (Chunking, Hashing, Upsert)
│   ├── rag_pipeline.py    # Logic RAG (Retriever, Reranker, Query Engine)
│   ├── database.py        # Kết nối Pinecone & Global Settings
│   └── config.py          # Quản lý cấu hình tập trung
├── app.py                 # Giao diện Streamlit
├── ingest_script.py       # Script chạy nạp dữ liệu
└── requirements.txt       # Danh sách thư viện
```

## Cài đặt và Chạy

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/nggiabao19/rag-regulation-chatbot.git](https://github.com/nggiabao19/rag-regulation-chatbot.git)
    cd rag-regulation-chatbot
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình môi trường:**
    Tạo file `.env` và điền API Key:
    ```ini
    OPENAI_API_KEY="sk-..."
    PINECONE_API_KEY="pc-..."
    ```

4.  **Nạp dữ liệu (Ingest):**
    Bỏ file PDF vào thư mục `data/` và chạy:
    ```bash
    python3 ingest_script.py
    ```

5.  **Chạy ứng dụng:**
    ```bash
    streamlit run app.py
    ```

## Tech Stack

* **Framework:** LlamaIndex
* **Vector Database:** Pinecone (Serverless)
* **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
* **Reranker:** BAAI/bge-reranker-base
* **LLM:** GPT-4o-mini
* **Interface:** Streamlit

---
*Project thực hiện bởi Nguyễn Gia Bảo*