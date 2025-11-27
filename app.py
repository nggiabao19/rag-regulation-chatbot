import streamlit as st
from src.rag_pipeline import get_rag_engine, rewrite_query 
from llama_index.llms.openai import OpenAI

st.set_page_config(page_title="Advanced RAG", layout="wide")

# Cache expensive resources
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_query_engine():
    return get_rag_engine()

@st.cache_resource(show_spinner=False)
def load_llm_rewriter():
    return OpenAI(model="gpt-4o-mini", temperature=0)

# Load cached resources
query_engine = load_query_engine()
llm_rewriter = load_llm_rewriter()

st.title("RAG: Chat với Tài liệu Quy chế học vụ ĐHCT")
st.markdown("Hệ thống sử dụng **Hybrid Search** (Vector + Keyword) và **Reranking** để tìm kiếm chính xác.")

# GIAO DIỆN
# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về tài liệu..."):
    # 1. Hiển thị câu hỏi gốc của user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý Rewrite Query
    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích câu hỏi..."):
            # Viết lại câu hỏi (để câu hỏi theo chuẩn văn phòng các tài liệu văn bản)
            rewritten_prompt = rewrite_query(prompt, llm_rewriter)
            
            # Hiển thị cho user thấy AI đã hiểu thế nào (rất tốt cho UX)
            st.info(f"**AI đã hiểu ý bạn là:** *{rewritten_prompt}*")
            
            # Tìm kiếm bằng câu hỏi ĐÃ ĐƯỢC CHUẨN HÓA
            response = query_engine.query(rewritten_prompt) 
            
            # Hiển thị kết quả
            st.markdown(response.response)
            
            # Citations 
            st.markdown("---")
            st.markdown("**Nguồn tham khảo:**")
            for i, node in enumerate(response.source_nodes):
                file_name = node.metadata.get('file_name', 'Unknown File')
                page_label = node.metadata.get('page_label', 'N/A')
                with st.expander(f"Nguồn {i+1}: {file_name} (Trang {page_label}) - Score: {node.score:.4f}"):
                    st.markdown(f"> {node.text}")

            st.session_state.messages.append({"role": "assistant", "content": response.response})