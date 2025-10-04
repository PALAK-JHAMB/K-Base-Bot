import streamlit as st
import yaml, sys, os
from thefuzz import process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
from src.vector_store.vector_builder import build_vector_store
from src.vector_store.retriever import get_retriever

st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    print("\n--- INITIATING RESOURCE LOADING ---")
    config = {}
    try:
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded base config from 'settings.yaml'.")
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Using hardcoded defaults.")
        config = {
            "data": {
                "pdf_path": "data/pdf",
                "excel_path": "data/excelfile.xlsx",
                "vector_store_path": "vector_store/faiss_index"
            },
            "ingestion": {"parsing_strategy": "fast", "process_images": False}
        }

    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Knowledge base not found. Building it now...")
        build_vector_store(config)
    
    faq_data, retriever, rag_chain = None, None, None
    
    try:
        excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
        faq_data = parse_excel_qa(excel_path)
        print(f"FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"FAQ Data Loaded: FAILED with an exception: {e}")

    try:
        retriever = get_retriever(config)
        print(f"Retriever Loaded: {'SUCCESS' if retriever is not None else 'FAILED'}")
    except Exception as e:
        print(f"Retriever Loaded: FAILED with an exception: {e}")

    try:
        rag_chain = get_rag_chain(retriever, config)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
    if faq_data is None or retriever is None or rag_chain is None:
        st.error("Failed to load one or more resources...")
        st.stop()
        
    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain

faq_data, retriever, rag_chain = load_all_resources()

# ...
def get_faq_answer(query: str, faqs: list[dict]) -> str or None:
    if not faqs: return None
    # ... (rest of function is fine)
    return None

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            faq_answer = get_faq_answer(prompt, faq_data)
            
            if faq_answer:
                response = f"**From FAQ:**\n\n{faq_answer}"
            else:
                st.info("No FAQ match found. Searching documents...")
                response = rag_chain.invoke(prompt)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})