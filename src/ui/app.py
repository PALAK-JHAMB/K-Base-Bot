
import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Backend Imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
from src.vector_store.vector_builder import build_vector_store
from langchain_huggingface import HuggingFaceEmbeddings   # ✅ modern import
from langchain_community.vectorstores import FAISS        # still in community


# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")


@st.cache_resource
def load_all_resources():
    """Loads configs, vector store, retriever, and RAG chain."""
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load Config ---
    config = {}
    try:
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded base config from 'settings.yaml'.")
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Using defaults...")
        config = {
            "data": {
                "pdf_path": "data/pdf",
                "excel_path": "data/excelfile.xlsx",
                "vector_store_path": "vector_store/faiss_index"
            },
            "ingestion": {"parsing_strategy": "fast", "process_images": False}
        }

    # --- 2. Build Vector Store if missing ---
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Knowledge base not found. Building it now...")
        build_vector_store(config)

    # --- 3. Load FAISS Retriever ---
    retriever = None
    try:
        print("Loading vector store and creating retriever...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        print("Retriever Loaded: SUCCESS")
    except Exception as e:
        st.error(f"Retriever Loaded: FAILED → {e}")
        print(f"Retriever Loaded: FAILED with exception: {e}")

    # --- 4. Load RAG Chain ---
    faq_data, rag_chain = None, None
    try:
        rag_chain = get_rag_chain(retriever, config)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain else 'FAILED'}")
    except Exception as e:
        st.error(f"RAG Chain Loaded: FAILED → {e}")
        print(f"RAG Chain Loaded: FAILED with exception: {e}")

    # --- Final Check ---
    if retriever is None or rag_chain is None:
        st.error("❌ Failed to load the RAG pipeline. Please check the logs above.")
        st.stop()

    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain


# --- Load resources ---
faq_data, retriever, rag_chain = load_all_resources()


# --- FAQ Answer Helper ---
def get_faq_answer(query: str, faqs: list[dict]) -> str | None:
    if not faqs:
        return None
    faq_questions = [item['user_desc'] for item in faqs]
    best_match = process.extractOne(query, faq_questions, score_cutoff=90)
    if best_match:
        for item in faqs:
            if item['user_desc'] == best_match[0]:
                return item['user_reply_desc']
    return None


# --- Chat Session Handling ---
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
