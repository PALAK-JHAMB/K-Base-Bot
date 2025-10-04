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
# --- THIS IS THE CORRECTED IMPORT PATH YOU FOUND ---
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    """
    Loads all resources, using the Hugging Face Inference API for embeddings.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load Config (Secrets-First) ---
    config = {}
    try:
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded base config from 'settings.yaml'.")
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Using hardcoded defaults.")
        config = {
            "huggingface": {},
            "data": {
                "pdf_path": "data/pdf",
                "excel_path": "data/excelfile.xlsx",
                "vector_store_path": "vector_store/faiss_index"
            },
            "ingestion": {"parsing_strategy": "fast", "process_images": False}
        }
    
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        if 'huggingface' not in config: config['huggingface'] = {}
        config['huggingface']['api_key'] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        print("   Successfully loaded Hugging Face API token from secrets.")
    else:
        st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")
        st.stop()

    # --- 2. Build Vector Store if it doesn't exist ---
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Knowledge base not found. Building it now...")
        build_vector_store(config)
    
    # --- 3. Load the Vector Store and Create the Retriever ---
    retriever = None
    try:
        print("Loading vector store and creating retriever...")
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=config['huggingface']['api_key'], 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        print(f"Retriever Loaded: SUCCESS")
    except Exception as e:
        print(f"Retriever Loaded: FAILED with an exception: {e}")

    # --- 4. Load other resources (FAQ DISABLED) ---
    faq_data = None
    rag_chain = None
    print(f"FAQ Data Loaded: SKIPPED (to conserve memory)")

    try:
        rag_chain = get_rag_chain(retriever, config)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
    if retriever is None or rag_chain is None:
        st.error("Failed to load the RAG pipeline. Please check the logs.")
        st.stop()
        
    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain

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