# src/ui/app.py

import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup (CRITICAL FOR MODULAR IMPORTS) ---
# This MUST be the very first thing the script does.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Now that the path is set, we can do our backend imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
from src.vector_store.vector_builder import build_vector_store
# We need these for loading the index directly, removing the need for retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    """
    Loads all necessary resources. Builds the vector store if needed, then loads it
    and creates the retriever directly. The FAQ feature is disabled to conserve memory.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load Config ---
    config = {}
    try:
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded base config from 'settings.yaml'.")
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Using hardcoded defaults for deployment.")
        config = {
            "data": {
                "pdf_path": "data/pdf",
                "excel_path": "data/excelfile.xlsx",
                "vector_store_path": "vector_store/faiss_index"
            },
            "ingestion": {"parsing_strategy": "fast", "process_images": False}
        }

    # --- 2. Build Vector Store if it doesn't exist ---
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Knowledge base not found. Building it now. This may take a few minutes...")
        build_vector_store(config)
    
    # --- 3. Load the Vector Store and Create the Retriever ---
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
        print(f"Retriever Loaded: SUCCESS")
    except Exception as e:
        print(f"Retriever Loaded: FAILED with an exception: {e}")

    # --- 4. Load other resources (FAQ DISABLED TO PREVENT MEMORY CRASH) ---
    faq_data = None  # Explicitly disable the FAQ feature
    rag_chain = None
    
    print(f"FAQ Data Loaded: SKIPPED (to conserve memory on Streamlit Cloud)")

    try:
        rag_chain = get_rag_chain(retriever, config)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
    # --- Final Check (Simplified for RAG-only) ---
    if retriever is None or rag_chain is None:
        st.error("Failed to load the RAG pipeline. Please check the logs for errors.")
        st.stop()
        
    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain


# --- Load all resources and assign them to variables ---
faq_data, retriever, rag_chain = load_all_resources()

# --- Chat Logic (This function will now always return None, which is correct) ---
def get_faq_answer(query: str, faqs: list[dict]) -> str or None:
    if not faqs:
        return None
    # The rest of the function is kept in case you want to re-enable it later
    # with a smaller Excel file.
    faq_questions = [item['user_desc'] for item in faqs]
    best_match = process.extractOne(query, faq_questions, score_cutoff=90)
    
    if best_match:
        best_matching_question_text = best_match[0]
        for item in faqs:
            if item['user_desc'] == best_matching_question_text:
                return item['user_reply_desc']
    return None

# --- UI State Management ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Interaction Logic ---
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The FAQ check will now always fail gracefully because faq_data is None
            faq_answer = get_faq_answer(prompt, faq_data)
            
            if faq_answer:
                response = f"**From FAQ:**\n\n{faq_answer}"
            else:
                st.info("No FAQ match found. Searching documents...")
                response = rag_chain.invoke(prompt)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})