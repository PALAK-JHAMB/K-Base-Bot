# src/ui/app.py

import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup (CRITICAL FOR MODULAR IMPORTS) ---
# This MUST be the very first thing the script does to ensure
# that the 'src' module can be found by the interpreter.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Now that the path is set, we can do our backend imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
from src.vector_store.vector_builder import build_vector_store
from src.vector_store.retriever import get_retriever # We still need this

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

# @st.cache_resource
# def load_all_resources():
#     """
#     Loads all necessary resources using absolute paths for deployment compatibility.
#     Builds the vector store on the first boot if it doesn't exist.
#     """
#     print("\n--- INITIATING RESOURCE LOADING ---")

#     # --- Load Config (Hybrid Approach) ---
#     config = {}
#     settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
#     try:
#         with open(settings_path, 'r') as f:
#             config = yaml.safe_load(f)
#         print("1. Loaded config from local 'settings.yaml' file.")
#         if "API_KEY" in st.secrets:
#             config['gemini']['api_key'] = st.secrets["API_KEY"]
#     except FileNotFoundError:
#         print("1. 'settings.yaml' not found. Loading config from Streamlit secrets.")
#         if "API_KEY" in st.secrets:
#             config = {
#                 "gemini": {
#                     "api_key": st.secrets["API_KEY"],
#                     "embedding_model": "models/embedding-001",
#                     "llm_model": "models/gemini-1.5-flash-latest"
#                 },
#                 "data": {
#                     "pdf_path": "data/pdf", # Corrected path
#                     "excel_path": "data/excelfile.xlsx",
#                     "vector_store_path": "vector_store/faiss_index"
#                 },
#                 "ingestion": {
#                     "parsing_strategy": "fast" # Use 'fast' for deployment
#                 }
#             }
#         else:
#             st.error("API Key not found in Streamlit secrets. Please add it to your app's secrets.")
#             st.stop()

#     # --- Build Vector Store if it doesn't exist (using absolute path) ---
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
#     if not os.path.exists(vector_store_path):
#         st.info("Vector store not found. Building it now. This may take a few minutes...")
#         print("Vector store not found. Triggering build process...")
#         build_vector_store()
#         print("Vector store built successfully.")
    
#     # --- Load all resources ---
#     faq_data, retriever, rag_chain = None, None, None
    
#     try:
#         excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
#         faq_data = parse_excel_qa(excel_path)
#         print(f"2. FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
#     except Exception as e:
#         print(f"2. FAQ Data Loaded: FAILED with an exception: {e}")

#     try:
#         retriever = get_retriever()
#         print(f"3. Retriever Loaded: {'SUCCESS' if retriever is not None else 'FAILED'}")
#     except Exception as e:
#         print(f"3. Retriever Loaded: FAILED with an exception: {e}")

#     try:
#         rag_chain = get_rag_chain(retriever)
#         print(f"4. RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
#     except Exception as e:
#         print(f"4. RAG Chain Loaded: FAILED with an exception: {e}")
    
#     # --- Final Check ---
#     if faq_data is None or retriever is None or rag_chain is None:
#         st.error("Failed to load one or more resources. Please check terminal logs for details.")
#         st.stop()
        
#     print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
#     return faq_data, retriever, rag_chain

# In src/ui/app.py

@st.cache_resource
def load_all_resources():
    """
    Loads all necessary resources and passes the config to the builder function.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- Load Config (Hybrid Approach) ---
    config = {}
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    try:
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded config from local 'settings.yaml' file.")
        if "API_KEY" in st.secrets:
            config['gemini']['api_key'] = st.secrets["API_KEY"]
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Loading config from Streamlit secrets.")
        if "API_KEY" in st.secrets:
            config = {
                "gemini": {
                    "api_key": st.secrets["API_KEY"],
                    "embedding_model": "models/embedding-001",
                    "llm_model": "models/gemini-1.5-flash-latest"
                },
                "data": {
                    "pdf_path": "data/pdf",
                    "excel_path": "data/excelfile.xlsx",
                    "vector_store_path": "vector_store/faiss_index"
                },
                "ingestion": {
                    "parsing_strategy": "fast"
                }
            }
        else:
            st.error("API Key not found in Streamlit secrets.")
            st.stop()

    # --- Build Vector Store if it doesn't exist ---
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Vector store not found. Building it now. This may take a few minutes...")
        # --- PASS THE CONFIG DICTIONARY TO THE BUILDER ---
        build_vector_store(config)
    
    # --- Load all resources ---
    faq_data, retriever, rag_chain = None, None, None
    
    try:
        excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
        faq_data = parse_excel_qa(excel_path)
        print(f"2. FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"2. FAQ Data Loaded: FAILED with an exception: {e}")

    try:
        retriever = get_retriever()
        print(f"3. Retriever Loaded: {'SUCCESS' if retriever is not None else 'FAILED'}")
    except Exception as e:
        print(f"3. Retriever Loaded: FAILED with an exception: {e}")

    try:
        rag_chain = get_rag_chain(retriever)
        print(f"4. RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"4. RAG Chain Loaded: FAILED with an exception: {e}")
    
    # --- Final Check ---
    if faq_data is None or retriever is None or rag_chain is None:
        st.error("Failed to load one or more resources. Please check terminal logs for details.")
        st.stop()
        
    print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
    return faq_data, retriever, rag_chain
# --- Load all resources and assign them to variables ---
faq_data, retriever, rag_chain = load_all_resources()

# --- Chat Logic ---
def get_faq_answer(query: str, faqs: list[dict]) -> str or None:
    if not faqs: return None
    faq_questions = [item['user_desc'] for item in faqs]
    best_match = process.extractOne(query, faq_questions, score_cutoff=90)
    
    if best_match:
        best_matching_question_text = best_match[0]
        for item in faqs:
            if item['user_desc'] == best_matching_question_text:
                print(f"FAQ Match Found: '{query}' -> '{best_matching_question_text}' (Score: {best_match[1]})")
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
            faq_answer = get_faq_answer(prompt, faq_data)
            
            if faq_answer:
                response = f"**From FAQ:**\n\n{faq_answer}"
            else:
                st.info("No FAQ match found. Searching documents...")
                response = rag_chain.invoke(prompt)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})