# src/ui/app.py

import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup (CRITICAL FOR MODULAR IMPORTS) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Now that the path is set, we can do our backend imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.bot_engine.gemini_responder import get_rag_chain
# --- FIX 1: Import the new, correct function name ---
from src.vector_store.vector_builder import load_or_build_vector_store
from src.vector_store.retriever import get_retriever

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    """
    Loads all necessary resources using the robust load_or_build_vector_store function.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load or Build the Vector Store ---
    # This single function now handles everything related to the vector store.
    # --- FIX 2: Call the new, correct function name ---
    vector_store = load_or_build_vector_store()
    if vector_store is None:
        st.error("Failed to load or build the vector store. App cannot continue.")
        st.stop()
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    print("Retriever created successfully.")

    # --- 2. Load the rest of the resources ---
    # We still need to load the config for the excel path
    config = {}
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    try:
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # This is a simplified fallback just for the excel path
        config = {"data": {"excel_path": "data/excelfile.xlsx"}}

    faq_data = None
    rag_chain = None

    try:
        excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
        faq_data = parse_excel_qa(excel_path)
        print(f"FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"FAQ Data Loaded: FAILED with an exception: {e}")

    try:
        rag_chain = get_rag_chain(retriever)
        print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
    except Exception as e:
        print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
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