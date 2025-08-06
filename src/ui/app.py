# src/ui/app.py

import streamlit as st
import yaml
import sys
import os
from thefuzz import process

# --- System Path Setup ---
# This ensures that the app can find the other modules (ingestion, bot_engine, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Backend Imports ---
from src.ingestion.excel_parser import parse_excel_qa
from src.vector_store.retriever import get_retriever
from src.bot_engine.gemini_responder import get_rag_chain

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("🤖 Document & FAQ Chatbot")
st.write("Ask a question about your documents, or check our FAQs!")

# --- Load Resources (with caching) ---
@st.cache_resource
def load_all_resources():
    """
    Load all necessary resources once and cache them.
    This includes config, FAQ data, the retriever, and the RAG chain.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # Initialize all variables to None to prevent NameErrors
    faq_data, retriever, rag_chain = None, None, None

    # --- Load Config ---
    try:
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        print("1. Config 'settings.yaml' loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load config/settings.yaml. Error: {e}")
        st.error(f"Could not load config/settings.yaml. Error: {e}")
        st.stop()
        
    # --- Load FAQ Data ---
    try:
        faq_data = parse_excel_qa(config['data']['excel_path'])
        print(f"2. FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"2. FAQ Data Loaded: FAILED with an exception: {e}")

    # --- Load Retriever ---
    try:
        retriever = get_retriever()
        print(f"3. Retriever Loaded: {'SUCCESS' if retriever is not None else 'FAILED'}")
    except Exception as e:
        print(f"3. Retriever Loaded: FAILED with an exception: {e}")

    # --- Load RAG Chain ---
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
    """
    Finds the best matching FAQ answer using fuzzy string matching.
    """
    if not faqs: return None # Safety check
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
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # First, try to find an answer in the FAQ (Excel)
            faq_answer = get_faq_answer(prompt, faq_data)
            
            if faq_answer:
                response = f"**From FAQ:**\n\n{faq_answer}"
            else:
                # If not in FAQ, use the RAG chain
                st.info("No FAQ match found. Searching documents...")
                response = rag_chain.invoke(prompt)
            
            # Display the final response (from either FAQ or RAG)
            st.markdown(response)
            
    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})