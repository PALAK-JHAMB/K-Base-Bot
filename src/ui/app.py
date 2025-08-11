
# # src/ui/app.py

# import streamlit as st
# import yaml
# import sys
# import os
# from thefuzz import process

# # --- System Path Setup ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(PROJECT_ROOT)

# # --- Backend Imports ---
# from src.ingestion.excel_parser import parse_excel_qa
# from src.bot_engine.gemini_responder import get_rag_chain
# # We now only need this one function for the vector store
# from src.vector_store.vector_builder import get_or_create_vector_store

# # --- Page Configuration ---
# st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
# st.title("IRCTC Chatbot: Ask all your queries")
# st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
# st.write("Ask a question about your documents, or check our FAQs!")

# @st.cache_resource
# def load_all_resources():
#     """
#     Loads all necessary resources using the robust get_or_create_vector_store function.
#     """
#     print("\n--- INITIATING RESOURCE LOADING ---")

#     # --- 1. Load Config (Hybrid Approach) ---
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
#                     "pdf_path": "data/pdf",
#                     "excel_path": "data/excelfile.xlsx",
#                     "vector_store_path": "vector_store/faiss_index"
#                 },
#                 "ingestion": {
#                     "parsing_strategy": "fast"
#                 }
#             }
#         else:
#             st.error("API Key not found in Streamlit secrets.")
#             st.stop()

#     # --- 2. Load or Build the Vector Store and Create Retriever ---
#     vector_store = get_or_create_vector_store(config)
#     if vector_store is None:
#         st.error("Failed to load or build the vector store. App cannot continue.")
#         st.stop()
    
#     retriever = vector_store.as_retriever(search_kwargs={"k": 7})
#     print("Retriever created successfully.")

#     # --- 3. Load other resources ---
#     faq_data = None
#     rag_chain = None

#     try:
#         excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
#         faq_data = parse_excel_qa(excel_path)
#         print(f"FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
#     except Exception as e:
#         print(f"FAQ Data Loaded: FAILED with an exception: {e}")

#     try:
#         rag_chain = get_rag_chain(retriever)
#         print(f"RAG Chain Loaded: {'SUCCESS' if rag_chain is not None else 'FAILED'}")
#     except Exception as e:
#         print(f"RAG Chain Loaded: FAILED with an exception: {e}")
    
#     # --- Final Check ---
#     if faq_data is None or retriever is None or rag_chain is None:
#         st.error("Failed to load one or more resources. Please check terminal logs for details.")
#         st.stop()
        
#     print("--- ALL RESOURCES LOADED SUCCESSFULLY ---\n")
#     return faq_data, retriever, rag_chain

# # --- Load all resources and assign them to variables ---
# faq_data, retriever, rag_chain = load_all_resources()

# # --- [The rest of your app.py (Chat Logic, UI State, Main Interaction) is correct and can remain the same] ---
# def get_faq_answer(query: str, faqs: list[dict]) -> str or None:
#     if not faqs: return None
#     faq_questions = [item['user_desc'] for item in faqs]
#     best_match = process.extractOne(query, faq_questions, score_cutoff=90)
    
#     if best_match:
#         best_matching_question_text = best_match[0]
#         for item in faqs:
#             if item['user_desc'] == best_matching_question_text:
#                 print(f"FAQ Match Found: '{query}' -> '{best_matching_question_text}' (Score: {best_match[1]})")
#                 return item['user_reply_desc']
#     return None

# if 'messages' not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask your question..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             faq_answer = get_faq_answer(prompt, faq_data)
            
#             if faq_answer:
#                 response = f"**From FAQ:**\n\n{faq_answer}"
#             else:
#                 st.info("No FAQ match found. Searching documents...")
#                 response = rag_chain.invoke(prompt)
#                 response = response.replace("<br><br>", "\n\n")
#             # Replace any single <br> with a single newline
#                 response = response.replace("<br>", "\n")
            
#             st.markdown(response)
            
#     st.session_state.messages.append({"role": "assistant", "content": response})

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
from src.vector_store.retriever import get_retriever

# --- Page Configuration ---
st.set_page_config(page_title="Document & FAQ Chatbot", layout="wide")
st.title("IRCTC Chatbot: Ask all your queries")
st.subheader("CENTER FOR RAILWAY INFORMATION SYSTEMS")
st.write("Ask a question about your documents, or check our FAQs!")

@st.cache_resource
def load_all_resources():
    """
    Loads all necessary resources, handling config, secrets, and building/loading the vector store.
    This function is the single source of truth for configuration.
    """
    print("\n--- INITIATING RESOURCE LOADING ---")

    # --- 1. Load Config (Secrets-First Approach) ---
    config = {}
    # First, try to load from local YAML for non-secret defaults
    try:
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        print("1. Loaded base config from 'settings.yaml'.")
    except FileNotFoundError:
        print("1. 'settings.yaml' not found. Using hardcoded defaults for deployment.")
        # Define default config if YAML is missing (for cloud environment)
        config = {
            "gemini": {
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

    # Now, overwrite the API key with the one from secrets (local or cloud)
    # This is the "secrets-first" principle.
    if "GEMINI_API_KEY" in st.secrets:
        # Ensure the nested dictionary exists
        if 'gemini' not in config:
            config['gemini'] = {}
        config['gemini']['api_key'] = st.secrets["GEMINI_API_KEY"]
        print("   Successfully loaded API key from Streamlit secrets.")
    else:
        # If running locally without a secrets.toml, check the yaml
        if not config.get('gemini', {}).get('api_key'):
             st.error("GEMINI_API_KEY not found in Streamlit secrets or settings.yaml! Please add it.")
             st.stop()

    # --- 2. Build Vector Store if it doesn't exist ---
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    if not os.path.exists(vector_store_path):
        st.info("Knowledge base not found. Building it now. This may take a few minutes...")
        # Pass the fully resolved config to the builder
        build_vector_store(config)
    
    # --- 3. Load all resources using the final config ---
    faq_data, retriever, rag_chain = None, None, None
    
    try:
        excel_path = os.path.join(PROJECT_ROOT, config['data']['excel_path'])
        faq_data = parse_excel_qa(excel_path)
        print(f"FAQ Data Loaded: {'SUCCESS' if faq_data is not None else 'FAILED'}")
    except Exception as e:
        print(f"FAQ Data Loaded: FAILED with an exception: {e}")

    try:
        # Pass the config to the retriever
        retriever = get_retriever(config)
        print(f"Retriever Loaded: {'SUCCESS' if retriever is not None else 'FAILED'}")
    except Exception as e:
        print(f"Retriever Loaded: FAILED with an exception: {e}")

    try:
        # Pass the retriever and config to the chain
        rag_chain = get_rag_chain(retriever, config)
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
                # The final chain from our corrected responder expects a string
                response = rag_chain.invoke(prompt)
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})