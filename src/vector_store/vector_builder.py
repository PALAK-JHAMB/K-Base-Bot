# src/vector_store/vector_builder.py

import sys
import os
import yaml
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- THIS IS THE CORRECTED IMPORT PATH YOU FOUND ---
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_and_process_pdfs

def build_vector_store(config: dict):
    """
    Builds and saves the vector store using the Hugging Face Inference API for embeddings.
    """
    print("Builder: Starting vector store build with Hugging Face API embeddings...")
    
    pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    api_key = config['huggingface']['api_key']

    documents = load_and_process_pdfs(pdf_path, config)
    if not documents:
        print("Builder: No documents were loaded. Exiting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    print(f"Builder: Created {len(docs)} chunks.")

    print("Builder: Initializing Hugging Face API embedding client...")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key, model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Builder: Hugging Face client initialized.")
    
    # --- MANUAL BATCHING TO AVOID TIMEOUTS AND API LIMITS ---
    batch_size = 15
    all_texts = [doc.page_content for doc in docs]
    all_metadatas = [doc.metadata for doc in docs]
    
    # Initialize FAISS with a dummy embedding to start
    vector_store = FAISS.from_texts(
        texts=[" "], embedding=embeddings, metadatas=[{"dummy": True}]
    )

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        print(f"  - Embedding batch {i//batch_size + 1}...")
        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        time.sleep(1) # Small delay to be safe with the free tier API

    vector_store.save_local(vector_store_path)
    print(f"Builder: Vector store created successfully at {vector_store_path}")

if __name__ == '__main__':
    import streamlit as st
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
        main_config = yaml.safe_load(f)
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        if 'huggingface' not in main_config: main_config['huggingface'] = {}
        main_config['huggingface']['api_key'] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    build_vector_store(main_config)