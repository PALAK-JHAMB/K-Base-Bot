# # src/vector_store/vector_builder.py

# import sys
# import os
# import yaml
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS

# # --- System Path Setup ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(PROJECT_ROOT)

# from src.ingestion.pdf_loader import load_and_process_pdfs

# def build_vector_store(config: dict):
#     """
#     Builds and saves the vector store. This is a pure backend script.
#     """
#     print("Builder: Starting the vector store build process...")
    
#     api_key = config['gemini']['api_key']
#     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

#     documents = load_and_process_pdfs(pdf_path, config)
#     if not documents:
#         print("Builder: No documents were loaded. Exiting.")
#         return

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#     docs = text_splitter.split_documents(documents)
#     print(f"Builder: Created {len(docs)} chunks.")

#     embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)

#     vector_store = FAISS.from_documents(docs, embeddings)
#     vector_store.save_local(vector_store_path)
#     print(f"Builder: Vector store created successfully at {vector_store_path}")

# if __name__ == '__main__':
#     with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
#         main_config = yaml.safe_load(f)
#     build_vector_store(main_config)
# src/vector_store/vector_builder.py

import sys
import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_and_process_pdfs

def get_or_create_vector_store(config: dict):
    """
    Checks if the vector store exists. If so, loads it.
    If not, builds it, saves it, and returns the store object directly from memory.
    """
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    api_key = config['gemini']['api_key']
    
    # --- 1. Check if store exists, and load it ---
    if os.path.exists(vector_store_path):
        print("Vector store found. Loading from disk...")
        embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")
        return vector_store

    # --- 2. If it doesn't exist, build it ---
    else:
        st.info("Knowledge base not found. Building it now. This may take a few minutes...")
        print("Knowledge base not found. Triggering build process...")
        
        pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
        documents = load_and_process_pdfs(pdf_path, config)
        if not documents:
            st.error("No documents were loaded to build the knowledge base.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        
        print("Building and saving FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"Knowledge base built and saved successfully at {vector_store_path}")
        # Return the newly created object directly from memory
        return vector_store