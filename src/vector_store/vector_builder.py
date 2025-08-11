

# import sys
# import os
# import yaml
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS

# # --- System Path Setup ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(PROJECT_ROOT)

# # --- Now import from your src module ---
# from src.ingestion.pdf_loader import load_and_process_pdfs

# def get_or_create_vector_store(config: dict):
#     """
#     Checks if the vector store exists. If so, loads it.
#     If not, builds it, saves it, and returns the store object directly from memory.
#     This function is now completely decoupled from Streamlit.
#     """
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
#     api_key = config['gemini']['api_key']
    
#     # --- 1. Check if store exists, and load it ---
#     if os.path.exists(vector_store_path):
#         print("Vector store found. Loading from disk...")
#         embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
#         vector_store = FAISS.load_local(
#             vector_store_path, 
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#         print("Vector store loaded successfully.")
#         return vector_store

#     # --- 2. If it doesn't exist, build it ---
#     else:
#         # UI messages like st.info() are now handled by the calling script (app.py)
#         print("Knowledge base not found. Triggering build process...")
        
#         pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
#         documents = load_and_process_pdfs(pdf_path, config)
#         if not documents:
#             # Error messages are now simple prints; app.py will show the st.error()
#             print("ERROR: No documents were loaded to build the knowledge base.")
#             return None

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#         docs = text_splitter.split_documents(documents)
        
#         embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        
#         print("Building and saving FAISS vector store...")
#         vector_store = FAISS.from_documents(docs, embeddings)
#         vector_store.save_local(vector_store_path)
#         print(f"Knowledge base built and saved successfully at {vector_store_path}")
#         # Return the newly created object directly from memory
#         return vector_store

# # This block allows you to still run this script directly from the command line for local building
# if __name__ == '__main__':
#     # When run directly, it loads its own config from the standard path
#     with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
#         main_config = yaml.safe_load(f)
#     get_or_create_vector_store(main_config)


# src/vector_store/vector_builder.py

import sys
import os
import yaml
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_and_process_pdfs

def build_vector_store(config: dict):
    """
    Builds and saves the vector store with manual batching and delays
    to respect API rate limits during the embedding process.
    """
    print("Builder: Starting the vector store build process...")
    
    api_key = config['gemini']['api_key']
    pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

    documents = load_and_process_pdfs(pdf_path, config)
    if not documents:
        print("Builder: No documents were loaded. Exiting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    print(f"Builder: Created {len(docs)} chunks.")

    print("Builder: Creating embeddings with Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)

    # --- MANUAL BATCHING AND EMBEDDING WITH DELAY ---
    print("Builder: Starting embedding process in batches to respect rate limits...")
    batch_size = 10  # Process 10 documents per API call
    all_embeddings = []
    all_texts = [doc.page_content for doc in docs]
    all_metadatas = [doc.metadata for doc in docs]

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        print(f"  - Embedding batch {i//batch_size + 1}/{(len(all_texts) - 1)//batch_size + 1}...")
        
        try:
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print("  - Batch complete. Waiting for 2 seconds...")
            time.sleep(2) # Wait for 2 seconds between each batch
        except Exception as e:
            print(f"  - ERROR embedding batch: {e}")
            print("  - Skipping this batch and continuing...")
            # Add placeholder embeddings for the failed batch to avoid length mismatch
            all_embeddings.extend([None] * len(batch_texts))


    # Filter out any failed embeddings before creating the index
    valid_embeddings = []
    valid_texts = []
    valid_metadatas = []
    for i, emb in enumerate(all_embeddings):
        if emb is not None:
            valid_embeddings.append(emb)
            valid_texts.append(all_texts[i])
            valid_metadatas.append(all_metadatas[i])

    print(f"Builder: Successfully embedded {len(valid_embeddings)} out of {len(all_texts)} documents.")
    
    text_embedding_pairs = list(zip(valid_texts, valid_embeddings))

    # --- Build FAISS from pre-computed embeddings ---
    print("Builder: Building FAISS index from embeddings...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=valid_metadatas
    )
    
    print("Builder: Saving FAISS index to disk...")
    vector_store.save_local(vector_store_path)
    print(f"Builder: Vector store created successfully at {vector_store_path}")

# This block allows you to still run this script directly from the command line
if __name__ == '__main__':
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
        main_config = yaml.safe_load(f)
    build_vector_store(main_config)