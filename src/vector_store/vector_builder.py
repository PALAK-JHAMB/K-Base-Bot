# src/vector_store/vector_builder.py

import sys
import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # <-- IMPORTANT: Switched to Hugging Face

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_and_process_pdfs

def build_vector_store(config: dict):
    """
    Builds and saves the vector store using a Hugging Face embedding model.
    This avoids all Google API embedding quotas.
    """
    print("Builder: Starting the vector store build process...")
    
    # --- 1. Get paths from the passed-in config dictionary ---
    pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

    # 2. Load PDF documents
    print(f"Builder: Loading PDFs from {pdf_path}...")
    documents = load_and_process_pdfs(pdf_path, config)
    if not documents:
        print("Builder: No documents were loaded. Exiting.")
        return

    # 3. Chunk the documents
    print(f"Builder: Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=300
    )
    docs = text_splitter.split_documents(documents)
    print(f"Builder: Created {len(docs)} chunks.")

    # 4. Create embeddings using Hugging Face (NO API KEY NEEDED)
    print("Builder: Initializing Hugging Face embedding model (this may download on first run)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("Builder: Hugging Face model initialized.")

    # 5. Build and save the vector store (FAISS)
    print("Builder: Building FAISS vector store from documents...")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    print("Builder: Saving FAISS index to disk...")
    vector_store.save_local(vector_store_path)
    print(f"Builder: Vector store created successfully at {vector_store_path}")

# This block allows you to still run this script directly from the command line for local building
if __name__ == '__main__':
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
        main_config = yaml.safe_load(f)
    build_vector_store(main_config)