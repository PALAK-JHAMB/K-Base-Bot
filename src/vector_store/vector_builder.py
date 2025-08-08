# # import sys
# # import os

# # import yaml
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain_community.vectorstores import FAISS
# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# # # Add src to path to allow imports
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# # from src.ingestion.pdf_loader import load_and_process_pdfs

# # def build_vector_store():
# #     """Builds and saves the vector store from PDF documents using advanced parsing."""
# #     # 1. Load configuration
# #     # with open("config/settings.yaml", 'r') as f:
# #     #     config = yaml.safe_load(f)
# #     settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
# #     with open(settings_path, 'r') as f:
# #         config = yaml.safe_load(f)
# #     api_key = config['gemini']['api_key']
# #     # pdf_path = config['data']['pdf_path']
# #     # vector_store_path = config['data']['vector_store_path']
# #     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
# #     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
# #     # 2. Load PDF documents using our new, powerful loader
# #     print("Loading and processing PDF documents with 'unstructured'...")
# #     documents = load_and_process_pdfs(pdf_path, config)
# #     if not documents:
# #         print("No documents were loaded. Check your PDF path and file content. Exiting.")
# #         return

# #     # 3. Chunk the documents
# #     # print(f"Chunking {len(documents)} processed documents...")
# #     # # These documents are already logically grouped, but we still chunk them to fit context windows
# #     # text_splitter = RecursiveCharacterTextSplitter(
# #     #     chunk_size=1500, 
# #     #     chunk_overlap=200,
# #     #     separators=["\n\n", "\n", " ", ""]
# #     # )
# #     # docs = text_splitter.split_documents(documents)
# #     # print(f"Created {len(docs)} chunks for embedding.")
    
# #     print(f"Chunking {len(documents)} processed documents...")
    
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         # Increase chunk size to ensure full procedures are captured
# #         chunk_size=2000, 
        
# #         # A slightly larger overlap is good practice with larger chunks
# #         chunk_overlap=300,
        
# #         # These are the characters it will try to split on, in order.
# #         # \n\n is the most important for keeping paragraphs/lists together.
# #         separators=["\n\n", "\n", ". ", " ", ""],
        
# #         # This is an important parameter to keep metadata
# #         keep_separator=False
# #     )
# #     docs = text_splitter.split_documents(documents)
# #     print(f"Created {len(docs)} chunks for embedding.")
# #     # ... rest of the function ...
# #     # 4. Create embeddings
# #     print("Creating embeddings with Gemini...")
# #     embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)

# #     # 5. Build and save the vector store (FAISS)
# #     print("Building and saving FAISS vector store...")
# #     vector_store = FAISS.from_documents(docs, embeddings)
# #     vector_store.save_local(vector_store_path)
# #     print(f"Vector store created successfully at {vector_store_path}")

# # if __name__ == '__main__':
# #     build_vector_store()


# # src/vector_store/vector_builder.py

# # src/vector_store/vector_builder.py

# import sys
# import os
# import yaml
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import streamlit as st # Import streamlit to access secrets

# # --- DEFINE PROJECT ROOT for reliable file paths ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# # --- Add src to path to allow imports ---
# sys.path.append(PROJECT_ROOT)
# from src.ingestion.pdf_loader import load_and_process_pdfs

# def build_vector_store():
#     """Builds and saves the vector store using absolute paths and hybrid config."""
    
#     # --- 1. Load Config (Hybrid Approach for Deployment) ---
#     config = {}
#     settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
#     try:
#         with open(settings_path, 'r') as f:
#             config = yaml.safe_load(f)
#         print("Builder: Loaded config from local 'settings.yaml' file.")
#         if "API_KEY" in st.secrets:
#             config['gemini']['api_key'] = st.secrets["API_KEY"]
#     except FileNotFoundError:
#         print("Builder: 'settings.yaml' not found. Loading config from Streamlit secrets.")
#         if "API_KEY" in st.secrets:
#             config = {
#                 "gemini": {
#                     "api_key": st.secrets["API_KEY"],
#                     "embedding_model": "models/embedding-001"
#                 },
#                 "data": {
#                     "pdf_path": "data/pdf",
#                     "vector_store_path": "vector_store/faiss_index"
#                 },
#                 "ingestion": {
#                     "parsing_strategy": "fast"
#                 }
#             }
#         else:
#             # In a non-UI script, we raise an error to stop execution
#             raise ValueError("API Key not found in Streamlit secrets. Build cannot proceed.")

#     api_key = config['gemini']['api_key']
    
#     # --- Use absolute paths for data and vector store ---
#     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

#     # 2. Load PDF documents
#     print("Builder: Loading and processing PDF documents...")
#     documents = load_and_process_pdfs(pdf_path, config)
#     if not documents:
#         print("Builder: No documents were loaded. Exiting.")
#         return

#     # 3. Chunk the documents
#     print(f"Builder: Chunking {len(documents)} processed documents...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000, 
#         chunk_overlap=300,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     docs = text_splitter.split_documents(documents)
#     print(f"Builder: Created {len(docs)} chunks for embedding.")

#     # 4. Create embeddings
#     print("Builder: Creating embeddings with Gemini...")
#     embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)

#     # 5. Build and save the vector store (FAISS)
#     print("Builder: Building and saving FAISS vector store...")
#     vector_store = FAISS.from_documents(docs, embeddings)
#     vector_store.save_local(vector_store_path)
#     print(f"Builder: Vector store created successfully at {vector_store_path}")

# if __name__ == '__main__':
#     build_vector_store()

# src/vector_store/vector_builder.py

import sys
import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# --- DEFINE PROJECT ROOT for reliable file paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(PROJECT_ROOT)
from src.ingestion.pdf_loader import load_and_process_pdfs


def load_or_build_vector_store():
    """
    Checks if the vector store exists. If so, loads it. 
    If not, builds it, saves it, and returns the store object.
    This is a robust function for both local and cloud environments.
    """
    # --- 1. Load Config (Hybrid Approach) ---
    # This logic is now self-contained here
    config = {}
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    try:
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        if "API_KEY" in st.secrets:
            config['gemini']['api_key'] = st.secrets["API_KEY"]
    except FileNotFoundError:
        if "API_KEY" in st.secrets:
            config = {
                "gemini": {"api_key": st.secrets["API_KEY"], "embedding_model": "models/embedding-001"},
                "data": {"pdf_path": "data/pdf", "vector_store_path": "vector_store/faiss_index"},
                "ingestion": {"parsing_strategy": "fast"}
            }
        else:
            raise ValueError("API Key not found in Streamlit secrets.")

    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    api_key = config['gemini']['api_key']
    
    # --- 2. Check if store exists, and load it ---
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

    # --- 3. If it doesn't exist, build it ---
    else:
        st.info("Vector store not found. Building it now. This may take a few minutes...")
        print("Vector store not found. Triggering build process...")
        
        pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
        documents = load_and_process_pdfs(pdf_path, config)
        if not documents:
            st.error("No documents were loaded to build the vector store. Check the data/pdf path.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)
        
        print("Building and saving FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"Vector store built and saved successfully at {vector_store_path}")
        return vector_store