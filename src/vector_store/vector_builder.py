

# # # src/vector_store/vector_builder.py
# # import sys, os, yaml
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import FAISS
# # from langchain_huggingface import HuggingFaceEmbeddings

# # PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# # sys.path.append(PROJECT_ROOT)
# # from src.ingestion.pdf_loader import load_and_process_pdfs

# # def build_vector_store(config: dict):
# #     print("Builder: Starting vector store build with Hugging Face embeddings...")
# #     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
# #     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
# #     documents = load_and_process_pdfs(pdf_path, config)
# #     if not documents: return

# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
# #     docs = text_splitter.split_documents(documents)
# #     print(f"Builder: Created {len(docs)} chunks.")

# #     print("Builder: Initializing Hugging Face embedding model...")
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     print("Builder: Hugging Face model initialized.")
    
# #     vector_store = FAISS.from_documents(docs, embeddings)
# #     vector_store.save_local(vector_store_path)
# #     print(f"Builder: Vector store created successfully at {vector_store_path}")

# # if __name__ == '__main__':
# #     with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
# #         main_config = yaml.safe_load(f)
# #     build_vector_store(main_config)
# # src/vector_store/vector_builder.py

# import sys
# import os
# import yaml
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# # We no longer need to import HuggingFaceEmbeddings here

# # --- System Path Setup ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(PROJECT_ROOT)

# from src.ingestion.pdf_loader import load_and_process_pdfs

# def build_vector_store(config: dict, embeddings): # <-- NOW ACCEPTS EMBEDDINGS
#     """
#     Builds and saves the vector store using a pre-initialized embedding model.
#     """
#     print("Builder: Starting the vector store build process...")
    
#     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

#     documents = load_and_process_pdfs(pdf_path, config)
#     if not documents:
#         print("Builder: No documents were loaded. Exiting.")
#         return

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#     docs = text_splitter.split_documents(documents)
#     print(f"Builder: Created {len(docs)} chunks.")

#     # --- NO LONGER CREATES EMBEDDINGS, IT USES THE PASSED-IN ONE ---
#     print("Builder: Building FAISS vector store from documents...")
#     vector_store = FAISS.from_documents(docs, embeddings) # <-- USES THE PASSED-IN embeddings
    
#     print("Builder: Saving FAISS index to disk...")
#     vector_store.save_local(vector_store_path)
#     print(f"Builder: Vector store created successfully at {vector_store_path}")

# # This __main__ block is for local testing and is now more complex, but correct
# if __name__ == '__main__':
#     from langchain_huggingface import HuggingFaceEmbeddings
#     with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
#         main_config = yaml.safe_load(f)
    
#     # Create the embeddings object here for the standalone run
#     local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     build_vector_store(main_config, local_embeddings)

# src/vector_store/vector_builder.py

# import sys
# import os
# import yaml
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# # We no longer need to import HuggingFaceEmbeddings here

# # --- System Path Setup ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(PROJECT_ROOT)

# from src.ingestion.pdf_loader import load_and_process_pdfs

# def build_vector_store(config: dict, embeddings): # <-- NOW ACCEPTS EMBEDDINGS
#     """
#     Builds and saves the vector store using a pre-initialized embedding model.
#     """
#     print("Builder: Starting the vector store build process...")
    
#     pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
#     vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

#     documents = load_and_process_pdfs(pdf_path, config)
#     if not documents:
#         print("Builder: No documents were loaded. Exiting.")
#         return

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#     docs = text_splitter.split_documents(documents)
#     print(f"Builder: Created {len(docs)} chunks.")

#     # --- NO LONGER CREATES EMBEDDINGS, IT USES THE PASSED-IN ONE ---
#     print("Builder: Building FAISS vector store from documents...")
#     vector_store = FAISS.from_documents(docs, embeddings) # <-- USES THE PASSED-IN embeddings
    
#     print("Builder: Saving FAISS index to disk...")
#     vector_store.save_local(vector_store_path)
#     print(f"Builder: Vector store created successfully at {vector_store_path}")

# # This __main__ block is for local testing
# if __name__ == '__main__':
#     from langchain_huggingface import HuggingFaceEmbeddings
#     with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
#         main_config = yaml.safe_load(f)
#     local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     build_vector_store(main_config, local_embeddings)

# src/vector_store/vector_builder.py

import sys
import os
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# We no longer need to import HuggingFaceEmbeddings here

# --- System Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.ingestion.pdf_loader import load_and_process_pdfs

def build_vector_store(config: dict, embeddings): # <-- NOW ACCEPTS EMBEDDINGS
    """
    Builds and saves the vector store using a pre-initialized embedding model.
    """
    print("Builder: Starting the vector store build process...")
    
    pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])

    documents = load_and_process_pdfs(pdf_path, config)
    if not documents:
        print("Builder: No documents were loaded. Exiting.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    print(f"Builder: Created {len(docs)} chunks.")

    # --- NO LONGER CREATES EMBEDDINGS, IT USES THE PASSED-IN ONE ---
    print("Builder: Building FAISS vector store from documents...")
    vector_store = FAISS.from_documents(docs, embeddings) # <-- USES THE PASSED-IN embeddings
    
    print("Builder: Saving FAISS index to disk...")
    vector_store.save_local(vector_store_path)
    print(f"Builder: Vector store created successfully at {vector_store_path}")

# This __main__ block is for local testing and is now more complex, but correct
if __name__ == '__main__':
    from langchain_huggingface import HuggingFaceEmbeddings
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), 'r') as f:
        main_config = yaml.safe_load(f)
    
    # Create the embeddings object here for the standalone run
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    build_vector_store(main_config, local_embeddings)