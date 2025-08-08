import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ingestion.pdf_loader import load_and_process_pdfs

def build_vector_store():
    """Builds and saves the vector store from PDF documents using advanced parsing."""
    # 1. Load configuration
    # with open("config/settings.yaml", 'r') as f:
    #     config = yaml.safe_load(f)
    settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
    api_key = config['gemini']['api_key']
    # pdf_path = config['data']['pdf_path']
    # vector_store_path = config['data']['vector_store_path']
    pdf_path = os.path.join(PROJECT_ROOT, config['data']['pdf_path'])
    vector_store_path = os.path.join(PROJECT_ROOT, config['data']['vector_store_path'])
    # 2. Load PDF documents using our new, powerful loader
    print("Loading and processing PDF documents with 'unstructured'...")
    documents = load_and_process_pdfs(pdf_path, config)
    if not documents:
        print("No documents were loaded. Check your PDF path and file content. Exiting.")
        return

    # 3. Chunk the documents
    # print(f"Chunking {len(documents)} processed documents...")
    # # These documents are already logically grouped, but we still chunk them to fit context windows
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1500, 
    #     chunk_overlap=200,
    #     separators=["\n\n", "\n", " ", ""]
    # )
    # docs = text_splitter.split_documents(documents)
    # print(f"Created {len(docs)} chunks for embedding.")
    
    print(f"Chunking {len(documents)} processed documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        # Increase chunk size to ensure full procedures are captured
        chunk_size=2000, 
        
        # A slightly larger overlap is good practice with larger chunks
        chunk_overlap=300,
        
        # These are the characters it will try to split on, in order.
        # \n\n is the most important for keeping paragraphs/lists together.
        separators=["\n\n", "\n", ". ", " ", ""],
        
        # This is an important parameter to keep metadata
        keep_separator=False
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks for embedding.")
    # ... rest of the function ...
    # 4. Create embeddings
    print("Creating embeddings with Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config['gemini']['embedding_model'], google_api_key=api_key)

    # 5. Build and save the vector store (FAISS)
    print("Building and saving FAISS vector store...")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"Vector store created successfully at {vector_store_path}")

if __name__ == '__main__':
    build_vector_store()