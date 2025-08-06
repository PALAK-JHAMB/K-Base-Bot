# src/vector_store/retriever.py

import nest_asyncio
nest_asyncio.apply()
import yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_retriever():
    """
    Loads the FAISS vector store from the local path specified in the config
    and returns it as a LangChain retriever.
    """
    try:
        # 1. Load configuration
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)

        api_key = config['gemini']['api_key']
        vector_store_path = config['data']['vector_store_path']
        embedding_model = config['gemini']['embedding_model']

        # 2. Initialize embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, 
            google_api_key=api_key
        )
        
        # 3. Load the local FAISS vector store
        log.info(f"Loading vector store from {vector_store_path}...")
        # allow_dangerous_deserialization is needed for FAISS with pickle
        vector_store = FAISS.load_local(
            vector_store_path, 
            embeddings,
            allow_dangerous_deserialization=True 
        )
        
        log.info("Vector store loaded successfully.")
        # Return the vector store as a retriever
        return vector_store.as_retriever(search_kwargs={"k": 6}) # Retrieve top 3 documents

    except FileNotFoundError:
        log.error(f"Vector store not found at {vector_store_path}. Please run vector_builder.py first.")
        return None
    except Exception as e:
        log.error(f"An error occurred while loading the retriever: {e}")
        return None
    
    