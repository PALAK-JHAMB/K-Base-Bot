

# import yaml
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# # --- DEFINE PROJECT ROOT for reliable file paths ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# def get_rag_chain(retriever):
#     """
#     Creates and returns a robust RAG chain using the "Stuff" method,
#     but with an advanced, conditional prompt to control the output format.
#     """
#     print("RAG Chain: Initializing...")
    
#     # --- 1. Load Config (Hybrid Approach) ---
#     config = {}
#     settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
#     try:
#         with open(settings_path, 'r') as f:
#             config = yaml.safe_load(f)
#         if "API_KEY" in st.secrets:
#             config['gemini']['api_key'] = st.secrets["API_KEY"]
#     except FileNotFoundError:
#         if "API_KEY" in st.secrets:
#             config = {
#                 "gemini": {
#                     "api_key": st.secrets["API_KEY"],
#                     "llm_model": "models/gemini-1.5-flash-latest"
#                 }
#             }
#         else:
#             raise ValueError("API Key not found in Streamlit secrets.")
            
#     api_key = config['gemini']['api_key']
#     llm_model_name = config['gemini']['llm_model']
    
#     print("RAG Chain: Initializing Gemini LLM...")
#     llm = ChatGoogleGenerativeAI(
#         model=llm_model_name,
#         google_api_key=api_key,
#         temperature=0.1, # Lowered for more factual responses
#         max_output_tokens=2048
#     )
#     print("RAG Chain: Gemini LLM initialized.")

#     # --- 2. Define the Advanced Conditional Prompt ---
#     # This is your powerful prompt that handles both summaries and detailed steps.
#     conditional_prompt = PromptTemplate.from_template(
#         """
#         You are an expert technical assistant. You have been given the following context from a user manual.
#         Your task is to synthesize this information into a single, high-quality answer to the user's original question.

#         First, analyze the user's question to determine the required level of detail.
#         - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer using a NUMBERED LIST.
#         - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using BULLET POINTS.

#         User's Original Question: {question}

#         Context to use:
#         {context}

#         **Final Instruction for ALL answers:**
#         - Do not say "the provided text excerpts do not offer further details" or similar phrases.
#         - Write the answer as if you are the definitive expert using only the provided context.
#         - After the main answer, skip two lines and add a "Sources:" section, citing the source and page number for the information used.

#         Begin:
#         """
#     )

#     # --- 3. Format Documents and Build the Chain ---
#     def format_docs_with_sources(docs):
#         # Joins the content of all retrieved documents and formats the sources
#         context = "\n\n---\n\n".join([d.page_content for d in docs])
        
#         sources = set()
#         for doc in docs:
#             source = doc.metadata.get("source", "Unknown").replace('\\', '/').split('/')[-1] # Clean up path
#             page = doc.metadata.get("page", "N/A")
#             sources.add(f"{source} (Page: {page})")
        
#         sources_str = "\n* ".join(sorted(list(sources)))
#         # We will append the sources to the context itself, so the LLM can see them.
#         return f"{context}\n\n---SOURCES---\n{sources_str}"

#     print("RAG Chain: Building the final LCEL chain...")
#     rag_chain = (
#         {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
#         | conditional_prompt
#         | llm
#         | StrOutputParser()
#     )
#     print("RAG Chain: Chain built successfully.")
    
#     return rag_chain

# src/bot_engine/gemini_responder.py

# src/bot_engine/gemini_responder.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever, config: dict):
    """
    Creates and returns a robust RAG chain using the "Stuff" method,
    but with an advanced, conditional prompt to control the output format.
    This version is architected for stable deployment.
    """

    # --- 1. Get configuration from the passed-in dictionary ---
    api_key = config['gemini']['api_key']
    llm_model_name = config['gemini']['llm_model']
    
    print("RAG Chain: Initializing Gemini LLM...")
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        google_api_key=api_key,
        temperature=0.1, # Lowered for more factual responses
        max_output_tokens=2048
    )
    print("RAG Chain: Gemini LLM initialized.")

    # --- 2. Define the Advanced Conditional Prompt ---
    # This is your powerful prompt that handles both summaries and detailed steps.
    conditional_prompt = PromptTemplate.from_template(
        """
        You are an expert technical assistant. You have been given the following context from a user manual.
        Your task is to synthesize this information into a single, high-quality answer to the user's original question.

        First, analyze the user's question to determine the required level of detail.
        - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer using a NUMBERED LIST.
        - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using BULLET POINTS.

        User's Original Question: {question}

        Context to use:
        {context}

        **Final Instruction for ALL answers:**
        - Do not say "the provided text excerpts do not offer further details" or similar phrases.
        - Write the answer as if you are the definitive expert using only the provided context.
        - After the main answer, skip two lines and add a "Sources:" section, citing the source and page number for the information used. The sources are provided at the end of the context.

        Begin:
        """
    )

    # --- 3. Format Documents and Build the Chain ---
    def format_docs_with_sources(docs):
        # Joins the content of all retrieved documents and formats the sources
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        
        # Create a clean, unique list of sources with page numbers
        sources_dict = {}
        for doc in docs:
            # Clean up the source path to just the filename
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            
            # Add 1 to page number if it's an integer (since they are often 0-indexed)
            if isinstance(page, int):
                page += 1

            if source not in sources_dict:
                sources_dict[source] = set()
            if page != "N/A":
                sources_dict[source].add(str(page))
        
        # Format the sources string beautifully
        sources_list = []
        for source, pages in sources_dict.items():
            if pages:
                page_str = ", ".join(sorted(list(pages), key=int))
                sources_list.append(f"{source} (Pages: {page_str})")
            else:
                sources_list.append(source)
        
        sources_str = "\n* ".join(sources_list)
        
        # Append the sources to the context itself, so the LLM can see them for citation.
        return f"{context}\n\n---SOURCES---\n{sources_str}"

    print("RAG Chain: Building the final LCEL chain...")
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | conditional_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG Chain: Chain built successfully.")
    
    return rag_chain