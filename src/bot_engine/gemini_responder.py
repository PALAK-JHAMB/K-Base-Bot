
# import yaml
# import time
# from operator import itemgetter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# def get_rag_chain(retriever):
#     """
#     Creates and returns a robust RAG chain that correctly implements a Map-Reduce
#     style flow with rate-limit handling and conditional response formatting,
#     using modern and correct LangChain Expression Language (LCEL).
#     """
#     # 1. Load config and initialize LLM
#     with open("config/settings.yaml", 'r') as f:
#         config = yaml.safe_load(f)
#     api_key = config['gemini']['api_key']
#     llm_model_name = config['gemini']['llm_model']
    
#     llm = ChatGoogleGenerativeAI(
#         model=llm_model_name,
#         google_api_key=api_key,
#         temperature=0.1,
#         max_output_tokens=2048
#     )

#     # --- 2. Define the "Map" stage prompt and chain ---
#     map_prompt = PromptTemplate.from_template(
#         "Based on this document snippet, extract all key information relevant to the user's question.\n\n"
#         "User's Question: {question}\n\n"
#         "Document Snippet:\n{document_content}\n\n"
#         "Extracted Information:"
#     )
#     map_chain = (
#         map_prompt
#         | llm
#         | StrOutputParser()
#     )

#     # --- 3. Define the "Reduce" stage prompt (The Conditional One) ---
#     reduce_prompt = PromptTemplate.from_template(
#         """
#         You are an expert technical assistant. You have been given several pieces of extracted information from a user manual.
#         Your task is to synthesize this information into a single, high-quality answer to the user's original question.

#         First, analyze the user's question to determine the required level of detail.
#         - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer using a NUMBERED LIST.
#         - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using BULLET POINTS.

#         User's Original Question: {question}

#         Here are the pieces of extracted information to use:
#         {context}

#         **Final Instruction for ALL answers:**
#         - Do not say "the provided text excerpts do not offer further details" or similar phrases.
#         - Write the answer as if you are the definitive expert using only the provided context.
#         - add a "Sources:" section, citing the source and page number for the information used.
#         - "Source" must give exact range of page numbers from which manual they are picked with manual name.

#         Begin:
#         """
#     )

    
    
#     def format_docs_with_sources(docs):
#         # This helper function creates both the context and a clean sources list
#         context = "\n\n---\n\n".join([d.page_content for d in docs])
#         sources = set()
#         for doc in docs:
#             source = doc.metadata.get("source", "Unknown")
#             page = doc.metadata.get("page", "N/A")
#             sources.add(f"{source} (Page: {page})")
#         sources_str = "\n* ".join(sorted(list(sources)))
#         return {"context": context, "sources": sources_str}

#     full_chain = (
#         RunnablePassthrough.assign(
#             # Retrieve documents and format them into context and sources
#             retrieved_info=itemgetter("question") | retriever | RunnableLambda(format_docs_with_sources)
#         )
#         # Unpack the context and sources from the retrieved_info dictionary
#         | RunnablePassthrough.assign(
#             context=itemgetter("retrieved_info") | itemgetter("context"),
#             sources=itemgetter("retrieved_info") | itemgetter("sources")
#         )
#         | {
#             "answer": reduce_prompt | llm | StrOutputParser(),
#             "sources": itemgetter("sources") # Pass the sources through to the final output
#           }
#     )

#     # Final wrapper to handle input and format the final output string
#     def format_final_output(result: dict):
#         return f"{result['answer']}\n\n**Sources:**\n* {result['sources']}"

#     final_chain = {"question": RunnablePassthrough()} | full_chain | RunnableLambda(format_final_output)
    
#     return final_chain    
# src/bot_engine/gemini_responder.py

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
#     Creates and returns a simple, robust RAG chain using the "Stuff" method.
#     This is the most reliable chain type for deployment.
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
#         temperature=0.2,
#         max_output_tokens=2048
#     )
#     print("RAG Chain: Gemini LLM initialized.")

#     # --- 2. Define a Simple, Robust Prompt ---
#     prompt = PromptTemplate(
#         template="""You are a helpful assistant. Answer the user's question based only on the following context.
#         If the answer is not in the context, say you don't know.
#         Provide a detailed answer and list any steps in a clear format.
#         After your answer, list the sources of the documents you used.

#         Context:
#         {context}

#         Question: {question}

#         Answer:""",
#         input_variables=["context", "question"],
#     )

#     # --- 3. Format Documents and Build the Chain ---
#     def format_docs(docs):
#         # Joins the content of all retrieved documents into a single string
#         # Also includes the source metadata for citations
#         formatted_context = []
#         sources = set()
#         for doc in docs:
#             formatted_context.append(doc.page_content)
#             source = doc.metadata.get("source", "Unknown")
#             page = doc.metadata.get("page", "N/A")
#             sources.add(f"{source} (Page: {page})")
        
#         context_str = "\n\n---\n\n".join(formatted_context)
#         sources_str = "\n\nSources:\n* " + "\n* ".join(sorted(list(sources)))
#         return context_str + sources_str

#     print("RAG Chain: Building the final LCEL chain...")
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     print("RAG Chain: Chain built successfully.")
    
#     return rag_chain
# src/bot_engine/gemini_responder.py

import yaml
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# --- DEFINE PROJECT ROOT for reliable file paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_rag_chain(retriever):
    """
    Creates and returns a robust RAG chain using the "Stuff" method,
    but with an advanced, conditional prompt to control the output format.
    """
    print("RAG Chain: Initializing...")
    
    # --- 1. Load Config (Hybrid Approach) ---
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
                "gemini": {
                    "api_key": st.secrets["API_KEY"],
                    "llm_model": "models/gemini-1.5-flash-latest"
                }
            }
        else:
            raise ValueError("API Key not found in Streamlit secrets.")
            
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
        - After the main answer, skip two lines and add a "Sources:" section, citing the source and page number for the information used.

        Begin:
        """
    )

    # --- 3. Format Documents and Build the Chain ---
    def format_docs_with_sources(docs):
        # Joins the content of all retrieved documents and formats the sources
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        
        sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "Unknown").replace('\\', '/').split('/')[-1] # Clean up path
            page = doc.metadata.get("page", "N/A")
            sources.add(f"{source} (Page: {page})")
        
        sources_str = "\n* ".join(sorted(list(sources)))
        # We will append the sources to the context itself, so the LLM can see them.
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