# # src/bot_engine/gemini_responder.py

# import os
# from langchain_huggingface import HuggingFaceEndpoint # <-- NEW IMPORT
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# def get_rag_chain(retriever, config: dict):
#     """
#     Creates and returns a robust RAG chain using the Hugging Face Inference API for the LLM.
#     This is a free, open-source solution.
#     """
#     print("RAG Chain: Initializing...")

#     # --- 1. Load the Hugging Face API Key from secrets ---
#     if "API_KEY" in st.secrets:
#         api_key = st.secrets["API_KEY"]
#     else:
#         raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")
    
#     # --- 2. Initialize the HuggingFaceEndpoint client ---
#     print("RAG Chain: Initializing LLM via Hugging Face Inference API (Mistral-7B)...")
    
#     # We will use a powerful, free model from Mistral AI.
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
#     llm = HuggingFaceEndpoint(
#         repo_id=repo_id,
#         api_key=api_key,
#         temperature=0.2,
#         max_new_tokens=1024, # Note: parameter is 'max_new_tokens' not 'max_tokens'
#         repetition_penalty=1.2,
#     )
#     print("RAG Chain: LLM initialized successfully.")

#     # --- 3. Define the Advanced Conditional Prompt ---
#     # This prompt works perfectly with Mistral models.
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
#         - Do not say "the provided text excerpts do not offer further details". Write the answer as if you are the definitive expert using only the provided context.
#         - After the main answer, skip two lines and add a "Sources:" section, citing the source and page number for the information used.

#         Begin:
#         """
#     )

#     # --- 4. Format Documents and Build the Chain (This logic is perfect) ---
#     def format_docs_with_sources(docs):
#         context = "\n\n---\n\n".join([d.page_content for d in docs])
        
#         sources_dict = {}
#         for doc in docs:
#             source = os.path.basename(doc.metadata.get("source", "Unknown"))
#             page = doc.metadata.get("page", "N/A")
#             if isinstance(page, int): page += 1

#             if source not in sources_dict:
#                 sources_dict[source] = set()
#             if page != "N/A":
#                 sources_dict[source].add(str(page))
        
#         sources_list = []
#         for source, pages in sources_dict.items():
#             if pages:
#                 page_str = ", ".join(sorted(list(pages), key=int))
#                 sources_list.append(f"{source} (Pages: {page_str})")
#             else:
#                 sources_list.append(source)
        
#         sources_str = "\n* ".join(sources_list)
        
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
from langchain_huggingface.chat_models import ChatHuggingFace # <-- NEW, CORRECT IMPORT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

def get_rag_chain(retriever, config: dict):
    """
    Creates and returns a robust RAG chain using the Hugging Face Inference API
    with the correct ChatHuggingFace client for conversational models.
    """
    print("RAG Chain: Initializing...")

    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")
    
    print("RAG Chain: Initializing LLM via Hugging Face Inference API (Mistral-7B)...")
    
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # --- USE THE CORRECT CLASS: ChatHuggingFace ---
    llm = ChatHuggingFace(
        repo_id=repo_id,
        huggingfacehub_api_token=api_key,
        temperature=0.2,
        max_new_tokens=1024,
    )
    print("RAG Chain: LLM initialized successfully.")

    # --- The rest of your file is PERFECT and does not need to change ---
    conditional_prompt = PromptTemplate.from_template(
        """
        You are an expert technical assistant... [Your full conditional prompt here] ...
        Begin:
        """
    )

    def format_docs_with_sources(docs):
        # ... [Your existing format_docs_with_sources function is correct] ...
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        # ... (rest of the formatting logic) ...
        return f"{context}\n\n---SOURCES---\n..."

    print("RAG Chain: Building the final LCEL chain...")
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | conditional_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG Chain: Chain built successfully.")
    
    return rag_chain