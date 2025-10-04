# import os
# from langchain_huggingface.chat_models import ChatHuggingFace
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st

# def get_rag_chain(retriever, config: dict):
#     print("RAG Chain: Initializing...")
#     if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
#         api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
#     else:
#         raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")
    
#     print("RAG Chain: Initializing LLM via Hugging Face Inference API (Mistral-7B)...")
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     llm = ChatHuggingFace(
#         repo_id=repo_id, huggingfacehub_api_token=api_key, temperature=0.2, max_new_tokens=1024
#     )
#     print("RAG Chain: LLM initialized successfully.")

#     conditional_prompt = PromptTemplate.from_template(
#         """
#         You are an expert technical assistant... [Your full conditional prompt here] ...
#         Begin:
#         """
#     )

#     def format_docs_with_sources(docs):
#         context = "\n\n---\n\n".join([d.page_content for d in docs])
#         sources_dict = {}
#         for doc in docs:
#             source = os.path.basename(doc.metadata.get("source", "Unknown"))
#             page = doc.metadata.get("page", 0) + 1
#             if source not in sources_dict: sources_dict[source] = set()
#             sources_dict[source].add(str(page))
        
#         sources_list = []
#         for source, pages in sources_dict.items():
#             page_str = ", ".join(sorted(list(pages), key=int))
#             sources_list.append(f"{source} (Pages: {page_str})")
        
#         sources_str = "\n* ".join(sources_list)
#         return f"{context}\n\n---SOURCES---\n{sources_str}"

#     rag_chain = (
#         {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
#         | conditional_prompt
#         | llm
#         | StrOutputParser()
#     )
#     print("RAG Chain: Chain built successfully.")
#     return rag_chain

# CHATGPT VAALAAA
import os
import streamlit as st
from langchain_community.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceHub   # ✅ import the hub LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(retriever, config: dict):
    print("RAG Chain: Initializing...")

    # --- API Key from Streamlit Secrets ---
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")

    # --- Initialize Hugging Face Hub model ---
    print("RAG Chain: Initializing LLM via Hugging Face Inference API (Mistral-7B)...")
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    # Create HuggingFaceHub LLM first
    hf_llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=api_key,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 1024}
    )

    # Wrap into a chat model
    llm = ChatHuggingFace(llm=hf_llm)
    print("RAG Chain: LLM initialized successfully.")

    # --- Prompt Template ---
    conditional_prompt = PromptTemplate.from_template(
        """
        You are an expert assistant for railway-related queries.
        Always use the provided context to answer the question. 
        If the answer is not in the context, say you don’t know.
        
        Context:
        {context}

        Question:
        {question}

        Answer (with clarity and precision):
        """
    )

    # --- Format retrieved docs ---
    def format_docs_with_sources(docs):
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources_dict = {}
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", 0) + 1
            sources_dict.setdefault(source, set()).add(str(page))

        sources_list = []
        for source, pages in sources_dict.items():
            page_str = ", ".join(sorted(list(pages), key=int))
            sources_list.append(f"{source} (Pages: {page_str})")

        sources_str = "\n* ".join(sources_list) if sources_list else "Unknown"
        return f"{context}\n\n---SOURCES---\n* {sources_str}"

    # --- Build RAG Chain ---
    rag_chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | conditional_prompt
        | llm
        | StrOutputParser()
    )

    print("RAG Chain: Chain built successfully.")
    return rag_chain

