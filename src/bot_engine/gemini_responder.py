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
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(retriever, config: dict):
    print("RAG Chain: Initializing...")

    # --- API Key ---
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets!")

    # --- Load model repo ID from config ---
    requested_repo = config.get("huggingface", {}).get("llm_repo_id", "google/flan-t5-base")
    active_repo = requested_repo
    print(f"🔎 Requested LLM from config: {requested_repo}")

    # --- Try to initialize LLM ---
    try:
        llm = HuggingFaceEndpoint(
            repo_id=requested_repo,
            task="text-generation",
            huggingfacehub_api_token=api_key,
            temperature=0.2,
            max_new_tokens=512
        )
        print(f"✅ Using Hugging Face LLM: {requested_repo}")
    except Exception as e:
        print(f"⚠️ Failed to init {requested_repo} → {e}")
        active_repo = "google/flan-t5-base"
        print(f"👉 Falling back to: {active_repo}")
        llm = HuggingFaceEndpoint(
            repo_id=active_repo,
            task="text-generation",
            huggingfacehub_api_token=api_key,
            temperature=0.2,
            max_new_tokens=512
        )

    # --- Display active model in Streamlit ---
    st.sidebar.info(f"**Active LLM Model:** {active_repo}")

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

    # --- Format retrieved docs with sources ---
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

    print(f" Final Active LLM in use: {active_repo}")
    return rag_chain
