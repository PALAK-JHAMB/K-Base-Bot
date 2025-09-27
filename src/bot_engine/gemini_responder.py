# src/bot_engine/gemini_responder.py

import os
from langchain_openai import ChatOpenAI # <-- NEW IMPORT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

def get_rag_chain(retriever, config: dict):
    """
    Creates and returns a robust RAG chain using the OpenAI API for the LLM.
    """
    print("RAG Chain: Initializing...")

    # --- 1. Load the OpenAI API Key from secrets ---
    # We no longer need the full config dictionary here, just the key.
    if "API_KEY" in st.secrets:
        api_key = st.secrets["API_KEY"]
    else:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets!")
    
    print("RAG Chain: Initializing OpenAI LLM (gpt-3.5-turbo)...")
    # --- USE CHATOPENAI INSTEAD OF GEMINI ---
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key,
        temperature=0.2,
        max_tokens=2048
    )
    print("RAG Chain: OpenAI LLM initialized.")

    # --- 2. Define the Advanced Conditional Prompt ---
    # This prompt works perfectly with GPT models as well.
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
        - Do not say "the provided text excerpts do not offer further details". Write the answer as if you are the definitive expert using only the provided context.
        - After the main answer, skip two lines and add a "Sources:" section, citing the source and page number for the information used.

        Begin:
        """
    )

    # --- 3. Format Documents and Build the Chain ---
    def format_docs_with_sources(docs):
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        
        sources_dict = {}
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            if isinstance(page, int): page += 1

            if source not in sources_dict:
                sources_dict[source] = set()
            if page != "N/A":
                sources_dict[source].add(str(page))
        
        sources_list = []
        for source, pages in sources_dict.items():
            if pages:
                page_str = ", ".join(sorted(list(pages), key=int))
                sources_list.append(f"{source} (Pages: {page_str})")
            else:
                sources_list.append(source)
        
        sources_str = "\n* ".join(sources_list)
        
        # We will append the sources to the context itself, so the LLM can see them for citation.
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