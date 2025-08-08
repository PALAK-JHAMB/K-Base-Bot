# # src/bot_engine/gemini_responder.py

# import yaml
# import time
# import os
# from operator import itemgetter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import streamlit as st # Import streamlit to access secrets

# # --- DEFINE PROJECT ROOT for reliable file paths ---
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# def get_rag_chain(retriever):
#     """
#     Creates and returns a robust RAG chain using absolute paths.
#     """
#     # --- 1. Load config and initialize LLM (Hybrid Approach) ---
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
    
#     llm = ChatGoogleGenerativeAI(
#         model=llm_model_name,
#         google_api_key=api_key,
#         temperature=0.1,
#         max_output_tokens=2048
#     )

#     # --- [The rest of your get_rag_chain function is correct] ---
#     # Prompts, map_chain, reduce_prompt, full_chain, final_chain...
#     # No changes are needed for the rest of the file.
    
#     # --- 2. Define the "Map" stage prompt and chain ---
#     map_prompt = PromptTemplate.from_template(
#         "Based on this document snippet, extract all key information relevant to the user's question.\n"
#         "The source of this snippet is '{source}' on page number {page}.\n\n"
#         "User's Question: {question}\n\n"
#         "Document Snippet:\n{document_content}\n\n"
#         "Extracted Information (including source and page number):"
#     )

#     # --- 3. Define the "Reduce" stage prompt (The Conditional One) ---
#     reduce_prompt = PromptTemplate.from_template(
#         """
#         You are an expert technical assistant... [Your full conditional prompt here] ...
#         Begin:
#         """
#     )
    
#     # --- 4. Construct the full Map-Reduce flow with LCEL ---
#     map_chain = (
#         map_prompt
#         | llm
#         | StrOutputParser()
#     )

#     def prepare_map_inputs(inputs: dict):
#         time.sleep(1.5)
#         map_inputs = []
#         for doc in inputs["documents"]:
#             map_inputs.append({
#                 "document_content": doc.page_content,
#                 "source": doc.metadata.get("source", "Unknown"),
#                 "page": doc.metadata.get("page", "N/A"),
#                 "question": inputs["question"]
#             })
#         return map_inputs

#     full_chain = (
#         RunnablePassthrough.assign(
#             documents=itemgetter("question") | retriever
#         )
#         | RunnablePassthrough.assign(
#             map_inputs=RunnableLambda(prepare_map_inputs)
#         )
#         | RunnablePassthrough.assign(
#             context=itemgetter("map_inputs") | map_chain.map() | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
#         )
#         | reduce_prompt
#         | llm
#         | StrOutputParser()
#     )

#     final_chain = {"question": RunnablePassthrough()} | full_chain
    
#     return final_chain
# src/bot_engine/gemini_responder.py

import yaml
import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever):
    """
    Creates and returns a robust RAG chain that correctly implements a Map-Reduce
    style flow with rate-limit handling and conditional response formatting,
    using modern and correct LangChain Expression Language (LCEL).
    """
    # 1. Load config and initialize LLM
    with open("config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)
    api_key = config['gemini']['api_key']
    llm_model_name = config['gemini']['llm_model']
    
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name,
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=2048
    )

    # --- 2. Define the "Map" stage prompt and chain ---
    map_prompt = PromptTemplate.from_template(
        "Based on this document snippet, extract all key information relevant to the user's question.\n\n"
        "User's Question: {question}\n\n"
        "Document Snippet:\n{document_content}\n\n"
        "Extracted Information:"
    )
    map_chain = (
        map_prompt
        | llm
        | StrOutputParser()
    )

    # --- 3. Define the "Reduce" stage prompt (The Conditional One) ---
    reduce_prompt = PromptTemplate.from_template(
        """
        You are an expert technical assistant. You have been given several pieces of extracted information from a user manual.
        Your task is to synthesize this information into a single, high-quality answer to the user's original question.

        First, analyze the user's question to determine the required level of detail.
        - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer using a NUMBERED LIST.
        - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using BULLET POINTS.

        User's Original Question: {question}

        Here are the pieces of extracted information to use:
        {context}

        **Final Instruction for ALL answers:**
        - Do not say "the provided text excerpts do not offer further details" or similar phrases.
        - Write the answer as if you are the definitive expert using only the provided context.
        - add a "Sources:" section, citing the source and page number for the information used.
        - "Source" must give exact range of page numbers from which manual they are picked with manual name.

        Begin:
        """
    )

    # --- 4. Construct the full Map-Reduce flow with LCEL ---
    # def prepare_map_inputs(inputs: dict):
    #     time.sleep(1.5)
    #     return [{"document_content": doc.page_content, "question": inputs["question"]} for doc in inputs["documents"]]

    # full_chain = (
    #     RunnablePassthrough.assign(
    #         documents=itemgetter("question") | retriever
    #     )
    #     | RunnablePassthrough.assign(
    #         map_inputs=RunnableLambda(prepare_map_inputs)
    #     )
    #     | RunnablePassthrough.assign(
    #         context=itemgetter("map_inputs") | map_chain.map() | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
    #     )
    #     | reduce_prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # final_chain = {"question": RunnablePassthrough()} | full_chain
    
    # return final_chain
    
     def format_docs_with_sources(docs):
        # This helper function creates both the context and a clean sources list
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        sources = set()
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            sources.add(f"{source} (Page: {page})")
        sources_str = "\n* ".join(sorted(list(sources)))
        return {"context": context, "sources": sources_str}

    full_chain = (
        RunnablePassthrough.assign(
            # Retrieve documents and format them into context and sources
            retrieved_info=itemgetter("question") | retriever | RunnableLambda(format_docs_with_sources)
        )
        # Unpack the context and sources from the retrieved_info dictionary
        | RunnablePassthrough.assign(
            context=itemgetter("retrieved_info") | itemgetter("context"),
            sources=itemgetter("retrieved_info") | itemgetter("sources")
        )
        | {
            "answer": reduce_prompt | llm | StrOutputParser(),
            "sources": itemgetter("sources") # Pass the sources through to the final output
          }
    )

    # Final wrapper to handle input and format the final output string
    def format_final_output(result: dict):
        return f"{result['answer']}\n\n**Sources:**\n* {result['sources']}"

    final_chain = {"question": RunnablePassthrough()} | full_chain | RunnableLambda(format_final_output)
    
    return final_chain