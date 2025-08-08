# src/bot_engine/gemini_responder.py

import yaml
import time
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
#     # This chain will be applied to each document.
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
#         - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer.
#         - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using bullet points.

#         User's Original Question: {question}

#         Here are the pieces of extracted information to use:
#         {context}

#         **Instructions for Detailed Answers:**
#         - Provide a complete, step-by-step guide in a numbered list.
#         - Combine all steps from the different pieces of information into one logical sequence.
#         - Ensure the final answer is exhaustive and does not leave out any details.

#         **Instructions for Summary Answers:**
#         - Summarize the key points into 2-4 user-friendly bullet points.
#         - Focus on the main definition, purpose, or outcome.

#         **Final Instruction for ALL answers:**
#         - Do not say "the provided text doesn't contain...". Write the answer as if you are the definitive expert using only the provided context.

#         Begin:
#         """
#     )

#     # --- 4. Construct the full Map-Reduce flow with LCEL ---

#     # This runnable takes the output of the retriever and prepares it for the map_chain.
#     # It creates a list of dictionaries, one for each document.
#     def prepare_map_inputs(inputs: dict):
#         # Add a delay here to rate-limit the start of the mapping process
#         time.sleep(1.5)
#         return [{"document_content": doc.page_content, "question": inputs["question"]} for doc in inputs["documents"]]

#     # This is the main chain that ties everything together.
#     full_chain = (
#         # a. Start with a dictionary containing the user's question.
#         #    Also retrieve the relevant documents.
#         RunnablePassthrough.assign(
#             documents=itemgetter("question") | retriever
#         )
#         # b. The output is now {"question": str, "documents": List[Document]}.
#         #    Now, prepare the inputs for the map stage.
#         | RunnablePassthrough.assign(
#             map_inputs=RunnableLambda(prepare_map_inputs)
#         )
#         # c. The output is now {"question": ..., "documents": ..., "map_inputs": [...]}.
#         #    Run the map chain on the prepared inputs and join the results.
#         | RunnablePassthrough.assign(
#             context=itemgetter("map_inputs") | map_chain.map() | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
#         )
#         # d. The output is now a complete dictionary with "question" and "context".
#         #    This is exactly what the final reduce prompt needs.
#         | reduce_prompt
#         | llm
#         | StrOutputParser()
#     )

#     # Wrap it one last time to handle the initial simple string input from app.py
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
import streamlit as st

def get_rag_chain(retriever):
    """
    Creates and returns a robust RAG chain that implements a Map-Reduce
    style flow with source citation capabilities.
    """
    # 1. Load config and initialize LLM
    # with open("config/settings.yaml", 'r') as f:
    #     config = yaml.safe_load(f)
    # api_key = config['gemini']['api_key']
    # llm_model_name = config['gemini']['llm_model']
    
    # llm = ChatGoogleGenerativeAI(
    #     model=llm_model_name,
    #     google_api_key=api_key,
    #     temperature=0.1,
    #     max_output_tokens=2048
    # )
    try:
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        if "API_KEY" in st.secrets: # <--- CHANGED HERE
            if 'gemini' not in config:
                config['gemini'] = {}
            config['gemini']['api_key'] = st.secrets["API_KEY"] # <--- CHANGED HERE
    except FileNotFoundError:
        if "API_KEY" in st.secrets: # <--- CHANGED HERE
            config = {
                "gemini": {
                    "api_key": st.secrets["API_KEY"], # <--- CHANGED HERE
                    "embedding_model": "models/embedding-001",
                    "llm_model": "models/gemini-1.5-flash-latest"
                }
            }
        else:
            raise ValueError("API Key not found in Streamlit secrets.")
            
    api_key = config['gemini']['api_key']
    llm_model_name = config['gemini']['llm_model']    

    # --- 2. Define the "Map" stage prompt ---
    # This prompt now also asks for the source of the information.
    map_prompt = PromptTemplate.from_template(
        "Based on this document snippet, extract all key information relevant to the user's question.\n"
        "The source of this snippet is '{source}' on page number {page}.\n\n"
        "User's Question: {question}\n\n"
        "Document Snippet:\n{document_content}\n\n"
        "Extracted Information (including source and page number):"
    )
    map_chain = (
        map_prompt
        | llm
        | StrOutputParser()
    )

    # --- 3. Define the "Reduce" stage prompt (Now with Citation Instruction) ---
    reduce_prompt = PromptTemplate.from_template(
        """
        You are an expert technical assistant. You have been given several pieces of extracted information from a user manual.
        Your task is to synthesize this information into a single, high-quality, and clearly formatted answer to the user's original question.

        First, analyze the user's question to determine the required level of detail.
        - If the question asks for a process, steps, or "how to", you MUST provide a detailed, step-by-step guide using a NUMBERED LIST (1., 2., 3., ...).
        - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using BULLET POINTS (*, -, or •).

        User's Original Question: {question}

        Here are the pieces of extracted information to use:
        {context}

        **Final Formatting Instructions:**
        1.  Generate the comprehensive answer according to the rules above (numbered list for steps, bullet points for summaries).
        2.  Add a "Sources:" section.
        3.  Under "Sources:", list each unique source document and its corresponding page number(s) as a bullet point. For example: "* Document_Name.pdf (Page: 10, 12)".
        4.  Do not make up sources or page numbers. Only use the ones provided in the context.
        
        Begin:
        """
    ) # Using a shortened version for brevity, use your full prompt here.

    # --- 4. Construct the full Map-Reduce flow with LCEL ---

    # This runnable prepares the inputs for the map_chain.
    def prepare_map_inputs(inputs: dict):
        time.sleep(1.5)
        # Now we pass the specific metadata fields
        map_inputs = []
        for doc in inputs["documents"]:
            map_inputs.append({
                "document_content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"), # Safely get source
                "page": doc.metadata.get("page", "N/A"),       # Safely get page
                "question": inputs["question"]
            })
        return map_inputs
    # This is the main chain that ties everything together.
    full_chain = (
        RunnablePassthrough.assign(
            documents=itemgetter("question") | retriever
        )
        | RunnablePassthrough.assign(
            map_inputs=RunnableLambda(prepare_map_inputs)
        )
        | RunnablePassthrough.assign(
            context=itemgetter("map_inputs") | map_chain.map() | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
        )
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap it one last time to handle the initial simple string input from app.py
    final_chain = {"question": RunnablePassthrough()} | full_chain
    
    return final_chain