# src/bot_engine/gemini_responder.py

import yaml
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda 
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import MapReduceDocumentsChain, LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import create_retrieval_chain


PROMPT_TEMPLATE = """You are an expert technical assistant. Your task is to answer the user's question with precision and detail, based ONLY on the provided context.

**Instructions:**
1.  Carefully read the entire context provided.
2.  If the user asks for a process or steps, identify ALL the steps in the context and list them out clearly. Use a numbered or bulleted list if appropriate.
3.  Do not summarize a multi-step process into a single sentence. Be thorough.
4.  If the context contains a numbered list (like i, ii, iii), you MUST use those steps in your answer.
5.  If the information is not in the context, state that you cannot answer from the provided documents. Do not add any information.

**Context:**
{context}

**Question:**
{question}

**Detailed Answer:**
"""
def get_rag_chain(retriever):
    """
    Creates and returns a robust RAG chain using the "map_reduce" strategy
    with modern LangChain syntax and rate-limit handling.
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

    # --- 2. Define the "Map" stage prompt and chain (using modern LCEL) ---
    map_prompt = PromptTemplate.from_template(
        "Based on the following document snippet, extract all information relevant to the user's question.\n\n"
        "User's Question: {question}\n\n"
        "Document Snippet:\n{context}\n\n"
        "Extracted Information:"
    )
    map_chain = map_prompt | llm

    # --- 3. Define the "Reduce" stage prompt and chain (using modern LCEL) ---
    reduce_prompt = PromptTemplate.from_template(
        "You are an expert technical writer. You have been given several pieces of extracted information.\n"
        "Your task is to synthesize this information into a single, comprehensive, and well-structured answer to the user's original question.\n\n"
        "User's Original Question: {question}\n\n"
        "Here are the pieces of extracted information:\n{context}\n\n"
        "**Your Final Answer MUST:**\n"
        "1. Start with a clear, detailed definition or explanation.\n"
        "2. If steps are present, provide a complete, step-by-step guide in a numbered list.\n"
        "3. Combine all details into one logical, final answer.\n"
        "4. Do not say 'the provided text doesn't contain...'. Write the answer as if you are the definitive expert.\n\n"
        "Comprehensive Final Answer:"
    )
    reduce_chain = reduce_prompt | llm

    # --- 4. Create the final "stuff" chain to combine documents in the reduce stage ---
    # This chain takes the results of the map stage and "stuffs" them into the reduce prompt
    combine_documents_chain = create_stuff_documents_chain(llm, reduce_prompt)

    # --- 5. Create the main retrieval-and-reduce chain ---
    # This retrieves documents, passes them to the combine_documents_chain, and includes the question
    retrieval_chain = create_retrieval_chain(retriever, combine_documents_chain)

    return retrieval_chain

