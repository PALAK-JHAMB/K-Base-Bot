# # src/bot_engine/gemini_responder.py

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
#     strategy with rate-limit handling and conditional response formatting,
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
#         RunnableLambda(lambda x: time.sleep(1.5) or x)  # Add delay for rate limiting
#         | map_prompt
#         | llm
#         | StrOutputParser()
#     )

#     # --- 3. Define the "Reduce" stage prompt ---
#     reduce_prompt = PromptTemplate.from_template(
#         """
#         You are an expert technical assistant. You have been given several pieces of extracted information from a user manual.
#         # Your task is to synthesize this information into a single, high-quality answer to the user's original question.

#         # First, analyze the user's question to determine the required level of detail.
#         # - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer.
#         # - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using bullet points.

#         # User's Original Question: {question}

#         # Here are the pieces of extracted information to use:
#         # {context}

#         # **Instructions for Detailed Answers:**
#         # - Provide a complete, step-by-step guide in a numbered list.
#         # - Combine all steps from the different pieces of information into one logical sequence.
#         # - Ensure the final answer is exhaustive and does not leave out any details.

#         # **Instructions for Summary Answers:**
#         # - Summarize the key points into 2-4 user-friendly bullet points.
#         # - Focus on the main definition, purpose, or outcome.

#         # **Final Instruction for ALL answers:**
#         # - Do not say "the provided text doesn't contain...". Write the answer as if you are the definitive expert using only the provided context.

#         # Begin:
#         # """
#     )
#     # Using a shortened version for brevity, use your full prompt here.

#     # --- 4. Construct the full Map-Reduce flow with LCEL ---

#     # This runnable takes the output of the retriever and prepares it for the map_chain.
#     # It creates a list of dictionaries, one for each document.
#     map_input_preparer = RunnableLambda(
#         lambda x: [{"document_content": doc.page_content, "question": x["question"]} for doc in x["documents"]]
#     )

#     This is the main chain that ties everything together.
#     full_chain = (
#         # a. Start with a dictionary containing the user's question.
#         #    Also retrieve the relevant documents.
#         RunnablePassthrough.assign(
#             documents=itemgetter("question") | retriever
#         )
#         # b. The output is now {"question": str, "documents": List[Document]}.
#         #    Now, create the 'context' for the reduce stage.
#         | RunnablePassthrough.assign(
#             context=(
#                 # First, prepare the input for the map operation.
#                 map_input_preparer
#                 # Then, run the map_chain on each item in the list.
#                 # The .map() method creates a runnable that processes lists.
#                 | map_chain.map()
#                 # Finally, join the list of string results into a single block.
#                 | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
#             )
#         )
#         # c. The output is now {"question": str, "documents": ..., "context": str}.
#         #    This dictionary has everything the reduce_prompt needs.
#         | reduce_prompt
#         | llm
#         | StrOutputParser()
#     )

#     # Wrap it one last time to handle the initial simple string input from app.py
#     final_chain = {"question": RunnablePassthrough()} | full_chain
    
#     return final_chain
# src/bot_engine/gemini_responder.py

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
    style flow with rate-limit handling and conditional response formatting.
    This version ensures correct data flow to the final prompt.
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
        - If the question contains words like "detail", "explain", "how to", "steps", "process", or is a "what are the steps" type of question, you MUST provide a detailed, step-by-step answer.
        - For all other questions (e.g., "what is", "describe"), you MUST provide a concise, high-level summary using bullet points.

        User's Original Question: {question}

        Here are the pieces of extracted information to use:
        {context}

        **Instructions for Detailed Answers:**
        - Provide a complete, step-by-step guide in a numbered list.
        - Combine all steps from the different pieces of information into one logical sequence.
        - Ensure the final answer is exhaustive and does not leave out any details.

        **Instructions for Summary Answers:**
        - Summarize the key points into 2-4 user-friendly bullet points.
        - Focus on the main definition, purpose, or outcome.

        **Final Instruction for ALL answers:**
        - Do not say "the provided text doesn't contain...". Write the answer as if you are the definitive expert using only the provided context.

        Begin:
        """
    ) # Using a shortened version for brevity, use your full prompt here.

    # --- 4. Construct the full Map-Reduce flow with LCEL ---

    # This runnable takes the output of the retriever and prepares it for the map_chain.
    # It creates a list of dictionaries, one for each document.
    def prepare_map_inputs(inputs: dict):
        # Add a delay here to rate-limit the start of the mapping process
        time.sleep(1.5)
        return [{"document_content": doc.page_content, "question": inputs["question"]} for doc in inputs["documents"]]

    # This is the main chain that ties everything together.
    full_chain = (
        # a. Start with a dictionary containing the user's question.
        #    Also retrieve the relevant documents.
        RunnablePassthrough.assign(
            documents=itemgetter("question") | retriever
        )
        # b. The output is now {"question": str, "documents": List[Document]}.
        #    Now, prepare the inputs for the map stage.
        | RunnablePassthrough.assign(
            # The result of this will be added as a new key, 'map_inputs'
            map_inputs=RunnableLambda(prepare_map_inputs)
        )
        # c. The output is now {"question": ..., "documents": ..., "map_inputs": [...]}.
        #    Run the map chain on the prepared inputs and join the results.
        | RunnablePassthrough.assign(
            # The result of this will be added as a new key, 'context'
            context=itemgetter("map_inputs") | map_chain.map() | (lambda mapped_results: "\n\n---\n\n".join(mapped_results))
        )
        # d. The output is now a complete dictionary with "question" and "context".
        #    This is exactly what the final reduce prompt needs.
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap it one last time to handle the initial simple string input from app.py
    final_chain = {"question": RunnablePassthrough()} | full_chain
    
    return final_chain