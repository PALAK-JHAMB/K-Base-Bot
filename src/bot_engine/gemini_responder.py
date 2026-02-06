
import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(retriever, config: dict):
    print("RAG Chain: Initializing with local model (Flan-T5)...")

    # --- Load model from config or default ---
    repo_id = config.get("huggingface", {}).get("llm_repo_id", "google/flan-t5-small")
    print(f"🔎 Requested local model: {repo_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

        # Build pipeline with truncation + safe max length
        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,          # CPU
            max_length=512,     # keep well under model's 2048 limit
            truncation=True
        )

        llm = HuggingFacePipeline(pipeline=generator)
        print(f"✅ Local model loaded: {repo_id}")
    except Exception as e:
        print(f"⚠️ Failed to load {repo_id} → {e}")
        repo_id = "distilgpt2"
        from transformers import AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_length=256,
            truncation=True
        )
        llm = HuggingFacePipeline(pipeline=generator)
        print(f"👉 Fallback activated: {repo_id}")

    # Show active model in sidebar
    st.sidebar.info(f"**Active Local Model:** {repo_id}")

    # --- Prompt Template ---
    conditional_prompt = PromptTemplate.from_template(
    """
    You are an expert assistant for Indian Railways queries.
    Answer the question clearly using ONLY the provided context. 
    Follow these rules strictly:
    1. Give the answer in **bullet points** (• or -).
    2. Keep each point short and clear (1–2 sentences max).
    3. If the answer is not in the context, reply: "I don’t know based on the provided documents."
    4. At the end of your answer, include a **Sources** section with the documents/pages mentioned in the context.

    Context:
    {context}

    Question:
    {question}

    Final Answer:
    """
)


    # --- Format retrieved docs with safeguard ---
    def format_docs_with_sources(docs):
        context = "\n\n---\n\n".join([d.page_content for d in docs])

        # 🚨 Trim context to avoid exceeding model range
        max_tokens = 800  # roughly safe under flan-t5-small’s 2048
        words = context.split()
        if len(words) > max_tokens:
            context = " ".join(words[:max_tokens]) + "... [truncated]"

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

    print(f" Final active local model: {repo_id}")
    return rag_chain
