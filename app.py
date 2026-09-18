"""
ResearchMate — Research Paper RAG Chatbot
Streamlit UI

Run locally with:
    streamlit run app.py

Requires:
    - A "faiss_index/" folder in the same directory (unzip faiss_index.zip from Stage 6/7 here)
    - A ".env" file in the same directory containing: GROQ_API_KEY=your_key_here
"""

import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------------------------------
# Structured response shape (same as Stage 9)
# ---------------------------------------------------------------------------
class RAGResponse(BaseModel):
    answer: str
    relevant_papers: List[str]
    topics: List[str]


# ---------------------------------------------------------------------------
# Pipeline setup (cached so it only runs once per session, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ResearchMate (this happens once)...")
def load_pipeline():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY not found. Create a .env file in this folder containing:\n"
            "GROQ_API_KEY=your_key_here"
        )
        st.stop()

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.isdir("faiss_index"):
        st.error(
            "faiss_index/ folder not found. Unzip faiss_index.zip (from Stage 6/7) "
            "into this same directory as app.py."
        )
        st.stop()

    vectorstore = FAISS.load_local(
        "faiss_index", embedding_model, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    system_prompt = """You are ResearchMate, a research paper assistant.
Answer the user's question using ONLY the research paper context provided below.

Rules:
- Do not invent or assume any information that is not present in the context.
- If the context does not contain enough information to answer the question, say so clearly instead of guessing.
- When you use information from a paper, mention its title.
- Keep your answer clear and concise.

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )

    def format_docs(docs):
        formatted = []
        for doc in docs:
            formatted.append(f"Title: {doc.metadata['title']}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, answer_chain


def ask_researchmate(question: str, retriever, answer_chain) -> RAGResponse:
    docs = retriever.invoke(question)
    answer_text = answer_chain.invoke(question)

    relevant_papers = [doc.metadata["title"] for doc in docs]

    all_topics = set()
    for doc in docs:
        for topic in doc.metadata["topics"].split(", "):
            if topic:
                all_topics.add(topic)

    return RAGResponse(
        answer=answer_text,
        relevant_papers=relevant_papers,
        topics=sorted(all_topics),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ResearchMate", page_icon="📚", layout="centered")

st.title("📚 ResearchMate")
st.caption(
    "A RAG-based research paper chatbot. Ask a question and it searches across "
    "~21,000 real arXiv paper abstracts (Computer Science, Physics, Mathematics, "
    "Statistics, Quantitative Biology, Quantitative Finance) to give a grounded answer."
)

retriever, answer_chain = load_pipeline()

question = st.text_input("Ask a research question:", placeholder="e.g. What research has been done on graph neural networks?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Searching papers and generating an answer..."):
        response = ask_researchmate(question, retriever, answer_chain)

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Relevant Papers")
    for title in response.relevant_papers:
        st.markdown(f"- {title}")

    st.subheader("Topics")
    st.write(", ".join(response.topics) if response.topics else "None")

elif ask_clicked and not question.strip():
    st.warning("Please enter a question first.")
