import streamlit as st
import tempfile

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector database
from langchain_community.vectorstores import FAISS

# Ollama local LLM
from langchain_community.llms import Ollama

# NEW LangChain Retrieval System
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title="Local RAG PDF Chatbot")

st.title("📚 Local RAG Chatbot (knowledgeBase)")
st.write("Upload PDF files and ask questions locally")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    st.success(f"✅ Total chunks: {len(split_docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

    st.success("🎉 Documents processed")

    llm = Ollama(model="phi3")

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the context below.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, document_chain)

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})

        st.subheader("Answer")
        st.write(response["answer"])