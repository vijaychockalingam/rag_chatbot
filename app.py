import streamlit as st
import tempfile
import re
import os
import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="StudyMind", page_icon="🧠", layout="wide")

# ─── Session State Init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Persistent Storage Paths ───────────────────────────────────────────────
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "studymind_data")
PDF_DIR     = os.path.join(STORAGE_DIR, "pdfs")
FAISS_DIR   = os.path.join(STORAGE_DIR, "faiss_index")
META_FILE   = os.path.join(STORAGE_DIR, "metadata.json")

os.makedirs(PDF_DIR,   exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)


def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return json.load(f)
    return {"files": []}


def save_metadata(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f)


def delete_pdf(filename):
    path = os.path.join(PDF_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    meta = load_metadata()
    meta["files"] = [f for f in meta["files"] if f != filename]
    save_metadata(meta)
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
        os.makedirs(FAISS_DIR, exist_ok=True)
    st.cache_resource.clear()


# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #212121; --surface: #2f2f2f; --card: #3a3a3a;
    --border: #4a4a4a; --text: #ececec; --muted: #8e8ea0;
    --accent: #ab68ff; --input-bg: #2f2f2f; --danger: #ef4444;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important; visibility: hidden !important;
}
.main .block-container {
    max-width: 760px !important;
    padding: 0 1rem 8rem !important;
    margin: 0 auto !important;
}
[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #333 !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stFileUploader"] {
    background: #2a2a2a !important;
    border: 1.5px dashed #444 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stButton > button {
    background: #2f2f2f !important; color: var(--text) !important;
    border: 1px solid #444 !important; border-radius: 8px !important;
    font-size: 0.85rem !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3a3a3a !important; border-color: var(--accent) !important;
}

.pdf-card {
    background: #1e1e1e; border: 1px solid #2e2e2e;
    border-radius: 10px; padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem; display: flex;
    align-items: center; justify-content: space-between;
    transition: border-color 0.2s;
}
.pdf-card:hover { border-color: #444; }
.pdf-name {
    font-size: 0.8rem; color: #ccc;
    overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 160px;
}
.pdf-icon { color: #ab68ff; margin-right: 0.4rem; font-size: 0.9rem; }

[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 1.2rem 0 !important; margin: 0 !important;
    border-bottom: 1px solid #2a2a2a !important;
    animation: fadeIn 0.25s ease !important;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] p {
    color: var(--text) !important; font-size: 0.95rem !important;
    line-height: 1.8 !important; margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #2a2a2a !important; border-radius: 12px !important;
    border: none !important; padding: 1rem 1.2rem !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important; padding: 1rem 0.2rem !important;
}
code {
    background: #1a1a1a !important; border: 1px solid #333 !important;
    border-radius: 6px !important; padding: 0.15rem 0.4rem !important;
    font-size: 0.85rem !important; color: #ab68ff !important;
}
pre {
    background: #1a1a1a !important; border: 1px solid #333 !important;
    border-radius: 10px !important; padding: 1rem !important;
}
pre code { border: none !important; padding: 0 !important; color: #e2e8f0 !important; }
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: var(--text) !important; font-weight: 600 !important;
    margin: 1rem 0 0.4rem !important;
}
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    padding-left: 1.4rem !important; margin: 0.4rem 0 !important;
}
[data-testid="stChatMessage"] li {
    color: var(--text) !important; line-height: 1.8 !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stChatMessage"] strong { color: #fff !important; font-weight: 600 !important; }

[data-testid="stChatInput"] {
    background: var(--input-bg) !important; border: 1px solid #444 !important;
    border-radius: 16px !important; box-shadow: 0 4px 24px rgba(0,0,0,0.5) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 4px 24px rgba(171,104,255,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important; color: var(--text) !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #666 !important; }
[data-testid="stChatInput"] button {
    background: var(--accent) !important; border-radius: 8px !important;
}
[data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.85rem !important; }
[data-testid="stAlert"] {
    background: #1e1e1e !important; border: 1px solid #333 !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-size: 0.85rem !important;
}
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.8rem 0 1.2rem; }
.chip {
    background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 20px;
    padding: 0.25rem 0.8rem; font-size: 0.78rem; color: #aaa;
}
.chip span { color: var(--accent); font-weight: 500; }
.welcome { text-align: center; padding: 5rem 1rem 3rem; }
.welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.welcome h2 { font-size: 1.6rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem; }
.welcome p { color: var(--muted); font-size: 0.9rem; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────
def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if re.search(r"\d{2}:\d{2}:\d{2}", line): continue
        if re.search(r"\[.*Thread.*\]", line): continue
        if re.search(r"(WARN|INFO|ERROR|DEBUG):", line): continue
        if re.search(r"(Tomcat|Apache|localhost)", line): continue
        if len(line.strip()) < 5: continue
        cleaned.append(line)
    return "\n".join(cleaned)


@st.cache_resource
def build_vectorstore(pdf_paths_tuple):
    docs = []
    for path in pdf_paths_tuple:
        loader = PyPDFLoader(path)
        pages = loader.load()
        total = len(pages)
        for i, page in enumerate(pages):
            page.page_content = clean_text(page.page_content)
            page.metadata["page_number"]  = i + 1
            page.metadata["total_pages"]  = total
            page.metadata["is_last_page"]  = (i == total - 1)
            page.metadata["is_first_page"] = (i == 0)
        docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    split_docs = [d for d in split_docs if len(d.page_content.strip()) > 100]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore, len(split_docs), len(docs)


def smart_retrieve(vectorstore, query):
    q = query.lower()

    # ✅ Last page intent
    if any(w in q for w in ["last", "final", "end", "last page", "last program",
                              "last question", "last exercise"]):
        all_docs = vectorstore.similarity_search(query, k=20)
        last_docs = [
            d for d in all_docs
            if d.metadata.get("is_last_page") or
               d.metadata.get("page_number", 0) >=
               d.metadata.get("total_pages", 999) - 2
        ]
        if last_docs:
            return last_docs[:6]

    # ✅ First page intent
    if any(w in q for w in ["first", "beginning", "start", "first page",
                              "first program", "first question"]):
        all_docs = vectorstore.similarity_search(query, k=20)
        first_docs = [
            d for d in all_docs
            if d.metadata.get("is_first_page") or
               d.metadata.get("page_number", 999) <= 2
        ]
        if first_docs:
            return first_docs[:6]

    # ✅ Default — wider retrieval
    return vectorstore.similarity_search(query, k=10)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem'>
        <div style='font-size:1.2rem; font-weight:600; color:#ececec'>🧠 StudyMind</div>
        <div style='font-size:0.75rem; color:#666; margin-top:0.2rem'>
            Local AI · phi3:mini · FAISS
        </div>
    </div>
    """, unsafe_allow_html=True)

    new_files = st.file_uploader(
        "Upload PDFs", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if new_files:
        meta = load_metadata()
        changed = False
        for f in new_files:
            save_path = os.path.join(PDF_DIR, f.name)
            if f.name not in meta["files"]:
                with open(save_path, "wb") as out:
                    out.write(f.read())
                meta["files"].append(f.name)
                changed = True
        if changed:
            save_metadata(meta)
            st.cache_resource.clear()
        st.rerun()

    meta = load_metadata()
    saved_files = meta["files"]

    if saved_files:
        st.markdown("""
        <div style='font-size:0.75rem; color:#888; font-weight:500;
                    margin: 1rem 0 0.5rem; letter-spacing:0.5px'>
            📚 YOUR SOURCES
        </div>
        """, unsafe_allow_html=True)

        for filename in saved_files:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="pdf-card">
                    <span>
                        <span class="pdf-icon">📄</span>
                        <span class="pdf-name" title="{filename}">{filename}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{filename}", help=f"Remove {filename}"):
                    delete_pdf(filename)
                    st.rerun()

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

    if st.button("＋ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
    <div style='margin-top:1.5rem; color:#555; font-size:0.75rem; line-height:1.8'>
        <div style='color:#888; font-weight:500; margin-bottom:0.4rem'>How to use</div>
        1. Upload your PDF notes<br>
        2. Files are saved automatically ✓<br>
        3. Ask any question below
    </div>
    """, unsafe_allow_html=True)


# ─── Main Area ───────────────────────────────────────────────────────────────
meta = load_metadata()
saved_files = meta["files"]
pdf_paths = [
    os.path.join(PDF_DIR, f)
    for f in saved_files
    if os.path.exists(os.path.join(PDF_DIR, f))
]

if not pdf_paths:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🧠</div>
        <h2>What do you want to study today?</h2>
        <p>Upload your PDF notes or textbooks from the sidebar.<br>
        Your files are saved and will reload automatically next time.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Loading your documents..."):
        vectorstore, chunk_count, page_count = build_vectorstore(tuple(pdf_paths))

    st.markdown(f"""
    <div class="stat-row">
        <div class="chip">📄 <span>{len(pdf_paths)}</span> file(s)</div>
        <div class="chip">📃 <span>{page_count}</span> pages</div>
        <div class="chip">🧩 <span>{chunk_count}</span> chunks</div>
        <div class="chip">✅ <span>Ready</span></div>
    </div>
    """, unsafe_allow_html=True)

    llm = Ollama(model="phi3:mini")

    prompt = ChatPromptTemplate.from_template("""
You are StudyMind, a helpful and thorough study assistant.
Using ONLY the context provided below, give a detailed and well-structured answer.
Each chunk is tagged with its page number so you know the order of content in the PDF.

Rules:
- Never give one-line answers. Always elaborate with full explanation.
- Use bullet points or numbered lists when explaining steps or multiple points.
- Use **bold** for key terms or important concepts.
- If someone asks about the "last program/question", focus on the highest page number chunks.
- If someone asks about the "first program/question", focus on the lowest page number chunks.
- If the answer involves code, wrap it in a code block.
- If the context does not contain enough info, say so clearly.
- End with a one-line summary starting with "**In summary:**"

Context:
{context}

Question:
{input}

Detailed Answer:
""")

    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center; padding:3rem 0 1rem; color:#555; font-size:0.88rem'>
            Your conversation will appear here
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything about your study material...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                retrieved = smart_retrieve(vectorstore, query)

            # ✅ Include page numbers in context
            clean_context = "\n\n".join(
                f"[Page {doc.metadata.get('page_number','?')} of "
                f"{doc.metadata.get('total_pages','?')}]\n"
                f"{clean_text(doc.page_content)}"
                for doc in retrieved
            )

            for chunk in llm.stream(
                prompt.format(context=clean_context, input=query)
            ):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        st.session_state.chat_history.append({
            "role": "assistant", "content": full_response
        })