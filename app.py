import streamlit as st
import re
import os
import json
import shutil
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ─── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="StudyMind", page_icon="🧠", layout="wide")

# ─── Persistent Storage Paths ───────────────────────────────────────────────
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "studymind_data")
PDF_DIR     = os.path.join(STORAGE_DIR, "pdfs")
FAISS_DIR   = os.path.join(STORAGE_DIR, "faiss_index")
META_FILE   = os.path.join(STORAGE_DIR, "metadata.json")
CHATS_DIR   = os.path.join(STORAGE_DIR, "chats")

os.makedirs(PDF_DIR,   exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(CHATS_DIR, exist_ok=True)


# ─── Chat Persistence ────────────────────────────────────────────────────────
def load_all_chats():
    chats = []
    for fname in sorted(os.listdir(CHATS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(CHATS_DIR, fname), "r") as f:
                chats.append(json.load(f))
    return chats


def save_chat(chat):
    path = os.path.join(CHATS_DIR, f"{chat['id']}.json")
    with open(path, "w") as f:
        json.dump(chat, f)


def delete_chat(chat_id):
    path = os.path.join(CHATS_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        os.remove(path)


def new_chat():
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }


def group_chats_by_date(chats):
    now      = datetime.now()
    today    = now.date()
    yesterday = (now - timedelta(days=1)).date()
    week_ago  = now - timedelta(days=7)

    groups = {"Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []}
    for chat in chats:
        if not chat["messages"]:
            continue
        dt = datetime.fromisoformat(chat["created_at"])
        d  = dt.date()
        if d == today:
            groups["Today"].append(chat)
        elif d == yesterday:
            groups["Yesterday"].append(chat)
        elif dt >= week_ago:
            groups["Previous 7 Days"].append(chat)
        else:
            groups["Older"].append(chat)
    return groups


# ─── PDF Metadata ────────────────────────────────────────────────────────────
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


# ─── Session State Init ──────────────────────────────────────────────────────
if "current_chat" not in st.session_state:
    all_chats = load_all_chats()
    if all_chats:
        st.session_state.current_chat = all_chats[0]
    else:
        st.session_state.current_chat = new_chat()


# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #212121; --surface: #2f2f2f;
    --border: #4a4a4a; --text: #ececec; --muted: #8e8ea0;
    --accent: #ab68ff; --input-bg: #2f2f2f;
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
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stFileUploader"] {
    background: #2a2a2a !important;
    border: 1.5px dashed #444 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
.stButton > button {
    background: #2f2f2f !important; color: var(--text) !important;
    border: 1px solid #444 !important; border-radius: 8px !important;
    font-size: 0.85rem !important; transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #3a3a3a !important; border-color: var(--accent) !important;
}
.chat-group-label {
    font-size: 0.7rem; color: #555; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    padding: 0.8rem 0 0.3rem;
}
.pdf-card {
    background: #1e1e1e; border: 1px solid #2e2e2e;
    border-radius: 10px; padding: 0.5rem 0.7rem;
    margin-bottom: 0.3rem; transition: border-color 0.2s;
}
.pdf-card:hover { border-color: #444; }
.pdf-name {
    font-size: 0.78rem; color: #ccc;
    overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 140px; display: inline-block;
}
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
            page.metadata["page_number"]   = i + 1
            page.metadata["total_pages"]   = total
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
    if any(w in q for w in ["last", "final", "end", "last page",
                              "last program", "last question", "last exercise"]):
        all_docs = vectorstore.similarity_search(query, k=20)
        last_docs = [
            d for d in all_docs
            if d.metadata.get("is_last_page") or
               d.metadata.get("page_number", 0) >=
               d.metadata.get("total_pages", 999) - 2
        ]
        if last_docs:
            return last_docs[:6]

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

    return vectorstore.similarity_search(query, k=10)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style='padding: 0.5rem 0 0.5rem'>
            <div style='font-size:1.1rem; font-weight:600; color:#ececec'>🧠 StudyMind</div>
            <div style='font-size:0.7rem; color:#555; margin-top:0.1rem'>Groq · FAISS · LangChain</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("＋", help="New Chat"):
            chat = new_chat()
            save_chat(chat)
            st.session_state.current_chat = chat
            st.rerun()

    st.markdown("<hr style='border-color:#2a2a2a; margin:0.5rem 0'>", unsafe_allow_html=True)

    # ── Chat History ──
    all_chats = load_all_chats()
    groups = group_chats_by_date(all_chats)

    for group_name, chats in groups.items():
        if not chats:
            continue
        st.markdown(f"<div class='chat-group-label'>{group_name}</div>",
                    unsafe_allow_html=True)

        for chat in chats:
            title = chat.get("title", "New Chat")[:28]
            col_a, col_b = st.columns([5, 1])
            with col_a:
                is_active = chat["id"] == st.session_state.current_chat["id"]
                label = f"{'▶ ' if is_active else '💬 '}{title}"
                if st.button(label, key=f"chat_{chat['id']}"):
                    st.session_state.current_chat = chat
                    st.rerun()
            with col_b:
                if st.button("✕", key=f"delchat_{chat['id']}"):
                    delete_chat(chat["id"])
                    remaining = load_all_chats()
                    st.session_state.current_chat = remaining[0] if remaining else new_chat()
                    st.rerun()

    st.markdown("<hr style='border-color:#2a2a2a; margin:0.8rem 0'>", unsafe_allow_html=True)

    # ── PDF Sources ──
    st.markdown("""
    <div style='font-size:0.7rem; color:#555; font-weight:600;
                letter-spacing:0.8px; margin-bottom:0.5rem'>
        📚 SOURCES
    </div>
    """, unsafe_allow_html=True)

    new_files = st.file_uploader(
        "Upload PDFs", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if new_files:
        meta = load_metadata()
        for f in new_files:
            save_path = os.path.join(PDF_DIR, f.name)
            if f.name not in meta["files"]:
                with open(save_path, "wb") as out:
                    out.write(f.getvalue())
                meta["files"].append(f.name)
        save_metadata(meta)
        st.cache_resource.clear()
        st.rerun()

    meta = load_metadata()
    for filename in meta["files"]:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div class="pdf-card">
                <span>📄</span>
                <span class="pdf-name" title="{filename}">{filename}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✕", key=f"delpdf_{filename}"):
                delete_pdf(filename)
                st.rerun()


# ─── Main Area ───────────────────────────────────────────────────────────────
meta = load_metadata()
pdf_paths = [
    os.path.join(PDF_DIR, f)
    for f in meta["files"]
    if os.path.exists(os.path.join(PDF_DIR, f))
]

if not pdf_paths:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-icon">🧠</div>
        <h2>What do you want to study today?</h2>
        <p>Upload your PDF notes from the sidebar.<br>
        Your files and chats are saved automatically.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    with st.spinner("Loading documents..."):
        vectorstore, chunk_count, page_count = build_vectorstore(tuple(pdf_paths))

    st.markdown(f"""
    <div class="stat-row">
        <div class="chip">📄 <span>{len(pdf_paths)}</span> file(s)</div>
        <div class="chip">📃 <span>{page_count}</span> pages</div>
        <div class="chip">🧩 <span>{chunk_count}</span> chunks</div>
        <div class="chip">⚡ <span>Groq</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Groq LLM — fast cloud inference
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

    prompt = ChatPromptTemplate.from_template("""
You are StudyMind. Answer using ONLY the context below.
Be detailed but concise. Use bullet points for multiple items.
Use **bold** for key terms. Wrap code in code blocks.
If asked about "last program/question" use highest page number chunks.
If asked about "first program/question" use lowest page number chunks.
End with "**In summary:**"

Context:
{context}

Question:
{input}

Answer:
""")

    messages = st.session_state.current_chat.get("messages", [])

    if not messages:
        st.markdown("""
        <div style='text-align:center; padding:3rem 0 1rem; color:#555; font-size:0.88rem'>
            Start asking questions about your study material
        </div>
        """, unsafe_allow_html=True)

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything about your study material...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.current_chat["messages"].append({
            "role": "user", "content": query
        })

        # Auto-title from first question
        if len(st.session_state.current_chat["messages"]) == 1:
            st.session_state.current_chat["title"] = query[:40]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                retrieved = smart_retrieve(vectorstore, query)

            clean_context = "\n\n".join(
                f"[Page {doc.metadata.get('page_number','?')} of "
                f"{doc.metadata.get('total_pages','?')}]\n"
                f"{clean_text(doc.page_content)}"
                for doc in retrieved
            )

            # ✅ Groq streaming
            for chunk in llm.stream(
                prompt.format(context=clean_context, input=query)
            ):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        st.session_state.current_chat["messages"].append({
            "role": "assistant", "content": full_response
        })

        save_chat(st.session_state.current_chat)
        st.rerun()