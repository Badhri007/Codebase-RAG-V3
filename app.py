"""Streamlit UI for Codebase RAG with dynamic LLM & Embedding provider selection."""

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from indexer import index_repository
from query import answer_query
import re

load_dotenv()

st.set_page_config(
    page_title="Codebase RAG",
    page_icon="🔍",
    layout="wide"
)

# -------------------- PROVIDER OPTIONS -------------------- #

LLM_OPTIONS = {
    "Claude Sonnet 4":  ("claude",    "claude-sonnet-4-20250514"),
    "Claude Haiku 3.5": ("claude",    "claude-haiku-3-5-20241022"),
    "GPT-4o":           ("openai",    "gpt-4o"),
    "GPT-4o Mini":      ("openai",    "gpt-4o-mini"),
    "Gemini Flash":     ("gemini",    "gemini-2.0-flash-exp"),
    "DeepSeek V3":      ("deepseek",  "deepseek-chat"),
}

HF_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1",
]

# -------------------- SESSION DEFAULTS -------------------- #

defaults = {
    "messages":         [],
    "current_repo":     None,
    "indexing":         False,
    "llm":              "Claude Sonnet 4",
    "emb_provider":     "huggingface",   # "huggingface" | "jina"
    "emb_model":        "all-MiniLM-L6-v2",
    "jina_base_url":    "http://localhost:8080",
    "jina_api_key":     "",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- HELPERS -------------------- #

def render_mermaid(code: str, height: int = 500):
    html = f"""
    <html><head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head>
    <body style="background:#f8fafc;padding:20px;">
        <pre class="mermaid">{code}</pre>
        <script>mermaid.initialize({{startOnLoad:true,theme:'default'}});</script>
    </body></html>
    """
    components.html(html, height=height, scrolling=True)


def display_message(content: str):
    parts = re.split(r'```mermaid\n(.*?)\n```', content, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            with st.expander("📊 View Diagram", expanded=True):
                render_mermaid(part, height=600)


def get_embedding_kwargs() -> dict:
    """Return kwargs to pass to VectorDB / index_repository."""
    if st.session_state.emb_provider == "jina":
        return {
            "embedding_provider": "jina",
            "model":              st.session_state.emb_model,
            "base_url":          st.session_state.jina_base_url,
            **({"api_key": st.session_state.jina_api_key} if st.session_state.jina_api_key else {}),
        }
    return {
        "embedding_provider": "huggingface",
        "model":              st.session_state.emb_model,
    }

# -------------------- STYLES -------------------- #

st.markdown("""
<style>
    h1 { text-align: center; color: #4F46E5; }
    .stChatMessage { border-radius: 10px; }
    .provider-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #EEF2FF;
        color: #4F46E5;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR -------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    # -------- LLM Provider -------- #
    st.subheader("🤖 LLM Provider")
    llm_choice = st.selectbox(
        "LLM Model",
        list(LLM_OPTIONS.keys()),
        index=list(LLM_OPTIONS.keys()).index(st.session_state.llm),
        label_visibility="collapsed",
        key="llm_select",
    )
    if llm_choice != st.session_state.llm:
        st.session_state.llm = llm_choice
        st.rerun()

    provider, model_id = LLM_OPTIONS[st.session_state.llm]
    st.caption(f"Provider: `{provider}` · Model: `{model_id}`")

    st.divider()

    # -------- Embedding Provider -------- #
    st.subheader("🧬 Embedding Provider")

    if st.session_state.current_repo:
        st.warning("Changing embeddings requires **re-indexing** (use Force).", icon="⚠️")

    emb_provider = st.radio(
        "Provider",
        ["huggingface", "jina"],
        index=["huggingface", "jina"].index(st.session_state.emb_provider),
        horizontal=True,
        label_visibility="collapsed",
    )
    if emb_provider != st.session_state.emb_provider:
        st.session_state.emb_provider = emb_provider
        st.rerun()

    if st.session_state.emb_provider == "huggingface":
        emb_model = st.selectbox("HF Model", HF_MODELS,
                                 index=HF_MODELS.index(st.session_state.emb_model)
                                       if st.session_state.emb_model in HF_MODELS else 0)
        if emb_model != st.session_state.emb_model:
            st.session_state.emb_model = emb_model
            st.rerun()
        st.caption(f"`{emb_model}`")

    else:  # jina
        emb_model   = st.text_input("Model name",   value=st.session_state.emb_model,
                                     placeholder="jina-embeddings-v3")
        jina_url    = st.text_input("Base URL",      value=st.session_state.jina_base_url,
                                     placeholder="http://localhost:8080")
        jina_key    = st.text_input("API key (opt)", value=st.session_state.jina_api_key,
                                     type="password", placeholder="leave blank if self-hosted")
        if (emb_model   != st.session_state.emb_model or
            jina_url    != st.session_state.jina_base_url or
            jina_key    != st.session_state.jina_api_key):
            st.session_state.emb_model     = emb_model
            st.session_state.jina_base_url = jina_url
            st.session_state.jina_api_key  = jina_key
            st.rerun()

    st.divider()

    # -------- Indexed Repos -------- #
    st.subheader("📚 Indexed Repositories")
    try:
        from vectordb import VectorDB
        from graph import Graph

        vector_db = VectorDB()
        graph = Graph()
        indexed_repos = vector_db.list_repos()

        if indexed_repos:
            for repo in indexed_repos:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        label = repo.split("__")[1] if "__" in repo else repo
                        if st.button(f"📦 {label}", key=f"select_{repo}", use_container_width=True):
                            st.session_state.current_repo = repo
                            st.session_state.messages = []
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"delete_{repo}"):
                            vector_db.delete_repo(repo)
                            graph.clear_repo(repo)
                            if st.session_state.current_repo == repo:
                                st.session_state.current_repo = None
                                st.session_state.messages = []
                            st.success(f"Deleted {repo}")
                            st.rerun()

                    if st.session_state.current_repo == repo:
                        st.markdown("**✓ Active**")
        else:
            st.info("No repositories indexed yet")

        graph.close()

    except Exception as e:
        st.error(f"Error loading repos: {e}")

    if st.session_state.current_repo:
        st.success(f"📦 Active: {st.session_state.current_repo}")

# -------------------- MAIN -------------------- #

st.title("🔍 Codebase RAG")
st.caption("Direct, accurate code understanding with Neo4j graph")

# Active provider summary
llm_prov, llm_mod = LLM_OPTIONS[st.session_state.llm]
emb_mod = st.session_state.emb_model
st.caption(
    f"🤖 **LLM:** {st.session_state.llm} &nbsp;|&nbsp; "
    f"🧬 **Embeddings:** `{emb_mod}`"
)

st.divider()

# -------------------- INDEX SECTION -------------------- #

st.subheader("📦 Index Repository")

col1, col2, col3 = st.columns([5, 1, 1])

with col1:
    url = st.text_input(
        "GitHub Repository URL or Local Path",
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
        disabled=st.session_state.indexing,
    )

with col2:
    index_btn = st.button("Index", type="primary", disabled=st.session_state.indexing)

with col3:
    force = st.checkbox("Force")

if index_btn and url:
    st.session_state.indexing = True

    try:
        with st.status("🔄 Indexing repository...") as status:
            if url.startswith("http"):
                repo_name = (
                    url.rstrip("/").split("/")[-2]
                    + "__"
                    + url.rstrip("/").split("/")[-1]
                ).lower()
            else:
                import os
                repo_name = os.path.basename(url.rstrip("/")).lower()

            st.write("📥 Cloning/reading repository...")
            st.write("📖 Parsing all files...")
            st.write("🔗 Building graph with accurate resolution...")
            st.write("💾 Storing in databases...")
            st.write(f"🧬 Using embedding model: `{emb_mod}`")

            emb_kwargs = get_embedding_kwargs()
            index_repository(url, force, **emb_kwargs)

            status.update(label="✅ Indexing complete!", state="complete")
            st.session_state.current_repo = repo_name
            st.session_state.messages = []

        st.success(f"✅ {repo_name} indexed successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Indexing failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

    finally:
        st.session_state.indexing = False

st.divider()

# -------------------- CHAT -------------------- #

if not st.session_state.current_repo:
    st.info("👆 Index a repository to start asking questions")
    st.stop()

st.subheader(f"💬 Ask Questions — {st.session_state.current_repo}")

# -------- Display History -------- #

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        display_message(msg["content"])

# -------- Chat Input -------- #

if prompt := st.chat_input("Ask about the codebase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("🤖 Processing...") as status:
                st.write("🔍 Searching vector database...")
                st.write("🔗 Expanding through graph...")
                st.write(f"🤖 Generating answer with {st.session_state.llm}...")

                answer = answer_query(
                    prompt,
                    st.session_state.current_repo,
                    llm_provider=llm_prov,
                    llm_model=llm_mod,
                    **get_embedding_kwargs(),
                )

                status.update(label="✅ Answer ready!", state="complete")

            display_message(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

# -------- Clear Chat -------- #

if st.session_state.messages:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()