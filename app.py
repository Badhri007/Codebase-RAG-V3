"""Enhanced Streamlit UI for Codebase RAG with all features."""
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enhanced Codebase RAG", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    h1 { text-align: center; color: #4F46E5; }
    .stChatMessage { border-radius: 10px; }
    .metric-card {
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4F46E5;
    }
</style>
""", unsafe_allow_html=True)


def render_mermaid(code: str, height: int = 500):
    """Render Mermaid diagram."""
    html = f"""
    <html><head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    </head><body style="background:#f8fafc;padding:20px;">
    <pre class="mermaid">{code}</pre>
    <script>mermaid.initialize({{startOnLoad:true,theme:'default'}});</script>
    </body></html>
    """
    components.html(html, height=height, scrolling=True)


LLM_OPTIONS = {
    "Claude Sonnet 4": ("claude", "claude-sonnet-4-20250514"),
    "GPT-4o": ("openai", "gpt-4o"),
    "GPT-4o Mini": ("openai", "gpt-4o-mini"),
    "Gemini Flash": ("gemini", "gemini-2.0-flash-exp"),
    "DeepSeek V3": ("deepseek", "deepseek-chat"),
}

TRAVERSAL_STRATEGIES = {
    "Smart (Balanced)": "smart",
    "Deep (Follow dependencies)": "deep",
    "Wide (Explore breadth)": "wide",
    "Dependency First": "dependency"
}

# Session state initialization
defaults = {
    'rag': None,
    'messages': [],
    'llm': "Claude Sonnet 4",
    'traversal_strategy': "smart",
    'initialized': False
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


def init_rag(provider, model, use_neo4j=True, use_contextual=True, use_llm_context=False):
    """Initialize RAG pipeline with configuration."""
    try:
        from rag import RAGPipeline
        from hierarchical_chunking import ChunkingStrategy

        # Create chunking strategy
        chunking_strategy = ChunkingStrategy(
            use_contextual_retrieval=use_contextual,
            use_llm_for_context=use_llm_context,
            llm_provider=provider,
            max_chunk_size=2000,
            create_file_summary=True,
            create_class_summary=True,
        )

        st.session_state.rag = RAGPipeline(
            llm_provider=provider,
            llm_model=model,
            use_neo4j=use_neo4j,
            use_contextual_retrieval=use_contextual,
            use_llm_context=use_llm_context,
            chunking_strategy=chunking_strategy
        )

        st.session_state.init_config = {
            'use_neo4j': use_neo4j,
            'use_contextual': use_contextual,
            'use_llm_context': use_llm_context
        }
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return False


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    if not st.session_state.initialized:
        st.subheader("Initial Setup")


        use_neo4j = st.checkbox("Use Neo4j graph database", value=True)
        use_contextual = st.checkbox("Use contextual retrieval", value=True)
        use_llm_context = st.checkbox("Use LLM for context (expensive)", value=False)

        if st.button("🚀 Initialize System", type="primary"):
            provider, model = LLM_OPTIONS["Claude Sonnet 4"]
            with st.spinner("Initializing enhanced RAG system..."):
                if init_rag(provider, model, use_neo4j, use_contextual, use_llm_context):
                    st.success("✅ System ready!")
                    st.rerun()

    else:
        # Show current config
        with st.expander("📊 Current Configuration", expanded=False):
            config = st.session_state.init_config or {}
            st.caption(f"**Graph:** {'Neo4j' if config.get('use_neo4j') else 'In-memory'}")
            st.caption(f"**Contextual:** {'✓' if config.get('use_contextual') else '✗'}")
            st.caption(f"**LLM Context:** {'✓' if config.get('use_llm_context') else '✗'}")

        st.divider()

        # LLM selection
        st.subheader("🤖 LLM Provider")
        llm_choice = st.selectbox(
            "Select LLM",
            list(LLM_OPTIONS.keys()),
            index=list(LLM_OPTIONS.keys()).index(st.session_state.llm),
            label_visibility="collapsed"
        )
        provider, model = LLM_OPTIONS[llm_choice]

        # Switch LLM
        if llm_choice != st.session_state.llm:
            if st.button("🔄 Switch LLM"):
                if st.session_state.rag:
                    st.session_state.rag.switch_llm(provider, model)
                    st.session_state.llm = llm_choice
                    st.rerun()

        st.divider()

        n_chunks = st.slider("Chunks to retrieve", 10, 40, 25, 5)

        st.divider()

        # Repositories
        st.subheader("📚 Repositories")

        if st.session_state.rag:
            repos = st.session_state.rag.list_repos()

            if repos:
                for repo in repos:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        current = st.session_state.rag.current_repo == repo
                        label = f"{'✓ ' if current else ''}{repo[:30]}"
                        if st.button(
                            label,
                            key=f"load_{repo}",
                            disabled=current,
                            use_container_width=True
                        ):
                            st.session_state.rag.load(repo)
                            st.session_state.messages = []
                            st.rerun()
                    with col2:
                        if st.button("🗑️", key=f"del_{repo}"):
                            st.session_state.rag.delete_repo(repo)
                            st.rerun()
            else:
                st.info("No repositories indexed yet")

        # Advanced
        with st.expander("⚙️ Advanced"):
            diagram_height = st.slider("Diagram height", 300, 800, 500)
            show_retrieval = st.checkbox("Show retrieval details", value=False)


# Main content
st.title("🔍 Codebase RAG")

if not st.session_state.initialized:
    st.info("👈 Configure and initialize the system in the sidebar to get started")
    st.stop()

# Index section
st.subheader("Index Repository")
col1, col2, col3 = st.columns([5, 1, 1])
with col1:
    url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed"
    )
with col2:
    index_btn = st.button("Index", type="primary")
with col3:
    force = st.checkbox("Force", help="Force re-index even if exists")

if index_btn and url and st.session_state.rag:
    with st.spinner("Indexing repository... (this may take several minutes)"):
        try:
            st.session_state.rag.index(url, force)
            st.session_state.messages = []
            st.success("✅ Repository indexed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Indexing failed: {str(e)}")

st.divider()

# Chat interface
if not st.session_state.rag.current_repo:
    st.info("👆 Index or select a repository from the sidebar to start asking questions")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("diagram"):
                with st.expander("📊 Architecture Diagram", expanded=True):
                    render_mermaid(msg["diagram"], diagram_height)

    # Chat input
    if prompt := st.chat_input("Ask about the codebase..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase..."):
                try:
                    # Call enhanced ask with retrieval strategy
                    result = st.session_state.rag.ask(
                        prompt,
                        n_chunks=n_chunks,
                        explain=show_retrieval
                    )

                    st.markdown(result['answer'])

                    if result.get('diagram'):
                        with st.expander("📊 Architecture Diagram", expanded=True):
                            render_mermaid(result['diagram'], diagram_height)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "diagram": result.get('diagram')
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())

    # Clear chat button
    if st.session_state.messages:
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.rag.clear_history()
                st.rerun()
        with col2:
            if st.button("📊 Show Stats"):
                stats = st.session_state.rag.get_stats()
                with st.expander("Repository Statistics", expanded=True):
                    for key, value in stats.items():
                        st.caption(f"**{key}:** {value}")
