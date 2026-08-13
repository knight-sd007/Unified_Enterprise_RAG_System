"""
Unified Enterprise RAG System — Streamlit Web Application.

A modern, production-grade Streamlit interface demonstrating Retrieval-Augmented Generation (RAG)
with unified AI provider abstraction (OpenAI, Google Gemini, NVIDIA NIM).
"""

import streamlit as st
import os
from config.settings import Config
from utils.security import verify_access_key, sanitize_error_message
from utils.logging import logger
from providers.openai_provider import OpenAIProvider
from providers.gemini_provider import GeminiProvider
from providers.nvidia_nim_provider import NvidiaNimProvider
from rag.loaders import DocumentLoader, Document
from rag.pipeline import RAGPipeline


# Page Configuration
st.set_page_config(
    page_title="Unified Enterprise RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []


def render_login():
    """Renders Security Authentication Gate."""
    st.markdown("<h2 style='text-align: center;'>🔐 Unified Enterprise RAG System</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Enter access key to unlock document intelligence interface.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            access_key = st.text_input("Access Key", type="password", placeholder="Enter APP_ACCESS_KEY")
            submit = st.form_submit_button("Authenticate & Access", use_container_width=True)

            if submit:
                expected = Config.get_app_access_key()
                if verify_access_key(access_key, expected):
                    st.session_state.authenticated = True
                    st.success("Authentication successful!")
                    st.rerun()
                else:
                    st.error("Invalid access key. Please check configuration.")


def get_active_provider(provider_choice: str):
    """Instantiates active provider object based on dropdown choice."""
    if provider_choice == "openai":
        return OpenAIProvider()
    elif provider_choice == "gemini":
        return GeminiProvider()
    elif provider_choice == "nvidia_nim":
        return NvidiaNimProvider()
    return OpenAIProvider()


def render_app():
    """Renders main enterprise RAG interface."""
    init_session_state()

    # Sidebar Controls
    with st.sidebar:
        st.title("⚙️ AI Pipeline Control")

        # Provider Selector
        provider_choice = st.selectbox(
            "Select AI Provider",
            options=["openai", "gemini", "nvidia_nim"],
            format_func=lambda x: {
                "openai": "OpenAI (GPT-4o / Embeddings)",
                "gemini": "Google Gemini (Gemini 2.5 / Embeddings)",
                "nvidia_nim": "NVIDIA NIM (Nemotron / E5)"
            }.get(x, x),
            index=0
        )

        provider = get_active_provider(provider_choice)
        status = provider.get_status()

        # Provider Status Card
        st.markdown("### Provider Status")
        st.markdown(f"**Status:** {status['status_icon']} {status['status_text']}")
        st.markdown(f"**Embedding Model:** `{status['embedding_model']}`")
        st.markdown(f"**Chat Model:** `{status['chat_model']}`")

        if not status["configured"]:
            st.warning(f"⚠️ Provider '{status['name']}' requires API credentials. Configure environment variables.")

        st.divider()

        # Vector Index Stats
        pipeline = st.session_state.rag_pipeline
        stats = pipeline.get_stats()
        st.markdown("### Vector Index Stats")
        st.markdown(f"**Record Count:** `{stats['count']}` chunks")
        st.markdown(f"**Embedding Dimension:** `{stats['dimension']}`")
        st.markdown(f"**Active Index Provider:** `{stats['provider_id']}`")

        if stats['count'] > 0:
            if st.button("🗑️ Clear Vector Index", use_container_width=True):
                pipeline.clear_index()
                st.session_state.ingested_files = []
                st.session_state.chat_history = []
                st.success("Vector index cleared.")
                st.rerun()

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # Main Workspace Area
    st.title("🔍 Unified Enterprise RAG System")
    st.caption("Vector Document Indexing & Semantic Question Answering Engine")

    tab1, tab2, tab3 = st.tabs(["📄 Document Ingestion", "💬 Semantic QA Search", "📊 System Architecture"])

    # TAB 1: DOCUMENT INGESTION
    with tab1:
        st.subheader("1. Ingest Documents into In-Memory Vector Index")
        st.markdown("Upload text or PDF documents to split into semantic chunks and build high-dimensional vector embeddings.")

        uploaded_files = st.file_uploader(
            "Upload Documents (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1500, value=500, step=50)
        with col2:
            chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=50, step=10)

        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please select at least one PDF or TXT file to ingest.")
            elif not provider.is_configured():
                st.error(f"Provider '{provider.name}' is not configured. Please supply API keys in `.env`.")
            else:
                documents = []
                for file in uploaded_files:
                    try:
                        content_bytes = file.read()
                        doc = DocumentLoader.load_from_file(content_bytes, file.name)
                        documents.append(doc)
                    except Exception as e:
                        st.error(f"Failed to read file '{file.name}': {sanitize_error_message(e)}")

                if documents:
                    with st.spinner(f"Ingesting {len(documents)} document(s) via '{provider.name}'..."):
                        # Reconfigure pipeline chunker if slider changed
                        pipeline.chunker.chunk_size = chunk_size
                        pipeline.chunker.chunk_overlap = chunk_overlap

                        res = pipeline.ingest_documents(documents, provider)

                    if res.get("status") == "success":
                        st.success(
                            f"Successfully ingested {res['document_count']} document(s) into {res['chunk_count']} vector chunks! "
                            f"(Vector Dim: {res['vector_dimension']}, Provider: {res['provider']})"
                        )
                        st.session_state.ingested_files = [f.name for f in uploaded_files]
                    else:
                        st.error(f"Ingestion failed: {res.get('message')}")

        if st.session_state.ingested_files:
            st.markdown("#### Currently Indexed Files")
            for fname in st.session_state.ingested_files:
                st.markdown(f"- 📄 `{fname}`")

    # TAB 2: SEMANTIC QA SEARCH
    with tab2:
        st.subheader("2. Semantic Question Answering")

        if pipeline.vector_store.count() == 0:
            st.info("💡 Vector index is empty. Please upload and index documents in the 'Document Ingestion' tab first.")
        else:
            top_k = st.slider("Top-K Retrieved Context Chunks", min_value=1, max_value=10, value=4)
            similarity_thresh = st.slider("Similarity Score Threshold", min_value=0.0, max_value=0.9, value=0.25, step=0.05)

            user_query = st.text_input("Ask a question about your indexed documents:", placeholder="e.g., What are the key terms in section 3?")

            if st.button("🔎 Submit Query", type="primary"):
                if not user_query.strip():
                    st.warning("Please enter a question.")
                elif not provider.is_configured():
                    st.error(f"Selected provider '{provider.name}' is not configured with valid API keys.")
                else:
                    with st.spinner(f"Retrieving context and generating answer using '{provider.name}'..."):
                        try:
                            result = pipeline.query(
                                question=user_query,
                                provider=provider,
                                top_k=top_k,
                                similarity_threshold=similarity_thresh
                            )

                            st.markdown("### Answer")
                            st.markdown(result["answer"])

                            st.markdown("---")
                            st.markdown("### 📚 Source Citation Back-References")

                            sources = result.get("sources", [])
                            if not sources:
                                st.caption("No context chunks met the similarity threshold.")
                            else:
                                for i, src in enumerate(sources, 1):
                                    meta = src.get("metadata", {})
                                    score = src.get("score", 0.0)
                                    filename = meta.get("filename", "Document")
                                    chunk_id = src.get("chunk_id", f"c{i}")

                                    with st.expander(f"Snippet [{i}] — {filename} (Similarity Score: {score:.4f})"):
                                        st.markdown(f"**Chunk ID:** `{chunk_id}`")
                                        st.markdown(f"```text\n{src['content']}\n```")

                        except Exception as e:
                            clean_err = sanitize_error_message(e)
                            st.error(f"Query Error: {clean_err}")

    # TAB 3: SYSTEM ARCHITECTURE
    with tab3:
        st.subheader("3. Technical Architecture Overview")
        st.markdown("""
        ### Core Engineering Design

        1. **Unified AI Provider Abstraction**:
           - Single active provider owns both embedding vector generation and chat completion.
           - Eliminates mismatches between embedding models and generation contexts.

        2. **In-Memory Vector Search Engine**:
           - Stores text chunks and high-dimensional embeddings directly in memory.
           - Uses **NumPy** matrix operations for exact Cosine Similarity calculations.

        3. **Provider Compatibility & Dimension Protection**:
           - Vector embedding spaces across AI providers are non-interchangeable.
           - Built-in validation checks vector dimensions (e.g. OpenAI `1536`, Gemini `768`) and prevents cross-provider distance evaluation errors.

        4. **Source Attribution & Citation Integrity**:
           - Preserves original file metadata, sub-chunk indices, and similarity scores throughout the RAG pipeline.
           - Prompts the LLM to reference sources explicitly.
        """)


def main():
    """Application entrypoint with login check."""
    init_session_state()
    if not st.session_state.authenticated:
        render_login()
    else:
        render_app()


if __name__ == "__main__":
    main()
