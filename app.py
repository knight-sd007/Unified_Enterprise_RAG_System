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

# Custom Enterprise CSS for visual hierarchy, contrast, and responsive layout
ENTERPRISE_CUSTOM_CSS = """
<style>
/* Enterprise Theme Adjustments */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1200px;
}

/* Header & Banner Styling */
.app-header {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.05) 0%, rgba(30, 41, 59, 0.08) 100%);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}

.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}

.app-header p {
    font-size: 0.95rem;
    margin-bottom: 0;
    opacity: 0.85;
}

/* Status & Tag Badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 6px;
    margin-right: 0.4rem;
    margin-bottom: 0.2rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-blue {
    background-color: rgba(37, 99, 235, 0.12);
    color: #2563eb;
    border: 1px solid rgba(37, 99, 235, 0.25);
}

.badge-green {
    background-color: rgba(22, 163, 74, 0.12);
    color: #16a34a;
    border: 1px solid rgba(22, 163, 74, 0.25);
}

.badge-amber {
    background-color: rgba(217, 119, 6, 0.12);
    color: #d97706;
    border: 1px solid rgba(217, 119, 6, 0.25);
}

.badge-slate {
    background-color: rgba(100, 116, 139, 0.12);
    color: #64748b;
    border: 1px solid rgba(100, 116, 139, 0.25);
}

/* Prompt Trust Boundary Banner */
.trust-boundary-card {
    background-color: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-top: 1.25rem;
    margin-bottom: 1.25rem;
}

.trust-boundary-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #d97706;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.2rem;
}

.trust-boundary-desc {
    font-size: 0.82rem;
    margin-bottom: 0;
    opacity: 0.9;
    line-height: 1.4;
}

/* Elevated Answer Container */
.answer-container {
    background-color: rgba(30, 41, 59, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* Card Container for Features/Stats */
.feature-card {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 10px;
    padding: 1.1rem;
    margin-bottom: 1rem;
    height: 100%;
}

.feature-card h4 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.feature-card p {
    font-size: 0.85rem;
    opacity: 0.85;
    margin-bottom: 0;
    line-height: 1.45;
}

/* Login Card Container */
.login-box {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 2rem;
    background: rgba(30, 41, 59, 0.02);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}

/* Metric chip */
.metric-chip {
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.15);
    margin-bottom: 0.5rem;
}
.metric-chip-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    opacity: 0.7;
    margin-bottom: 0.1rem;
}
.metric-chip-value {
    font-size: 0.95rem;
    font-weight: 600;
    font-family: monospace;
}
</style>
"""


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

    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None

    if "last_ingest_result" not in st.session_state:
        st.session_state.last_ingest_result = None


def render_login():
    """Renders Accessible Security Authentication Gate."""
    st.markdown(ENTERPRISE_CUSTOM_CSS, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-box">
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔐</div>
                    <h2 style="margin: 0; font-size: 1.5rem; font-weight: 700;">Enterprise Access Gate</h2>
                    <p style="font-size: 0.9rem; margin-top: 0.4rem; opacity: 0.8;">
                        Unified Enterprise RAG System &mdash; Restricted Interface
                    </p>
                </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("login_form"):
            access_key = st.text_input(
                "Access Key",
                type="password",
                placeholder="Enter APP_ACCESS_KEY",
                help="Enter the configured enterprise access key to unlock document intelligence capabilities."
            )
            submit = st.form_submit_button("Authenticate & Access Interface", type="primary", use_container_width=True)

            if submit:
                expected = Config.get_app_access_key()
                if verify_access_key(access_key, expected):
                    st.session_state.authenticated = True
                    st.success("Authentication successful. Redirecting to workspace...")
                    st.rerun()
                else:
                    st.error("Invalid access key. Verification failed.")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 1rem; font-size: 0.78rem; opacity: 0.65;">
                Protected by constant-time string comparison & session token isolation.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def get_active_provider(provider_choice: str):
    """Instantiates active provider object based on dropdown choice."""
    if provider_choice == "openai":
        return OpenAIProvider()
    elif provider_choice == "gemini":
        return GeminiProvider()
    elif provider_choice == "nvidia_nim":
        return NvidiaNimProvider()
    return OpenAIProvider()


def render_sidebar(provider):
    """Renders Sidebar Controls, Provider Diagnostics, and Index Stats."""
    with st.sidebar:
        st.markdown("### ⚙️ Pipeline Control")
        st.caption("Active Provider & Vector Storage")

        # Provider Selector
        provider_choice = st.selectbox(
            "Active AI Provider",
            options=["openai", "gemini", "nvidia_nim"],
            format_func=lambda x: {
                "openai": "OpenAI (GPT-4o / text-embed-3)",
                "gemini": "Google Gemini (Gemini 2.5 / text-embed-004)",
                "nvidia_nim": "NVIDIA NIM (Nemotron / E5-v5)"
            }.get(x, x),
            index=0,
            help="Select the AI provider that owns both vector embedding generation and chat completions."
        )

        # Update provider if choice changed
        provider = get_active_provider(provider_choice)
        status = provider.get_status()

        # Provider Status Section
        st.markdown("---")
        st.markdown("#### Provider Status")

        status_class = "badge-green" if status["configured"] else "badge-amber"
        status_label = "Configured & Ready" if status["configured"] else "Credentials Required"

        st.markdown(
            f"""
            <div style="margin-bottom: 0.75rem;">
                <span class="badge {status_class}">{status['status_icon']} {status_label}</span>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Embedding Model</div>
                <div class="metric-chip-value">{status['embedding_model']}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-label">Chat Model</div>
                <div class="metric-chip-value">{status['chat_model']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not status["configured"]:
            st.warning(f"⚠️ Provider '{status['name']}' requires API credentials. Configure environment variables in `.env`.")

        st.markdown("---")

        # Vector Storage & Index Stats
        pipeline = st.session_state.rag_pipeline
        stats = pipeline.get_stats(provider.provider_id)
        spec = Config.get_provider_spec(provider.provider_id)

        st.markdown("#### Vector Storage & Index")
        store_type = stats.get('store_type', 'Vector Store')
        backend_badge = "badge-blue" if "Qdrant" in store_type else "badge-slate"

        st.markdown(
            f"""
            <div style="margin-bottom: 0.75rem;">
                <span class="badge {backend_badge}">Storage: {store_type}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Chunks", stats['count'])
        with col_b:
            dim_val = str(stats['dimension']) if stats['dimension'] else "N/A"
            st.metric("Vector Dim", dim_val)

        if spec:
            st.markdown(
                f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Target Collection</div>
                    <div class="metric-chip-value" style="font-size: 0.8rem;">{spec.collection_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Provider mismatch warning in sidebar
        active_p = pipeline.vector_store.get_active_provider_id()
        if active_p is not None and active_p != provider.provider_id:
            st.error(
                f"⚠️ **Vector Space Mismatch**\n\n"
                f"Active index was built with **{active_p}** vectors. "
                f"Selected provider is **{provider.name}**. "
                f"Switch back to **{active_p}** or clear and re-index documents."
            )

        st.markdown("---")

        # Actions section
        st.markdown("#### Maintenance Actions")
        if stats['count'] > 0 or active_p is not None:
            if st.button("🗑️ Clear Vector Index", use_container_width=True, help="Wipes all in-memory or session index chunks."):
                pipeline.clear_index()
                st.session_state.ingested_files = []
                st.session_state.chat_history = []
                st.session_state.last_query_result = None
                st.session_state.last_ingest_result = None
                st.success("Vector index session cleared.")
                st.rerun()

        if st.button("🚪 Logout Session", use_container_width=True, help="Locks the interface and terminates current session."):
            st.session_state.authenticated = False
            st.rerun()

    return provider


def render_app():
    """Renders main enterprise RAG interface with modern UX."""
    init_session_state()
    st.markdown(ENTERPRISE_CUSTOM_CSS, unsafe_allow_html=True)

    # Resolve active provider through sidebar
    provider_default_choice = Config.get_ai_provider()
    provider = get_active_provider(provider_default_choice)
    provider = render_sidebar(provider)

    pipeline = st.session_state.rag_pipeline
    active_p = pipeline.vector_store.get_active_provider_id()
    spec = Config.get_provider_spec(provider.provider_id)
    stats = pipeline.get_stats(provider.provider_id)

    # Enterprise Header
    st.markdown(
        """
        <div class="app-header">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <h1 style="margin: 0; color: inherit;">🔍 Unified Enterprise RAG System</h1>
                    <p style="margin-top: 0.3rem;">
                        Production-grade document ingestion, high-dimensional vector search, and citation-backed question answering.
                    </p>
                </div>
                <div>
                    <span class="badge badge-blue">Enterprise Edition</span>
                    <span class="badge badge-green">Production Gate Active</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top-level runtime compatibility notification
    if active_p is not None and active_p != provider.provider_id:
        st.warning(
            f"⚠️ **Active Index Compatibility Notice**: The active vector index currently holds embeddings generated with "
            f"**{active_p}** ({stats.get('dimension', 'N/A')} dims). You have selected **{provider.name}**. "
            f"Cross-provider vector querying is rejected to prevent dimensional and semantic corruption. "
            f"Please switch the active provider to **{active_p}** or clear the index to ingest new documents with **{provider.name}**."
        )

    tab1, tab2, tab3 = st.tabs([
        "💬 Semantic QA Search",
        "📄 Document Ingestion",
        "📊 System Architecture"
    ])

    # TAB 1: SEMANTIC QA SEARCH (Primary Interaction)
    with tab1:
        st.markdown("### Semantic Question Answering")
        st.caption("Ask natural-language questions grounded strictly in your indexed enterprise documents.")

        # Query Formulation Container
        with st.container():
            user_query = st.text_input(
                "Search Query / Question",
                placeholder="e.g., What are the critical compliance obligations outlined in section 4?",
                help="Type your question. The system will retrieve relevant semantic chunks and generate a grounded answer."
            )

            # Collapsible Advanced Retrieval Parameters to keep UX clean
            with st.expander("⚙️ Retrieval Parameters & Similarity Thresholds", expanded=False):
                col_k, col_thresh = st.columns(2)
                with col_k:
                    top_k = st.slider(
                        "Top-K Retrieved Context Chunks",
                        min_value=1,
                        max_value=10,
                        value=4,
                        help="Number of most similar semantic chunks to pass as grounded context."
                    )
                with col_thresh:
                    similarity_thresh = st.slider(
                        "Similarity Score Cutoff Threshold",
                        min_value=0.0,
                        max_value=0.9,
                        value=0.25,
                        step=0.05,
                        help="Minimum cosine similarity score required for a chunk to be included in the generation context."
                    )

            col_sub, col_info = st.columns([1, 3])
            with col_sub:
                submit_query = st.button("🔎 Submit Query", type="primary", use_container_width=True)
            with col_info:
                if stats['count'] == 0:
                    st.caption("ℹ️ Vector index is currently empty. Ingest documents in the **Document Ingestion** tab to query.")
                else:
                    st.caption(f"Ready to search across **{stats['count']} indexed chunks** via **{provider.name}**.")

        # Execute Query Logic
        if submit_query:
            if not user_query.strip():
                st.warning("Please enter a question before submitting.")
            elif not provider.is_configured():
                st.error(f"Selected provider '{provider.name}' is not configured with valid API keys.")
            elif active_p is not None and active_p != provider.provider_id:
                st.error(
                    f"Cannot execute query: Provider mismatch between index ('{active_p}') "
                    f"and query provider ('{provider.name}')."
                )
            else:
                with st.spinner(f"Retrieving context & generating answer via '{provider.name}'..."):
                    try:
                        result = pipeline.query(
                            question=user_query,
                            provider=provider,
                            top_k=top_k,
                            similarity_threshold=similarity_thresh
                        )
                        st.session_state.last_query_result = {
                            "query": user_query,
                            "result": result
                        }
                    except Exception as e:
                        clean_err = sanitize_error_message(e)
                        st.error(f"Query Execution Error: {clean_err}")

        # Render Active Query Result
        if st.session_state.last_query_result:
            last_q = st.session_state.last_query_result["query"]
            res = st.session_state.last_query_result["result"]

            st.markdown("---")
            st.markdown(f"#### Generated Answer")
            st.caption(f"Query: *\"{last_q}\"* &bull; Provider: **{res.get('provider', provider.name)}** &bull; Model: `{res.get('chat_model', provider.get_chat_model_name())}`")

            st.markdown(
                f"""
                <div class="answer-container">
                    {res["answer"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Prompt Trust Boundary Banner & Citations
            st.markdown(
                """
                <div class="trust-boundary-card">
                    <div class="trust-boundary-title">🛡️ Prompt Trust Boundary & Source Citation Integrity</div>
                    <div class="trust-boundary-desc">
                        Retrieved document snippets below are treated as <strong>untrusted passive reference data</strong>.
                        They provide factual grounding and exact similarity scores but are isolated from system instructions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            sources = res.get("sources", [])
            if not sources:
                st.info("No context chunks met the similarity threshold for this query.")
            else:
                st.markdown(f"**Retrieved Sources ({len(sources)} matching chunks)**")
                for i, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    score = src.get("score", 0.0)
                    filename = meta.get("filename", "Document")
                    chunk_id = src.get("chunk_id", f"c{i}")

                    with st.expander(f"Reference [{i}] &mdash; {filename} (Similarity: {score:.4f})", expanded=(i == 1)):
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span class="badge badge-blue">File: {filename}</span>
                                <span class="badge badge-green">Score: {score:.4f}</span>
                                <span class="badge badge-slate">Chunk ID: {chunk_id}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(f"```text\n{src['content']}\n```")

        elif stats['count'] == 0:
            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="feature-card" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📂</div>
                    <h4 style="margin-bottom: 0.4rem;">No Documents Currently Indexed</h4>
                    <p style="margin-bottom: 1rem;">
                        Upload your PDF or TXT documents in the <strong>Document Ingestion</strong> tab to build vector embeddings and enable semantic search.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # TAB 2: DOCUMENT INGESTION
    with tab2:
        st.markdown("### Document Ingestion & Vector Indexing")
        st.caption("Upload enterprise documents, configure sliding-window text chunking, and generate vector embeddings.")

        col_up, col_settings = st.columns([3, 2])

        with col_up:
            st.markdown("#### 1. Select Documents")
            uploaded_files = st.file_uploader(
                "Upload Documents (PDF or TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Select one or more PDF or TXT files to parse, chunk, and embed."
            )

        with col_settings:
            st.markdown("#### 2. Chunking Parameters")
            chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=200,
                max_value=1500,
                value=500,
                step=50,
                help="Length of each semantic slice in characters. Smaller chunks preserve precise context."
            )
            chunk_overlap = st.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=300,
                value=50,
                step=10,
                help="Character overlap between consecutive chunks to prevent contextual cutoff at boundaries."
            )

        st.markdown("#### 3. Build & Ingest Vector Embeddings")
        target_col_name = spec.collection_name if spec else "In-Memory Vector Store"
        target_dim = spec.dimension if spec else "Dynamic"

        st.markdown(
            f"""
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem;">
                <span class="badge badge-blue">Target Provider: {provider.name}</span>
                <span class="badge badge-green">Target Collection: {target_col_name}</span>
                <span class="badge badge-slate">Dimension: {target_dim}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🚀 Process & Ingest Documents", type="primary", use_container_width=True):
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
                        clean_err = sanitize_error_message(e)
                        st.error(f"Failed to parse file '{file.name}': {clean_err}")

                if documents:
                    with st.spinner(f"Chunking, embedding, and indexing {len(documents)} document(s) via '{provider.name}'..."):
                        # Reconfigure pipeline chunker if slider changed
                        pipeline.chunker.chunk_size = chunk_size
                        pipeline.chunker.chunk_overlap = chunk_overlap

                        res = pipeline.ingest_documents(documents, provider)
                        st.session_state.last_ingest_result = res

                    if res.get("status") == "success":
                        st.success(
                            f"✅ Ingestion Complete: Successfully indexed {res['document_count']} document(s) "
                            f"into {res['chunk_count']} vector chunks! (Vector Dim: {res['vector_dimension']}, "
                            f"Collection: `{res.get('collection_name', 'In-Memory')}`)"
                        )
                        st.session_state.ingested_files = list(set(st.session_state.ingested_files + [f.name for f in uploaded_files]))
                    else:
                        st.error(f"Ingestion failed: {res.get('message')}")

        st.markdown("---")
        st.markdown("#### Indexed Document Inventory")
        if st.session_state.ingested_files:
            for fname in st.session_state.ingested_files:
                st.markdown(
                    f"""
                    <div style="display: inline-flex; align-items: center; padding: 0.35rem 0.75rem; background: rgba(148, 163, 184, 0.08); border-radius: 6px; margin: 0.2rem 0.4rem 0.2rem 0; border: 1px solid rgba(148, 163, 184, 0.15);">
                        <span style="margin-right: 0.4rem;">📄</span>
                        <span style="font-size: 0.88rem; font-weight: 500;">{fname}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.caption("No files indexed in the current session. Upload documents above to populate the index.")

    # TAB 3: SYSTEM ARCHITECTURE
    with tab3:
        st.markdown("### Technical Architecture & Governance")
        st.caption("Overview of the core engineering pillars powering the Unified Enterprise RAG System.")

        col_arch1, col_arch2 = st.columns(2)

        with col_arch1:
            st.markdown(
                """
                <div class="feature-card">
                    <h4>1. Unified Provider Abstraction</h4>
                    <p>A single active provider (Google Gemini, NVIDIA NIM, or OpenAI) exclusively owns both vector embedding generation and chat completions. Eliminates cross-model embedding skew and context token incompatibility.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="feature-card">
                    <h4>2. Vector Space Isolation & Dimension Guard</h4>
                    <p>Vector spaces across providers are strictly non-interchangeable. Enforces deterministic collection routing (768-dim Gemini, 1024-dim NVIDIA, 1536-dim OpenAI) and blocks cross-provider query pollution.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="feature-card">
                    <h4>3. Deterministic Idempotent Ingestion</h4>
                    <p>Uses deterministic UUIDv5 identifiers generated from chunk payload and metadata hashes. Re-indexing existing documents updates records idempotently without duplicate bloat or ghost vectors.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_arch2:
            st.markdown(
                """
                <div class="feature-card">
                    <h4>4. Prompt Trust Boundary Hardening</h4>
                    <p>Retrieved document chunks are strictly framed in the prompt as untrusted passive reference data. Protects downstream LLM reasoning against indirect prompt injection embedded in documents.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="feature-card">
                    <h4>5. Persistent Cloud & Standalone In-Memory Storage</h4>
                    <p>Supports enterprise-scale vector search via Qdrant Cloud clusters with deterministic collection mapping, while maintaining full standalone offline capability via NumPy matrix cosine similarity.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="feature-card">
                    <h4>6. Security & Citation Back-References</h4>
                    <p>Preserves full source attribution (document name, chunk ID, cosine similarity score) and enforces constant-time authentication access gating alongside credential sanitization.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### End-to-End Dataflow Diagram")
        st.code(
            """
Uploaded Files (PDF / TXT)
        │
        ▼
   DocumentLoader ──► Parse text & metadata
        │
        ▼
  DocumentChunker ──► Sliding-window chunking (size + overlap)
        │
        ▼
   Active Provider ──► Validated high-dimensional embeddings (768 / 1024 / 1536)
        │
        ▼
QdrantVectorStore ──► Store chunk payload & vectors via deterministic UUIDv5
        │
        ▼
SemanticRetriever ──► Top-K Cosine similarity retrieval with cutoff threshold
        │
        ▼
   Active Provider ──► Generate grounded answer (Untrusted Context Boundary)
            """,
            language="text"
        )


def main():
    """Application entrypoint with security access gate."""
    init_session_state()
    if not st.session_state.authenticated:
        render_login()
    else:
        render_app()


if __name__ == "__main__":
    main()
