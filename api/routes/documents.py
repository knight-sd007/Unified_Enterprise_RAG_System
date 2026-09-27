"""
Document ingestion routes.
"""

import os
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from api.dependencies import APIError, get_rag_pipeline, require_authentication
from api.schemas import DocumentIngestResponse
from config.settings import Config
from providers.factory import get_provider_by_id
from rag.loaders import Document, DocumentLoader
from rag.pipeline import RAGPipeline
from utils.logging import logger
from utils.security import sanitize_error_message

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(require_authentication)],
)

# Guardrails
MAX_FILES_PER_REQUEST = 10
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB per file
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    summary="Ingest enterprise documents",
    description=(
        "Uploads PDF and TXT documents, chunks text, generates high-dimensional embeddings "
        "using the active AI provider, and indexes vectors into the vector store. Requires authentication."
    ),
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="One or more PDF or TXT files to ingest."),
    provider_id: Optional[str] = Form(None, description="Optional AI provider identifier ('openai', 'gemini', 'nvidia_nim'). Defaults to configured system provider."),
    chunk_size: Optional[int] = Form(500, description="Sliding-window chunk size in characters (100 to 2000)."),
    chunk_overlap: Optional[int] = Form(50, description="Sliding-window chunk overlap in characters (0 to 500)."),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestResponse:
    """Handles document upload, chunking, and vector embedding indexing."""
    if not files:
        raise APIError(
            status_code=400,
            code="NO_FILES_PROVIDED",
            message="Please provide at least one PDF or TXT file for ingestion.",
        )

    if len(files) > MAX_FILES_PER_REQUEST:
        raise APIError(
            status_code=400,
            code="TOO_MANY_FILES",
            message=f"Maximum of {MAX_FILES_PER_REQUEST} files permitted per ingestion request.",
        )

    # 1. Resolve and validate target AI provider
    target_provider_id = (provider_id or Config.get_ai_provider()).strip().lower()
    try:
        provider = get_provider_by_id(target_provider_id)
    except ValueError as e:
        raise APIError(
            status_code=400,
            code="INVALID_PROVIDER",
            message=str(e),
        )

    if not provider.is_configured():
        raise APIError(
            status_code=400,
            code="PROVIDER_NOT_CONFIGURED",
            message=f"Provider '{provider.name}' is not configured with valid API credentials.",
        )

    # 2. Read and parse uploaded files
    documents: List[Document] = []
    processed_filenames: List[str] = []

    for upload in files:
        # Sanitize filename against path traversal
        raw_name = upload.filename or "uploaded_doc.txt"
        safe_name = os.path.basename(raw_name)
        ext = os.path.splitext(safe_name)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise APIError(
                status_code=400,
                code="UNSUPPORTED_FILE_TYPE",
                message=f"Unsupported file format '{ext}' for file '{safe_name}'. Allowed formats: .pdf, .txt.",
            )

        try:
            content_bytes = await upload.read()
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Failed to read upload payload for '{safe_name}': {clean_err}")
            raise APIError(
                status_code=400,
                code="FILE_READ_ERROR",
                message=f"Failed to read file '{safe_name}'.",
            )

        if len(content_bytes) == 0:
            raise APIError(
                status_code=400,
                code="EMPTY_FILE",
                message=f"File '{safe_name}' is empty.",
            )

        if len(content_bytes) > MAX_FILE_SIZE_BYTES:
            raise APIError(
                status_code=413,
                code="FILE_TOO_LARGE",
                message=f"File '{safe_name}' exceeds maximum allowed size of 10MB.",
            )

        try:
            doc = DocumentLoader.load_from_file(content_bytes, safe_name)
            if not doc.content.strip():
                raise APIError(
                    status_code=400,
                    code="EMPTY_DOCUMENT_CONTENT",
                    message=f"No readable text could be extracted from file '{safe_name}'.",
                )
            documents.append(doc)
            processed_filenames.append(safe_name)
        except APIError:
            raise
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Failed to parse document '{safe_name}': {clean_err}")
            raise APIError(
                status_code=400,
                code="DOCUMENT_PARSE_ERROR",
                message=f"Failed to parse file '{safe_name}': {clean_err}",
            )

    # 3. Configure chunker parameters on pipeline
    if chunk_size is not None:
        pipeline.chunker.chunk_size = max(100, min(2000, chunk_size))
    if chunk_overlap is not None:
        pipeline.chunker.chunk_overlap = max(0, min(500, chunk_overlap))

    # 4. Ingest parsed documents into vector store
    res = pipeline.ingest_documents(documents, provider)

    if res.get("status") == "error":
        raise APIError(
            status_code=400,
            code="INGESTION_FAILED",
            message=res.get("message", "Document ingestion failed."),
        )

    return DocumentIngestResponse(
        status=res.get("status", "success"),
        document_count=res.get("document_count", len(documents)),
        chunk_count=res.get("chunk_count", 0),
        provider=res.get("provider", provider.name),
        provider_id=res.get("provider_id", provider.provider_id),
        embedding_model=res.get("embedding_model", provider.get_embedding_model_name()),
        vector_dimension=res.get("vector_dimension"),
        collection_name=res.get("collection_name"),
        message=res.get("message"),
        filenames=processed_filenames,
    )
