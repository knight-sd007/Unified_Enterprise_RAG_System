"""
Document ingestion and lifecycle management routes.
Integrates vector indexing, durable SQLite metadata repository, and Google Drive storage.
"""

import os
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, Path, Query, UploadFile

from api.dependencies import APIError, UserSession, get_rag_pipeline, require_authentication
from api.schemas import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentItem,
    DocumentListResponse,
    UserClearResponse,
)
from config.settings import Config
from providers.factory import get_provider_by_id
from rag.loaders import Document, DocumentLoader
from rag.pipeline import RAGPipeline
from rag.storage.google_drive import GoogleDriveStorage
from rag.storage.metadata_db import get_metadata_repo
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
        "using the active AI provider, syncs with Google Drive if authorized, and indexes vectors "
        "bound to the authenticated owner. Requires authentication."
    ),
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="One or more PDF or TXT files to ingest."),
    provider_id: Optional[str] = Form(None, description="Optional AI provider identifier ('openai', 'gemini', 'nvidia_nim'). Defaults to configured system provider."),
    chunk_size: Optional[int] = Form(500, description="Sliding-window chunk size in characters (100 to 2000)."),
    chunk_overlap: Optional[int] = Form(50, description="Sliding-window chunk overlap in characters (0 to 500)."),
    session: UserSession = Depends(require_authentication),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestResponse:
    """Handles document upload, chunking, and vector embedding indexing with owner binding."""
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
    file_bytes_map = {}

    drive_storage = GoogleDriveStorage()

    for upload in files:
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
            doc = DocumentLoader.load_from_file(content_bytes, safe_name, owner_id=session.user_id)
            if not doc.content.strip():
                raise APIError(
                    status_code=400,
                    code="EMPTY_DOCUMENT_CONTENT",
                    message=f"No readable text could be extracted from file '{safe_name}'.",
                )
            documents.append(doc)
            processed_filenames.append(safe_name)
            file_bytes_map[doc.metadata["doc_id"]] = (safe_name, content_bytes, ext)
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
    res = pipeline.ingest_documents(documents, provider, owner_id=session.user_id)

    if res.get("status") == "error":
        raise APIError(
            status_code=400,
            code="INGESTION_FAILED",
            message=res.get("message", "Document ingestion failed."),
        )

    # 5. Persist document metadata in SQLite repository & sync with Drive if available
    repo = get_metadata_repo()
    for doc in documents:
        doc_id = doc.metadata.get("doc_id")
        fname, c_bytes, ext = file_bytes_map.get(doc_id, (doc.metadata.get("filename", "unknown"), b"", ".txt"))
        mime_type = "application/pdf" if ext == ".pdf" else "text/plain"

        drive_file_id = None
        try:
            drive_file_id = await drive_storage.upload_file(
                owner_id=session.user_id,
                filename=fname,
                content=c_bytes,
                mime_type=mime_type,
            )
        except Exception as e:
            logger.warning(f"Drive upload skipped or failed for {fname}: {e}")

        # Count chunks for this specific document
        doc_chunk_count = sum(1 for c in pipeline.chunker.chunk_documents([doc])) if hasattr(pipeline, "chunker") else 1

        repo.save_document(
            doc_id=doc_id,
            owner_id=session.user_id,
            filename=fname,
            file_size=len(c_bytes),
            char_count=len(doc.content),
            provider_id=provider.provider_id,
            embedding_model=provider.get_embedding_model_name(),
            chunk_count=doc_chunk_count,
            drive_file_id=drive_file_id,
            status="indexed",
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
        doc_ids=res.get("doc_ids", []),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List active documents",
    description="Returns list of documents owned by the authenticated user. Requires authentication.",
)
async def list_documents(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier filter."),
    session: UserSession = Depends(require_authentication),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentListResponse:
    """Lists documents owned by the current user."""
    owner_scope = None if session.is_admin else session.user_id
    repo = get_metadata_repo()
    db_docs = repo.list_documents(owner_id=owner_scope)

    # If DB has documents, use them; otherwise fallback to vector store list
    if db_docs:
        doc_items = [
            DocumentItem(
                doc_id=d["doc_id"],
                filename=d["filename"],
                owner_id=d["owner_id"],
                provider_id=d["provider_id"],
                embedding_model=d.get("embedding_model"),
                chunk_count=d["chunk_count"],
                created_at=d["created_at"],
                char_count=d.get("char_count"),
                file_size=d.get("file_size"),
                drive_file_id=d.get("drive_file_id"),
                status=d.get("status", "indexed"),
            )
            for d in db_docs
            if provider_id is None or d["provider_id"] == provider_id
        ]
    else:
        raw_docs = pipeline.list_documents(owner_id=owner_scope, provider_id=provider_id)
        doc_items = [
            DocumentItem(
                doc_id=d["doc_id"],
                filename=d["filename"],
                owner_id=d["owner_id"],
                provider_id=d["provider_id"],
                chunk_count=d["chunk_count"],
                created_at=d["created_at"],
                char_count=d.get("char_count"),
            )
            for d in raw_docs
        ]

    return DocumentListResponse(
        documents=doc_items,
        total_documents=len(doc_items),
        owner_id=session.user_id,
        user_id=session.user_id,
    )


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its chunks",
    description="Deletes a specific document and purges all its vector chunks. Caller must own the document. Requires authentication.",
)
async def delete_document(
    doc_id: str = Path(..., description="Unique document ID to delete."),
    provider_id: Optional[str] = Query(None, description="Optional provider identifier filter."),
    session: UserSession = Depends(require_authentication),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentDeleteResponse:
    """Deletes a specific document owned by the user."""
    owner_scope = None if session.is_admin else session.user_id
    repo = get_metadata_repo()

    # Check and delete from SQLite repository
    deleted_meta = repo.delete_document(doc_id=doc_id, owner_id=owner_scope)
    deleted_chunks = pipeline.delete_document(doc_id=doc_id, owner_id=owner_scope, provider_id=provider_id)

    if deleted_chunks == 0 and not deleted_meta:
        raise APIError(
            status_code=404,
            code="DOCUMENT_NOT_FOUND",
            message=f"Document '{doc_id}' not found or you do not have permission to delete it.",
        )

    # Cleanup Google Drive if remote file was tracked
    if deleted_meta and deleted_meta.get("drive_file_id"):
        try:
            drive_storage = GoogleDriveStorage()
            await drive_storage.delete_file(deleted_meta["drive_file_id"], user_id=session.user_id)
        except Exception as e:
            logger.warning(f"Failed to delete Google Drive file {deleted_meta['drive_file_id']}: {e}")

    effective_chunks = deleted_chunks or (deleted_meta["chunk_count"] if deleted_meta else 0)

    return DocumentDeleteResponse(
        status="success",
        message=f"Document '{doc_id}' and {effective_chunks} associated vector chunk(s) purged successfully.",
        doc_id=doc_id,
        deleted_chunks=effective_chunks,
    )


@router.delete(
    "",
    response_model=UserClearResponse,
    summary="Clear all documents owned by user",
    description="Purges all documents and vector chunks owned by the authenticated user. Does not affect other users. Requires authentication.",
)
async def clear_user_documents(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier filter."),
    session: UserSession = Depends(require_authentication),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> UserClearResponse:
    """Purges all documents owned by the caller."""
    repo = get_metadata_repo()
    deleted_docs = repo.delete_all_for_user(owner_id=session.user_id)

    doc_count, chunk_count = pipeline.clear_user_documents(owner_id=session.user_id, provider_id=provider_id)
    eff_doc_count = max(doc_count, len(deleted_docs))
    eff_chunk_count = max(chunk_count, sum(d.get("chunk_count", 0) for d in deleted_docs))

    # Cleanup remote Drive files if any
    drive_storage = GoogleDriveStorage()
    for d in deleted_docs:
        if d.get("drive_file_id"):
            try:
                await drive_storage.delete_file(d["drive_file_id"], user_id=session.user_id)
            except Exception:
                pass

    return UserClearResponse(
        status="success",
        message=f"Successfully purged {eff_chunk_count} vector chunk(s) across {eff_doc_count} document(s) for user '{session.user_id}'.",
        owner_id=session.user_id,
        user_id=session.user_id,
        deleted_documents=eff_doc_count,
        deleted_chunks=eff_chunk_count,
    )
