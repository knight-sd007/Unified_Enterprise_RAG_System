"""
RAG System Prompt Templates and Context Formatting.
"""

from typing import List, Dict, Any

RAG_SYSTEM_PROMPT = """You are an expert Enterprise RAG Assistant.
Answer the user's question using ONLY the provided document context snippets.

Strict Instructions:
1. Base your answer strictly on the provided Context. Do NOT invent information or assume facts not present in the context.
2. If the context does not contain enough information to answer the question, clearly state: "Based on the provided documents, I do not have sufficient information to answer this question."
3. Include inline citation back-references where applicable using the source filename or chunk ID [e.g. (Source: doc_c1)].
4. Maintain a professional, clear, and objective tone.
"""

def format_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved context chunks and user query into final RAG prompt."""
    if not retrieved_chunks:
        context_text = "No relevant context found in vector index."
    else:
        formatted_sources = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown Document")
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            score = chunk.get("score", 0.0)
            formatted_sources.append(
                f"--- Context Snippet [{idx}] (Source: {filename} | ID: {chunk_id} | Similarity: {score:.2f}) ---\n"
                f"{chunk['content']}\n"
            )
        context_text = "\n".join(formatted_sources)

    return f"""Context Information:
====================
{context_text}
====================

User Question: {query}

Answer:"""
