"""
Unit tests for Document Chunker.
"""

import unittest
from rag.loaders import Document
from rag.chunker import DocumentChunker


class TestDocumentChunker(unittest.TestCase):
    """Test suite verifying text chunking with overlap."""

    def test_short_document_single_chunk(self):
        """Documents smaller than chunk_size should produce exactly one chunk."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        doc = Document(content="Hello world RAG system.", metadata={"filename": "test.txt"})
        chunks = chunker.chunk_documents([doc])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Hello world RAG system.")
        self.assertEqual(chunks[0].metadata["filename"], "test.txt")

    def test_chunk_overlap(self):
        """Long text should split into multiple chunks with metadata."""
        text = ("Word " * 200).strip()  # ~1000 chars
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
        doc = Document(content=text, metadata={"filename": "long.txt"})
        chunks = chunker.chunk_documents([doc])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIsNotNone(chunk.content)
            self.assertIn("chunk_id", chunk.metadata)


if __name__ == "__main__":
    unittest.main()
