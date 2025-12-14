"""Document loaders for Agent layer - moved from src/infrastructure/document_loaders/"""

from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class DoclingDocumentLoader:
    """Load and process documents using Docling."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.converter = DocumentConverter()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self, document_paths: list[str | Path]) -> list[Document]:
        all_chunks = []
        for doc_path in document_paths:
            try:
                logger.info(f"Loading document: {doc_path}")
                result = self.converter.convert(str(doc_path))
                text = result.document.export_to_markdown()
                doc = Document(page_content=text, metadata={"source": str(doc_path), "filename": Path(doc_path).name})
                chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
                logger.info(f"✓ Loaded {len(chunks)} chunks from {doc_path}")
            except Exception as e:
                logger.error(f"✗ Failed to load {doc_path}: {e}")
                logger.exception("Full error:")
        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks
