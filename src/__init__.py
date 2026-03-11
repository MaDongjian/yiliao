"""
Document Retrieval System
文档检索系统 - 支持Word、PDF、PPT的离线向量化检索
"""

from .parsers import DocumentParser
from .chunker import TextChunker
from .embedder import EmbeddingModel
from .storage import VectorStore
from .retriever import DocumentRetriever

__version__ = "1.0.0"
__all__ = ["DocumentParser", "TextChunker", "EmbeddingModel", "VectorStore", "DocumentRetriever"]
