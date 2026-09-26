from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings


logger = logging.getLogger(__name__)

SUPPORTED_KNOWLEDGE_EXTENSIONS = {".md", ".txt"}


class IngestionResult(BaseModel):
    document_id: str
    title: str
    source_path: str
    chunk_count: int
    status: str = "completed"


class VectorSearchResult(BaseModel):
    evidence_id: str
    document_id: str
    title: str
    document_type: str
    text: str
    source_path: str
    chunk_index: int

    distance: float = 0.0
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStoreService:
    """
    Local persistent ChromaDB service for CORTEX enterprise knowledge.

    The service:

    - Creates or opens the configured collection
    - Chunks Markdown and text documents
    - Generates local embeddings
    - Upserts document chunks
    - Performs semantic retrieval
    - Returns evidence metadata and source references
    """

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        if self.settings.embedding_provider != "local":
            raise RuntimeError(
                "This implementation currently supports "
                "EMBEDDING_PROVIDER=local. "
                "Azure embeddings will be added after the "
                "local RAG pipeline is validated."
            )

        self.persist_path = self.settings.chroma_path
        self.persist_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_function = DefaultEmbeddingFunction()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            embedding_function=self.embedding_function,
            metadata={
                "description": (
                    "Approved enterprise knowledge used by CORTEX RAG"
                ),
                "embedding_provider": "local",
                "application": "CORTEX",
            },
        )

    @staticmethod
    def chunk_text(
        text: str,
        *,
        chunk_words: int = 220,
        overlap_words: int = 40,
    ) -> list[str]:
        normalized_text = re.sub(
            r"\n{3,}",
            "\n\n",
            text.strip(),
        )

        words = normalized_text.split()

        if not words:
            return []

        if chunk_words <= overlap_words:
            raise ValueError(
                "chunk_words must be greater than overlap_words."
            )

        chunks: list[str] = []
        step = chunk_words - overlap_words

        for start in range(0, len(words), step):
            end = start + chunk_words
            chunk = " ".join(words[start:end]).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

        return chunks

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            value.strip().lower(),
        ).strip("-")

        return slug or "document"

    @classmethod
    def _create_document_id(
        cls,
        path: Path,
    ) -> str:
        slug = cls._slugify(path.stem)

        path_hash = hashlib.sha1(
            str(path.resolve()).encode("utf-8")
        ).hexdigest()[:8]

        return f"{slug}-{path_hash}"

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        sanitized: dict[str, str | int | float | bool] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)

        return sanitized

    def upsert_document(
        self,
        *,
        document_id: str,
        title: str,
        text: str,
        source_path: str,
        document_type: str = "policy",
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        chunks = self.chunk_text(text)

        if not chunks:
            raise ValueError(
                f"Document '{title}' contains no readable text."
            )

        # Delete previous chunks for this logical document so that
        # re-ingestion cannot leave outdated chunks behind.
        self.collection.delete(
            where={"document_id": document_id},
        )

        chunk_ids: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = (
                f"{document_id}-chunk-{index:03d}"
            )

            metadata: dict[str, Any] = {
                "document_id": document_id,
                "title": title,
                "source_path": source_path,
                "document_type": document_type,
                "chunk_index": index,
                "total_chunks": len(chunks),
            }

            if extra_metadata:
                metadata.update(extra_metadata)

            chunk_ids.append(chunk_id)
            metadatas.append(
                self._sanitize_metadata(metadata)
            )

        self.collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            metadatas=metadatas,
        )

        return IngestionResult(
            document_id=document_id,
            title=title,
            source_path=source_path,
            chunk_count=len(chunks),
        )

    def ingest_file(
        self,
        path: str | Path,
        *,
        document_type: str = "policy",
    ) -> IngestionResult:
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(
                f"Knowledge document not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Knowledge path is not a file: {file_path}"
            )

        if file_path.suffix.lower() not in (
            SUPPORTED_KNOWLEDGE_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported knowledge file type: "
                f"{file_path.suffix}"
            )

        text = file_path.read_text(
            encoding="utf-8",
        )

        document_id = self._create_document_id(
            file_path
        )

        title = file_path.stem.replace(
            "_",
            " ",
        ).replace(
            "-",
            " ",
        ).title()

        return self.upsert_document(
            document_id=document_id,
            title=title,
            text=text,
            source_path=str(file_path),
            document_type=document_type,
            extra_metadata={
                "file_name": file_path.name,
                "file_extension": file_path.suffix.lower(),
            },
        )

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        document_type: str = "policy",
    ) -> list[IngestionResult]:
        directory_path = Path(directory).resolve()

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: "
                f"{directory_path}"
            )

        results: list[IngestionResult] = []

        for path in sorted(directory_path.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_KNOWLEDGE_EXTENSIONS
            ):
                result = self.ingest_file(
                    path,
                    document_type=document_type,
                )
                results.append(result)

        return results

    def search(
        self,
        query: str,
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Knowledge search query cannot be empty."
            )

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        result_count = min(
            max(n_results, 1),
            collection_count,
        )

        query_arguments: dict[str, Any] = {
            "query_texts": [clean_query],
            "n_results": result_count,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_arguments["where"] = where

        response = self.collection.query(
            **query_arguments
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (
            response.get("documents") or [[]]
        )[0]
        metadatas = (
            response.get("metadatas") or [[]]
        )[0]
        distances = (
            response.get("distances") or [[]]
        )[0]

        results: list[VectorSearchResult] = []

        for index, evidence_id in enumerate(ids):
            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and metadatas[index]
                else {}
            )

            distance = float(
                distances[index]
                if index < len(distances)
                and distances[index] is not None
                else 0.0
            )

            # This transformation works for non-negative distance
            # values without assuming a particular Chroma metric.
            relevance_score = 1.0 / (
                1.0 + max(distance, 0.0)
            )

            results.append(
                VectorSearchResult(
                    evidence_id=str(evidence_id),
                    document_id=str(
                        metadata.get(
                            "document_id",
                            "",
                        )
                    ),
                    title=str(
                        metadata.get(
                            "title",
                            "Untitled document",
                        )
                    ),
                    document_type=str(
                        metadata.get(
                            "document_type",
                            "knowledge",
                        )
                    ),
                    text=str(document or ""),
                    source_path=str(
                        metadata.get(
                            "source_path",
                            "",
                        )
                    ),
                    chunk_index=int(
                        metadata.get(
                            "chunk_index",
                            0,
                        )
                    ),
                    distance=round(
                        distance,
                        6,
                    ),
                    relevance_score=round(
                        relevance_score,
                        4,
                    ),
                    metadata=dict(metadata),
                )
            )

        return results

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        self.collection.delete(
            where={"document_id": document_id},
        )

    def collection_count(self) -> int:
        return int(self.collection.count())

    def health_check(self) -> dict[str, Any]:
        try:
            return {
                "status": "ready",
                "collection": self.collection.name,
                "document_chunks": self.collection_count(),
                "persist_path": str(self.persist_path),
                "embedding_provider": (
                    self.settings.embedding_provider
                ),
            }

        except Exception as exc:
            logger.exception(
                "ChromaDB health check failed."
            )

            return {
                "status": "failed",
                "collection": (
                    self.settings.chroma_collection
                ),
                "document_chunks": 0,
                "persist_path": str(self.persist_path),
                "embedding_provider": (
                    self.settings.embedding_provider
                ),
                "error": str(exc),
            }


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService()