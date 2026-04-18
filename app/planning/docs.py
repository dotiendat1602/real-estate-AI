from __future__ import annotations

import re
import unicodedata
from typing import Optional

from langchain_core.documents import Document

from .profiles import strip_accents


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = strip_accents(lowered)
    lowered = lowered.replace("Ä‘", "d").replace("Ä", "d")
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def dedupe_planning_docs(docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()

    for doc in docs:
        md = doc.metadata or {}
        key = "|".join(
            [
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or ""),
                str(md.get("pageNumber") or ""),
                (doc.page_content or "")[:80],
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)

    return out


def planning_doc_identity(doc: Document) -> str:
    md = doc.metadata or {}
    chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
    return "|".join(
        [
            str(md.get("planningDocumentId") or ""),
            str(md.get("chunkType") or ""),
            str(chunk_idx or ""),
            str(md.get("pageNumber") or ""),
            _normalize_text((doc.page_content or "")[:120]),
        ]
    )


def planning_doc_pid_idx(doc: Document) -> tuple[Optional[int], Optional[int]]:
    md = doc.metadata or {}

    raw_pid = md.get("planningDocumentId")
    try:
        planning_document_id = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        planning_document_id = None

    raw_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
    try:
        chunk_index = int(raw_idx) if raw_idx is not None else None
    except (TypeError, ValueError):
        chunk_index = None

    return planning_document_id, chunk_index


def planning_chunk_type(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunkType") or "").lower().strip()
