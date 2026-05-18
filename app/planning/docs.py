from __future__ import annotations

from langchain_core.documents import Document

from ..utils.text import normalize_text as _normalize_text



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


def planning_chunk_type(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunkType") or "").lower().strip()
