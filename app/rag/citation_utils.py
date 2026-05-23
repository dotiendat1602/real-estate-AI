from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def rerank_citations(
    citations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order citations by source ranking metadata instead of keyword overlap."""
    if not citations:
        return citations

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        index, citation = item
        retriever_score = float(citation.get("score") or 0.0)
        return (int(citation.get("planningDocumentId") is not None), retriever_score, -index)

    return [citation for _, citation in sorted(enumerate(citations), key=sort_key, reverse=True)]


def build_planning_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for doc in docs:
        md = doc.metadata or {}
        doc_id = str(md.get("planningDocumentId") or "")
        chunk_type = md.get("chunkType") or "text"
        global_chunk_index = md.get("globalChunkIndex")
        chunk_index = md.get("chunkIndex")
        page_number = md.get("pageNumber")
        key = f"{doc_id}:{chunk_type}:{global_chunk_index}:{chunk_index}:{page_number}"
        if key in seen_ids:
            continue
        seen_ids.add(key)

        out.append({
            "postId": None,
            "propertyId": md.get("propertyId"),
            "planningDocumentId": md.get("planningDocumentId"),
            "documentScope": md.get("documentScope"),
            "documentType": md.get("documentType"),
            "dossierCode": md.get("dossierCode"),
            "city": md.get("city"),
            "district": md.get("district"),
            "planYear": md.get("planYear"),
            "title": md.get("title"),
            "sourceUrl": md.get("sourceUrl"),
            "format": md.get("format"),
            "chunkType": chunk_type,
            "chunkIndex": chunk_index,
            "globalChunkIndex": global_chunk_index,
            "pageNumber": page_number,
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "chunker": md.get("chunker"),
            "snippet": (doc.page_content or "")[:300],
        })

    return out
