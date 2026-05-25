from __future__ import annotations

from typing import Any
from urllib.parse import urldefrag

from langchain_core.documents import Document


def _coerce_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def source_url_with_page(source_url: Any, page_number: Any) -> str | None:
    raw_url = str(source_url or "").strip()
    if not raw_url:
        return None

    page = _coerce_positive_int(page_number)
    if page is None:
        return raw_url

    base_url, fragment = urldefrag(raw_url)
    if fragment:
        return raw_url
    return f"{base_url}#page={page}"


def _append_unique(target: list[Any], value: Any) -> None:
    if value in (None, ""):
        return
    if value not in target:
        target.append(value)


def _merge_citation_metadata(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    previous_page_number = existing.get("pageNumber")
    for field in ("pageNumber", "lineStart", "lineEnd", "sourceLocator"):
        if existing.get(field) in (None, "") and incoming.get(field) not in (None, ""):
            existing[field] = incoming[field]

    for field, list_field in (
        ("pageNumber", "pageNumbers"),
        ("sourceLocator", "sourceLocators"),
    ):
        values = existing.setdefault(list_field, [])
        if isinstance(values, list):
            _append_unique(values, incoming.get(field))

    if not existing.get("snippet") and incoming.get("snippet"):
        existing["snippet"] = incoming["snippet"]

    if (
        existing.get("planningDocumentId") is not None
        and previous_page_number in (None, "")
        and existing.get("pageNumber") not in (None, "")
    ):
        raw_url = existing.get("sourceUrlRaw") or incoming.get("sourceUrlRaw") or incoming.get("sourceUrl")
        existing["sourceUrlRaw"] = existing.get("sourceUrlRaw") or raw_url
        existing["sourceUrl"] = source_url_with_page(raw_url, existing.get("pageNumber"))


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not citations:
        return citations

    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    for citation in citations:
        planning_document_id = citation.get("planningDocumentId")
        post_id = citation.get("postId")
        if planning_document_id is not None:
            key = ("planning", str(planning_document_id))
        elif post_id is not None:
            key = ("post", str(post_id))
        else:
            key = (
                "source",
                "|".join(
                    str(citation.get(field) or "")
                    for field in ("sourceUrl", "title", "sourceLocator")
                ),
            )

        existing = seen.get(key)
        if existing is None:
            item = dict(citation)
            page_number = item.get("pageNumber")
            if item.get("planningDocumentId") is not None:
                raw_source_url = item.get("sourceUrlRaw") or item.get("sourceUrl")
                item["sourceUrlRaw"] = raw_source_url
                item["sourceUrl"] = source_url_with_page(raw_source_url, page_number)
            if page_number not in (None, ""):
                item.setdefault("pageNumbers", [page_number])
            if item.get("sourceLocator"):
                item.setdefault("sourceLocators", [item.get("sourceLocator")])
            seen[key] = item
            out.append(item)
            continue

        _merge_citation_metadata(existing, citation)

    return out


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
        if not doc_id:
            continue
        chunk_type = md.get("chunkType") or "text"
        global_chunk_index = md.get("globalChunkIndex")
        chunk_index = md.get("chunkIndex")
        page_number = md.get("pageNumber")
        key = doc_id
        if key in seen_ids:
            for citation in out:
                if str(citation.get("planningDocumentId") or "") == doc_id:
                    _merge_citation_metadata(
                        citation,
                        {
                            "pageNumber": page_number,
                            "lineStart": md.get("lineStart"),
                            "lineEnd": md.get("lineEnd"),
                            "sourceLocator": md.get("sourceLocator"),
                            "sourceUrl": md.get("sourceUrl"),
                            "sourceUrlRaw": md.get("sourceUrl"),
                            "snippet": (doc.page_content or "")[:300],
                        },
                    )
                    break
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
            "sourceUrl": source_url_with_page(md.get("sourceUrl"), page_number),
            "sourceUrlRaw": md.get("sourceUrl"),
            "format": md.get("format"),
            "chunkType": chunk_type,
            "chunkIndex": chunk_index,
            "globalChunkIndex": global_chunk_index,
            "pageNumber": page_number,
            "pageNumbers": [page_number] if page_number not in (None, "") else [],
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "sourceLocators": [md.get("sourceLocator")] if md.get("sourceLocator") else [],
            "chunker": md.get("chunker"),
            "snippet": (doc.page_content or "")[:300],
        })

    return out
