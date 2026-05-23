from __future__ import annotations

from typing import Any
import re

from langchain_core.documents import Document

from ..utils.text import repair_mojibake
from .text_utils import normalize_text as _normalize_text, sanitize_llm_text
from .listing_context import (
    build_structured_listing_context as _build_structured_listing_context,
    merge_context_snippets as _merge_context_snippets,
)

_CONTEXT_DOC_HEADER_PREFIXES = (
    "=== BAT DONG SAN",
    "LISTING_ID:",
    "=== PLANNING",
    "--- THÔNG TIN CHI TIẾT ---",
    "--- DAC DIEM ---",
    "--- VI TRI ---",
)

def _dedupe_repeated_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text or "") if block.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        key = _normalize_text(block)[:220]
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return "\n\n".join(out)


def _split_long_line(line: str) -> list[str]:
    prepared = line
    for marker in (
        "Giá thuê",
        "Giá:",
        "Liên hệ",
        "Ưu điểm",
        "Thông tin nhà",
        "Tiện ích",
        "Diện tích",
        "Kết cấu",
        "Gồm",
        "Tổng cộng",
        "Nhà phù hợp",
        "Loại:",
        "Danh mục:",
    ):
        prepared = prepared.replace(marker, f"\n{marker}")

    parts = re.split(r"\n+|(?<=[\.!?;])\s+", prepared)
    return [part.strip(" -") for part in parts if part and part.strip()]


def _compact_doc_content(question: str, content: str, max_chars: int, max_lines: int = 8) -> str:
    if not content:
        return ""

    prepared = content.replace("<br/>", "\n").replace("<br>", "\n")
    prepared = _dedupe_repeated_blocks(prepared)
    if len(prepared) <= max_chars and prepared.count("\n") <= max_lines:
        return prepared

    raw_lines = [line.strip() for line in prepared.splitlines() if line.strip()]
    lines: list[str] = []
    for raw_line in raw_lines:
        if len(raw_line) > 200:
            lines.extend(_split_long_line(raw_line))
        else:
            lines.append(raw_line)

    # Remove near-duplicate fragments after sentence splitting.
    deduped_lines: list[str] = []
    deduped_seen: set[str] = set()
    for line in lines:
        key = _normalize_text(line)
        if not key or key in deduped_seen:
            continue
        deduped_seen.add(key)
        deduped_lines.append(line)
    lines = deduped_lines

    if not lines:
        return prepared[:max_chars]

    selected: list[str] = []
    selected_norm: set[str] = set()
    used_chars = 0

    # Keep the first section header when available so the model retains listing identity.
    for line in lines:
        if any(line.startswith(prefix) for prefix in _CONTEXT_DOC_HEADER_PREFIXES):
            key = _normalize_text(line)
            if key and key not in selected_norm:
                selected.append(line)
                selected_norm.add(key)
                used_chars += len(line) + 1
            break

    for line in lines:
        if len(selected) >= max_lines:
            break

        normalized = _normalize_text(line)
        if not normalized or normalized in selected_norm:
            continue

        next_chars = used_chars + len(line) + 1
        if next_chars > max_chars:
            continue

        selected.append(line)
        selected_norm.add(normalized)
        used_chars = next_chars

    if not selected:
        # Safe fallback: keep a compact prefix if scoring filtered everything.
        return prepared[:max_chars]

    compact = "\n".join(selected).strip()
    return compact[:max_chars]


def _doc_identity(doc: Document) -> str:
    md = doc.metadata or {}
    return "|".join(
        [
            str(md.get("postId") or ""),
            str(md.get("propertyId") or ""),
            str(md.get("planningDocumentId") or ""),
            str(md.get("chunkType") or ""),
            str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or ""),
            _normalize_text((doc.page_content or "")[:120]),
        ]
    )


def _is_planning_context_doc(doc: Document) -> bool:
    md = doc.metadata or {}
    return md.get("planningDocumentId") is not None and md.get("postId") is None


def _listing_raw_evidence(question: str, doc: Document) -> str:
    return _compact_doc_content(question, doc.page_content or "", max_chars=1100, max_lines=16)


def prepare_docs_for_context(
    question: str,
    docs: list[Document],
    max_docs: int = 4,
    max_chars_per_doc: int = 1400,
) -> list[Document]:
    if not docs:
        return []

    max_docs = max(1, int(max_docs))
    max_chars_per_doc = max(400, int(max_chars_per_doc))

    deduped: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        key = _doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    planning_only = bool(deduped) and all(_is_planning_context_doc(doc) for doc in deduped)

    if planning_only:
        # Planning docs have already been ranked/compacted in planning retrieval.
        selected = deduped[:max_docs]
    else:
        # Keep retriever order as the document ranking source; this avoids a
        # second keyword-based rerank layer inside context preparation.
        selected = deduped[:max_docs]

    # Keep listing context rich by default to reduce intent-keyword coupling.
    needs_rich_listing_context = not planning_only

    prepared: list[Document] = []
    seen_compacted_keys: set[str] = set()
    structured_post_ids: set[str] = set()

    for doc in selected:
        if _is_planning_context_doc(doc):
            compacted = (doc.page_content or "").strip()
            planning_char_limit = max(max_chars_per_doc, 2600)
            if len(compacted) > planning_char_limit:
                compacted = compacted[:planning_char_limit].rstrip()
        else:
            md = doc.metadata or {}
            structured = _build_structured_listing_context(
                question,
                doc,
                raw_evidence_builder=_listing_raw_evidence,
            )
            post_id_key = str(md.get("postId") or "")
            use_structured = bool(structured)
            if use_structured and post_id_key and post_id_key in structured_post_ids:
                use_structured = False

            if use_structured:
                compacted = structured or ""
                if post_id_key:
                    structured_post_ids.add(post_id_key)

                if needs_rich_listing_context:
                    raw_excerpt_max_lines = 16
                    raw_excerpt = _compact_doc_content(
                        question,
                        doc.page_content or "",
                        max_chars=max_chars_per_doc,
                        max_lines=raw_excerpt_max_lines,
                    )
                    compacted = _merge_context_snippets(compacted, raw_excerpt, max_chars=max_chars_per_doc)
            else:
                compacted = _compact_doc_content(question, doc.page_content or "", max_chars=max_chars_per_doc)

        if not compacted:
            continue

        compacted = sanitize_llm_text(repair_mojibake(compacted))
        if not compacted:
            continue

        if _is_planning_context_doc(doc):
            md = doc.metadata or {}
            planning_chunk_idx = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
            compacted_key = "|".join(
                [
                    "planning",
                    str(md.get("planningDocumentId") or ""),
                    str(md.get("chunkType") or ""),
                    str(planning_chunk_idx or ""),
                ]
            )
        else:
            compacted_key = _normalize_text(compacted)[:260]

        if compacted_key and compacted_key in seen_compacted_keys:
            continue
        if compacted_key:
            seen_compacted_keys.add(compacted_key)

        prepared.append(Document(page_content=compacted, metadata=doc.metadata))

    if prepared:
        return prepared

    # Final fallback keeps at least one document so generation/evaluation remains grounded.
    fallback = selected[0]
    return [Document(page_content=(fallback.page_content or "")[:max_chars_per_doc], metadata=fallback.metadata)]


def build_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
    seen_post_ids = set()
    
    for d in docs:
        md = d.metadata or {}
        post_id = md.get("postId")
        
        # Deduplicate by postId (avoid showing same post multiple times)
        if post_id and post_id in seen_post_ids:
            continue
        
        if post_id:
            seen_post_ids.add(post_id)
        
        out.append({
            "postId": post_id,
            "propertyId": md.get("propertyId"),
            "planningDocumentId": md.get("planningDocumentId"),
            "title": md.get("title"),
            "postTitle": md.get("postTitle"),
            "sourceUrl": md.get("sourceUrl") or (f"/posts/{post_id}" if post_id else None),
            "format": md.get("format"),
            "documentScope": md.get("documentScope"),
            "documentType": md.get("documentType"),
            "dossierCode": md.get("dossierCode"),
            "planYear": md.get("planYear"),
            "chunkType": md.get("chunkType"),
            "chunkIndex": md.get("chunkIndex"),
            "globalChunkIndex": md.get("globalChunkIndex"),
            "pageNumber": md.get("pageNumber"),
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "chunker": md.get("chunker"),
            "postType": md.get("postType"),
            "categoryName": md.get("categoryName"),
            "city": md.get("city"),
            "district": md.get("district"),
            "ward": md.get("ward"),
            "location": md.get("location"),
            "price": md.get("price"),
            "area": md.get("area"),
            "bedrooms": md.get("bedrooms"),
            "amenities": md.get("amenities", []),
            "snippet": repair_mojibake(d.page_content or "")[:300],
        })
    
    return out


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        sanitize_llm_text(doc.page_content)
        for doc in docs
        if doc.page_content
    )
