from __future__ import annotations

import io
import importlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from pypdf import PdfReader

from ..utils.chunking import build_splitter

_splitter = build_splitter(chunk_size=1500, chunk_overlap=120)
_logger = logging.getLogger(__name__)


def _resolve_docling_max_pages() -> int:
    raw = os.getenv("PLANNING_DOCLING_MAX_PAGES", "20").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 20


@dataclass
class PlanningIngestPayload:
    planning_document_id: int
    title: str
    source_url: str
    format: str | None = None
    document_type: str | None = None
    dossier_code: str | None = None
    city: str | None = None
    district: str | None = None
    plan_year: int | None = None
    property_id: int | None = None
    raw_meta: dict[str, Any] | None = None


def _extract_pdf_text(binary: bytes) -> str:
    reader = PdfReader(io.BytesIO(binary))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def _extract_pdf_pages(binary: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(binary))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(_clean_text(page.extract_text() or ""))
    return pages


def _clean_text(text: str) -> str:
    value = re.sub(r"\u0000", "", text or "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _normalize_for_match(text: str) -> str:
    value = _clean_text(text).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _build_page_line_maps(page_texts: list[str]) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        lines = [_clean_text(line) for line in page_text.splitlines() if _clean_text(line)]
        norm_lines = [_normalize_for_match(line) for line in lines]
        maps.append(
            {
                "pageNumber": page_number,
                "lines": lines,
                "normLines": norm_lines,
                "cursor": 0,
            }
        )
    return maps


def _is_table_chunk(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return False

    if any("|" in line for line in lines):
        return True

    table_like_lines = sum(1 for line in lines if _looks_like_table_line(line))
    return table_like_lines >= max(2, len(lines) // 2)


def _locate_chunk_line_span(
    chunk_text: str,
    page_maps: list[dict[str, Any]],
    preferred_page: int | None = None,
) -> tuple[int | None, int | None, int | None]:
    normalized = _normalize_for_match(chunk_text)
    if not normalized:
        return None, None, None

    tokenized = [token for token in normalized.split(" ") if token]
    if not tokenized:
        return None, None, None

    candidate_maps = page_maps
    if preferred_page is not None:
        candidate_maps = sorted(page_maps, key=lambda p: 0 if p["pageNumber"] == preferred_page else 1)

    best: tuple[float, int | None, int | None, int | None, dict[str, Any] | None] = (0.0, None, None, None, None)

    for page in candidate_maps:
        norm_lines: list[str] = page["normLines"]
        if not norm_lines:
            continue

        start_idx = min(page["cursor"], len(norm_lines) - 1)
        for i in range(start_idx, len(norm_lines)):
            first_line = norm_lines[i]
            if tokenized[0] and tokenized[0] not in first_line and tokenized[0] not in normalized[:40]:
                continue

            for j in range(i, min(len(norm_lines), i + 25)):
                window = " ".join(norm_lines[i : j + 1]).strip()
                if not window:
                    continue

                overlap = sum(1 for tok in tokenized if tok in window)
                score = overlap / max(1, len(tokenized))

                if normalized in window:
                    score += 0.4

                if score > best[0]:
                    best = (score, page["pageNumber"], i + 1, j + 1, page)

                if score >= 0.92:
                    page["cursor"] = min(j + 1, len(norm_lines) - 1)
                    return page["pageNumber"], i + 1, j + 1

    if best[0] >= 0.35 and best[4] is not None:
        page = best[4]
        line_end = best[3] or best[2] or 1
        page["cursor"] = min(line_end, len(page["normLines"]) - 1)
        return best[1], best[2], best[3]

    return None, None, None


def _safe_meta_get(obj: Any, keys: list[str], default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] is not None:
                return obj[key]
        return default

    for key in keys:
        value = getattr(obj, key, None)
        if value is not None:
            return value
    return default


def _safe_chunk_text(obj: Any) -> str:
    value = _safe_meta_get(obj, ["text", "chunk_text", "content", "chunkText"], "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return str(obj).strip()


def _is_docling_table(chunk: Any, text: str) -> bool:
    meta = _safe_meta_get(chunk, ["meta", "metadata"], None)
    label_value = _safe_meta_get(meta, ["label", "labels", "item_type", "content_type", "kind"], "")
    labels = ""
    if isinstance(label_value, list):
        labels = " ".join(str(x).lower() for x in label_value)
    else:
        labels = str(label_value).lower()

    if "table" in labels:
        return True
    return _is_table_chunk(text)


def _extract_docling_page_number(chunk: Any) -> int | None:
    meta = _safe_meta_get(chunk, ["meta", "metadata"], None)

    candidates = [
        _safe_meta_get(meta, ["page_no", "page", "page_number", "start_page"], None),
        _safe_meta_get(chunk, ["page_no", "page", "page_number", "start_page"], None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            page_value = int(candidate)
            if page_value > 0:
                return page_value
        except Exception:
            continue

    return None


def _docling_structural_chunks(binary: bytes) -> list[dict[str, Any]]:
    try:
        chunking_module = importlib.import_module("docling.chunking")
        converter_module = importlib.import_module("docling.document_converter")
        HierarchicalChunker = getattr(chunking_module, "HierarchicalChunker")
        HybridChunker = getattr(chunking_module, "HybridChunker")
        DocumentConverter = getattr(converter_module, "DocumentConverter")
    except Exception:
        return []

    temp_path: str | None = None
    try:
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        temp_path = path

        with open(temp_path, "wb") as f:
            f.write(binary)

        converter = DocumentConverter()
        conv_result = converter.convert(temp_path)
        document = getattr(conv_result, "document", None)
        if document is None:
            return []

        chunks_out: list[dict[str, Any]] = []
        for chunker in (HybridChunker(), HierarchicalChunker()):
            chunk_iter = None
            if hasattr(chunker, "chunk"):
                chunk_iter = chunker.chunk(document)
            elif callable(chunker):
                chunk_iter = chunker(document)

            if chunk_iter is None:
                continue

            seen_texts: set[str] = set()
            for chunk in chunk_iter:
                chunk_text = _clean_text(_safe_chunk_text(chunk))
                if not chunk_text:
                    continue

                dedupe_key = _normalize_for_match(chunk_text)
                if not dedupe_key or dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)

                chunks_out.append(
                    {
                        "content": chunk_text,
                        "chunkType": "table" if _is_docling_table(chunk, chunk_text) else "text",
                        "pageNumber": _extract_docling_page_number(chunk),
                        "chunker": chunker.__class__.__name__,
                    }
                )

            if chunks_out:
                return chunks_out
    except Exception:
        # Any Docling parsing issue should gracefully fall back to splitter-based chunking.
        _logger.warning("Docling conversion failed. Falling back to splitter-based chunking.", exc_info=True)
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return []


def _is_docling_output_reliable(structural_chunks: list[dict[str, Any]], total_pages: int) -> bool:
    if not structural_chunks:
        return False

    if total_pages <= 1:
        return len(structural_chunks) >= 1

    page_numbers = {
        int(item.get("pageNumber"))
        for item in structural_chunks
        if isinstance(item.get("pageNumber"), int) and int(item.get("pageNumber")) > 0
    }

    # For large PDFs, Docling can partially fail with std::bad_alloc but still return a tiny result.
    # If page coverage/chunk volume is too low, prefer deterministic fallback.
    expected_min_pages = max(2, int(total_pages * 0.2))
    expected_min_chunks = max(8, int(total_pages * 0.4))

    if len(page_numbers) < expected_min_pages:
        return False

    if len(structural_chunks) < expected_min_chunks:
        return False

    return True


def _fallback_structural_chunks(text: str, page_maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_blocks, table_blocks = _split_text_and_table_blocks(text)
    chunks: list[dict[str, Any]] = []

    for block in text_blocks:
        for chunk in _splitter.split_text(block):
            clean_chunk = _clean_text(chunk)
            if clean_chunk:
                chunks.append({"content": clean_chunk, "chunkType": "text", "pageNumber": None, "chunker": "RecursiveCharacterTextSplitter"})

    for block in table_blocks:
        for chunk in _splitter.split_text(block):
            clean_chunk = _clean_text(chunk)
            if clean_chunk:
                chunks.append({"content": clean_chunk, "chunkType": "table", "pageNumber": None, "chunker": "RecursiveCharacterTextSplitter"})

    if chunks:
        return chunks

    for page in page_maps:
        for chunk in _splitter.split_text("\n".join(page["lines"])):
            clean_chunk = _clean_text(chunk)
            if clean_chunk:
                chunks.append({"content": clean_chunk, "chunkType": "text", "pageNumber": page["pageNumber"], "chunker": "RecursiveCharacterTextSplitter"})

    return chunks


def _build_source_locator(page_number: int | None, line_start: int | None, line_end: int | None) -> str:
    if page_number is None:
        return "unknown"
    if line_start is None:
        return f"page:{page_number}"
    if line_end is None or line_end == line_start:
        return f"page:{page_number},line:{line_start}"
    return f"page:{page_number},line:{line_start}-{line_end}"


def _looks_like_table_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False
    if "|" in compact:
        return True

    # Light heuristic for table-like rows with many numeric fragments.
    numeric_tokens = re.findall(r"\d+[\d.,]*", compact)
    return len(numeric_tokens) >= 3 and len(compact.split()) >= 6


def _split_text_and_table_blocks(text: str) -> tuple[list[str], list[str]]:
    text_blocks: list[str] = []
    table_blocks: list[str] = []

    current: list[str] = []
    current_type = "text"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        line_type = "table" if _looks_like_table_line(line) else "text"

        if not line:
            if current:
                block = "\n".join(current).strip()
                if block:
                    (table_blocks if current_type == "table" else text_blocks).append(block)
                current = []
            current_type = "text"
            continue

        if current and line_type != current_type:
            block = "\n".join(current).strip()
            if block:
                (table_blocks if current_type == "table" else text_blocks).append(block)
            current = [line]
            current_type = line_type
        else:
            current.append(line)
            current_type = line_type

    if current:
        block = "\n".join(current).strip()
        if block:
            (table_blocks if current_type == "table" else text_blocks).append(block)

    return text_blocks, table_blocks


def _enrich_for_embedding(content: str, metadata: dict[str, Any]) -> str:
    tags = [
        f"[document_scope={metadata.get('documentScope', 'planning')}]",
        f"[document_type={metadata.get('documentType') or 'unknown'}]",
        f"[city={metadata.get('city') or 'unknown'}]",
        f"[district={metadata.get('district') or 'unknown'}]",
        f"[plan_year={metadata.get('planYear') or 'unknown'}]",
        f"[chunk_type={metadata.get('chunkType') or 'text'}]",
        f"[dossier_code={metadata.get('dossierCode') or 'unknown'}]",
        f"[title={metadata.get('title') or 'unknown'}]",
    ]
    return "\n".join(tags + [content])


async def fetch_document_bytes(source_url: str, timeout: int = 60) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(source_url)
        response.raise_for_status()
        return response.content


async def build_planning_documents(payload: PlanningIngestPayload) -> tuple[list[Document], dict[str, int]]:
    source_bytes = await fetch_document_bytes(payload.source_url)

    fmt = (payload.format or "").lower().strip(".")
    if not fmt:
        fmt = "pdf" if payload.source_url.lower().endswith(".pdf") else "txt"

    if fmt == "pdf":
        raw_text = _extract_pdf_text(source_bytes)
        page_texts = _extract_pdf_pages(source_bytes)
    else:
        raw_text = source_bytes.decode("utf-8", errors="ignore")
        page_texts = [_clean_text(raw_text)] if raw_text else []

    cleaned = _clean_text(raw_text)
    if not cleaned:
        return [], {"textChunks": 0, "tableChunks": 0}

    page_maps = _build_page_line_maps(page_texts)
    total_pages = len(page_maps)

    structural_chunks: list[dict[str, Any]] = []
    if fmt == "pdf":
        max_docling_pages = _resolve_docling_max_pages()
        if max_docling_pages > 0 and total_pages > max_docling_pages:
            _logger.warning(
                "Skipping Docling for large PDF and using fallback chunking. total_pages=%s, max_docling_pages=%s",
                total_pages,
                max_docling_pages,
            )
        else:
            structural_chunks = _docling_structural_chunks(source_bytes)
            if structural_chunks and not _is_docling_output_reliable(structural_chunks, total_pages):
                _logger.warning(
                    "Docling output considered unreliable; switching to fallback chunking. total_pages=%s, docling_chunks=%s",
                    total_pages,
                    len(structural_chunks),
                )
                structural_chunks = []

    if not structural_chunks:
        structural_chunks = _fallback_structural_chunks(cleaned, page_maps)

    docs: list[Document] = []
    text_count = 0
    table_count = 0
    global_count = 0

    base_metadata: dict[str, Any] = {
        "documentScope": "planning",
        "planningDocumentId": payload.planning_document_id,
        "title": payload.title,
        "sourceUrl": payload.source_url,
        "format": payload.format,
        "documentType": payload.document_type,
        "dossierCode": payload.dossier_code,
        "city": payload.city,
        "district": payload.district,
        "planYear": payload.plan_year,
        "propertyId": payload.property_id,
        "rawMeta": payload.raw_meta or {},
    }

    text_chunk_index = 0
    table_chunk_index = 0

    for item in structural_chunks:
        chunk = _clean_text(item.get("content") or "")
        if not chunk:
            continue

        chunk_type = item.get("chunkType") or ("table" if _is_table_chunk(chunk) else "text")
        preferred_page = item.get("pageNumber")
        page_number, line_start, line_end = _locate_chunk_line_span(chunk, page_maps, preferred_page=preferred_page)

        if chunk_type == "table":
            table_count += 1
            table_chunk_index += 1
            chunk_index = table_chunk_index - 1
        else:
            text_count += 1
            text_chunk_index += 1
            chunk_index = text_chunk_index - 1

        global_count += 1

        source_locator = _build_source_locator(page_number, line_start, line_end)
        md = dict(base_metadata)
        md.update(
            {
                "chunkType": chunk_type,
                "chunkIndex": chunk_index,
                "globalChunkIndex": global_count,
                "pageNumber": page_number,
                "lineStart": line_start,
                "lineEnd": line_end,
                "sourceLocator": source_locator,
                "chunker": item.get("chunker") or "unknown",
                "chunkPreview": chunk[:400],
            }
        )
        docs.append(Document(page_content=_enrich_for_embedding(chunk, md), metadata=md))

    return docs, {"textChunks": text_count, "tableChunks": table_count}
