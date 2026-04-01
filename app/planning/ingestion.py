from __future__ import annotations

import io
import importlib
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter

from ..utils.chunking import build_splitter

_splitter = build_splitter(chunk_size=1500, chunk_overlap=120)
_logger = logging.getLogger(__name__)
_docling_disabled_due_bad_alloc = False


def _resolve_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_http_verify() -> bool | str:
    if not _resolve_bool_env("PLANNING_INGEST_SSL_VERIFY", True):
        _logger.warning("Planning ingest SSL verification is disabled by PLANNING_INGEST_SSL_VERIFY=false")
        return False

    ca_bundle = (
        os.getenv("PLANNING_INGEST_CA_BUNDLE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or ""
    ).strip()

    if not ca_bundle:
        return True

    if not os.path.exists(ca_bundle):
        _logger.warning(
            "CA bundle path does not exist. Falling back to system trust store. path=%s",
            ca_bundle,
        )
        return True

    return ca_bundle


def _resolve_docling_max_pages() -> int:
    raw = os.getenv("PLANNING_DOCLING_MAX_PAGES", "20").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 20


def _resolve_docling_enabled() -> bool:
    return _resolve_bool_env("PLANNING_DOCLING_ENABLED", True)


def _resolve_docling_max_pdf_bytes() -> int:
    raw = os.getenv("PLANNING_DOCLING_MAX_PDF_BYTES", "15728640").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 15728640


def _resolve_docling_disable_after_bad_alloc() -> bool:
    return _resolve_bool_env("PLANNING_DOCLING_DISABLE_AFTER_BAD_ALLOC", True)


def _resolve_docling_batch_pages() -> int:
    raw = os.getenv("PLANNING_DOCLING_BATCH_PAGES", "6").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 6


def _resolve_ssl_allow_insecure_fallback() -> bool:
    return _resolve_bool_env("PLANNING_INGEST_SSL_ALLOW_INSECURE_FALLBACK", False)


def _resolve_pdf_ocr_fallback_enabled() -> bool:
    return _resolve_bool_env("PLANNING_PDF_OCR_FALLBACK_ENABLED", True)


def _resolve_scan_pdf_use_docling_first() -> bool:
    return _resolve_bool_env("PLANNING_SCANNED_PDF_USE_DOCLING_FIRST", False)


def _resolve_pdf_ocr_max_pages() -> int:
    raw = os.getenv("PLANNING_PDF_OCR_MAX_PAGES", "0").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def _resolve_pdf_ocr_render_scale() -> float:
    raw = os.getenv("PLANNING_PDF_OCR_RENDER_SCALE", "1.25").strip()
    try:
        value = float(raw)
        return max(0.5, min(value, 3.0))
    except Exception:
        return 1.25


def _resolve_ingest_soft_timeout_seconds() -> int:
    raw = os.getenv("PLANNING_INGEST_SOFT_TIMEOUT_SECONDS", "0").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def _resolve_ingest_require_full() -> bool:
    return _resolve_bool_env("PLANNING_INGEST_REQUIRE_FULL", False)


def _resolve_ocr_progress_every_pages() -> int:
    raw = os.getenv("PLANNING_OCR_PROGRESS_EVERY_PAGES", "5").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 5


def _extract_rapidocr_lines(ocr_output: Any) -> list[str]:
    txts: Any = getattr(ocr_output, "txts", None)

    if txts is None and isinstance(ocr_output, tuple):
        for value in ocr_output:
            if isinstance(value, (list, tuple)) and value and all(isinstance(x, str) for x in value):
                txts = value
                break

    if isinstance(txts, tuple):
        txts = list(txts)

    if not isinstance(txts, list):
        return []

    return [_clean_text(line) for line in txts if isinstance(line, str) and _clean_text(line)]


def _ocr_lines_with_stripes(ocr_engine: Any, arr: Any) -> list[str]:
    height = int(getattr(arr, "shape", [0])[0]) if getattr(arr, "shape", None) is not None else 0
    if height <= 1:
        return []

    stripe_candidates = (2, 3, 4, 6)
    for stripes in stripe_candidates:
        step = max(1, height // stripes)
        lines: list[str] = []

        try:
            for index in range(stripes):
                top = index * step
                bottom = height if index == stripes - 1 else min(height, (index + 1) * step)
                if bottom - top <= 1:
                    continue

                tile = arr[top:bottom, :]
                output = ocr_engine(tile)
                lines.extend(_extract_rapidocr_lines(output))

            if lines:
                _logger.warning(
                    "OCR stripe fallback succeeded. stripes=%s lines=%s",
                    stripes,
                    len(lines),
                )
                return lines
        except Exception:
            continue

    return []


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


def _docling_structural_chunks(binary: bytes, page_offset: int = 0) -> list[dict[str, Any]]:
    global _docling_disabled_due_bad_alloc

    if not _resolve_docling_enabled():
        return []

    if _docling_disabled_due_bad_alloc and _resolve_docling_disable_after_bad_alloc():
        return []

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

                page_number = _extract_docling_page_number(chunk)

                dedupe_key = _normalize_for_match(chunk_text)
                if not dedupe_key or dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)

                chunks_out.append(
                    {
                        "content": chunk_text,
                        "chunkType": "table" if _is_docling_table(chunk, chunk_text) else "text",
                        "pageNumber": (page_number + page_offset) if isinstance(page_number, int) else None,
                        "chunker": chunker.__class__.__name__,
                    }
                )

            if chunks_out:
                return chunks_out
    except Exception as exc:
        # Any Docling parsing issue should gracefully fall back to splitter-based chunking.
        message = str(exc).lower()
        if "bad_alloc" in message and _resolve_docling_disable_after_bad_alloc():
            _docling_disabled_due_bad_alloc = True
            _logger.warning(
                "Docling reported bad_alloc. Disabling Docling for this process and switching to fallback chunking.",
                exc_info=True,
            )
        else:
            _logger.warning("Docling conversion failed. Falling back to splitter-based chunking.", exc_info=True)
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return []


def _dedupe_structural_chunks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        content = _clean_text(str(item.get("content") or ""))
        if not content:
            continue
        page_number = item.get("pageNumber")
        key = f"{_normalize_for_match(content)}::{page_number}::{item.get('chunkType') or 'text'}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _docling_structural_chunks_batched(
    binary: bytes,
    total_pages: int,
    deadline_monotonic: float | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    if not _resolve_docling_enabled():
        return [], False

    if _docling_disabled_due_bad_alloc and _resolve_docling_disable_after_bad_alloc():
        return [], False

    batch_pages = _resolve_docling_batch_pages()
    if batch_pages <= 0 or total_pages <= 0 or total_pages <= batch_pages:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return [], True
        return _docling_structural_chunks(binary), False

    try:
        reader = PdfReader(io.BytesIO(binary))
    except Exception:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            return [], True
        return _docling_structural_chunks(binary), False

    all_chunks: list[dict[str, Any]] = []
    timed_out = False
    for start in range(0, total_pages, batch_pages):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            timed_out = True
            _logger.warning(
                "Docling batching reached soft timeout. Returning partial chunks. processed_pages=%s total_pages=%s",
                start,
                total_pages,
            )
            break

        end = min(total_pages, start + batch_pages)
        writer = PdfWriter()
        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])

        batch_io = io.BytesIO()
        writer.write(batch_io)
        batch_bytes = batch_io.getvalue()

        batch_chunks = _docling_structural_chunks(batch_bytes, page_offset=start)
        if batch_chunks:
            all_chunks.extend(batch_chunks)

        if _docling_disabled_due_bad_alloc and _resolve_docling_disable_after_bad_alloc():
            break

    return _dedupe_structural_chunks(all_chunks), timed_out


def _ocr_structural_chunks(
    binary: bytes,
    total_pages: int,
    deadline_monotonic: float | None = None,
    require_full: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    if not _resolve_pdf_ocr_fallback_enabled():
        return [], False

    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr import RapidOCR
    except Exception:
        _logger.warning(
            "OCR fallback dependencies are unavailable. Install pypdfium2 + rapidocr to enable scanned PDF ingestion."
        )
        return [], False

    try:
        doc = pdfium.PdfDocument(io.BytesIO(binary))
    except Exception:
        _logger.warning("Unable to open PDF for OCR fallback.", exc_info=True)
        return [], False

    max_pages = _resolve_pdf_ocr_max_pages()
    if max_pages > 0 and not require_full:
        total = min(total_pages, len(doc), max_pages)
    else:
        total = min(total_pages, len(doc))

    if total < min(total_pages, len(doc)):
        _logger.warning(
            "OCR fallback is limited by max pages. total_pages=%s, ocr_max_pages=%s",
            min(total_pages, len(doc)),
            max_pages,
        )

    render_scale = _resolve_pdf_ocr_render_scale()
    ocr_engine = RapidOCR()
    progress_every = _resolve_ocr_progress_every_pages()
    timed_out = False
    fallback_scales = [render_scale, 1.0, 0.85, 0.7]

    attempt_scales: list[float] = []
    for value in fallback_scales:
        rounded = round(float(value), 2)
        if rounded <= 0:
            continue
        if rounded not in attempt_scales:
            attempt_scales.append(rounded)

    _logger.warning(
        "Starting OCR fallback for scanned PDF. pages_to_process=%s total_pages=%s render_scale=%s",
        total,
        min(total_pages, len(doc)),
        render_scale,
    )

    out: list[dict[str, Any]] = []
    for page_index in range(total):
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            timed_out = True
            _logger.warning(
                "OCR fallback reached soft timeout. Returning partial chunks. processed_pages=%s total_pages=%s",
                page_index,
                total,
            )
            break

        page = doc[page_index]
        try:
            for scale_index, current_scale in enumerate(attempt_scales):
                bitmap = None
                image = None
                try:
                    bitmap = page.render(scale=current_scale)
                    image = bitmap.to_pil()
                    arr = np.array(image)
                    ocr_output = ocr_engine(arr)
                    page_lines = _extract_rapidocr_lines(ocr_output)

                    if not page_lines:
                        page_lines = _ocr_lines_with_stripes(ocr_engine, arr)

                    if page_lines:
                        out.append(
                            {
                                "content": "\n".join(page_lines),
                                "chunkType": "text",
                                "pageNumber": page_index + 1,
                                "chunker": "RapidOCRPage",
                            }
                        )
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    memory_pressure = isinstance(exc, MemoryError) or "bad allocation" in msg or "bad_alloc" in msg
                    can_retry_lower_scale = memory_pressure and scale_index < len(attempt_scales) - 1

                    if can_retry_lower_scale:
                        _logger.warning(
                            "OCR memory pressure on page. Retrying with lower render scale. page_number=%s scale=%s",
                            page_index + 1,
                            current_scale,
                        )
                        continue

                    if memory_pressure:
                        tile_lines = _ocr_lines_with_stripes(ocr_engine, arr)
                        if tile_lines:
                            out.append(
                                {
                                    "content": "\n".join(tile_lines),
                                    "chunkType": "text",
                                    "pageNumber": page_index + 1,
                                    "chunker": "RapidOCRPageTile",
                                }
                            )
                            break

                    _logger.warning("OCR fallback failed on page. page_number=%s", page_index + 1, exc_info=True)
                    break
                finally:
                    try:
                        if image is not None:
                            close_image = getattr(image, "close", None)
                            if callable(close_image):
                                close_image()
                        if bitmap is not None:
                            close_bitmap = getattr(bitmap, "close", None)
                            if callable(close_bitmap):
                                close_bitmap()
                    except Exception:
                        pass

            if progress_every > 0 and ((page_index + 1) % progress_every == 0 or page_index + 1 == total):
                _logger.warning(
                    "OCR fallback progress: %s/%s pages processed, chunks=%s",
                    page_index + 1,
                    total,
                    len(out),
                )
        finally:
            try:
                page.close()
            except Exception:
                pass

    try:
        doc.close()
    except Exception:
        pass

    return out, timed_out


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
    verify = _resolve_http_verify()

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            verify=verify,
            trust_env=True,
        ) as client:
            response = await client.get(source_url)
            response.raise_for_status()
            return response.content
    except httpx.ConnectError as exc:
        message = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in message:
            if verify is not False and _resolve_ssl_allow_insecure_fallback():
                _logger.warning(
                    "Retrying planning document download with SSL verify disabled due to certificate verification failure. source_url=%s",
                    source_url,
                )
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=True,
                    verify=False,
                    trust_env=True,
                ) as retry_client:
                    retry_response = await retry_client.get(source_url)
                    retry_response.raise_for_status()
                    return retry_response.content

            raise RuntimeError(
                "Failed to download planning document due to SSL certificate verification. "
                "Set PLANNING_INGEST_CA_BUNDLE to a trusted CA bundle path, or only for local/dev set "
                "PLANNING_INGEST_SSL_VERIFY=false. "
                "Optionally allow automatic local/dev retry by setting PLANNING_INGEST_SSL_ALLOW_INSECURE_FALLBACK=true."
            ) from exc
        raise


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
    force_docling_ocr = fmt == "pdf" and not cleaned

    start_monotonic = time.monotonic()
    soft_timeout_seconds = _resolve_ingest_soft_timeout_seconds()
    require_full = _resolve_ingest_require_full()
    docling_enabled = _resolve_docling_enabled()
    scanned_pdf_use_docling_first = _resolve_scan_pdf_use_docling_first()
    if require_full and soft_timeout_seconds > 0:
        _logger.warning(
            "PLANNING_INGEST_REQUIRE_FULL=true ignores soft timeout. configured_timeout_seconds=%s",
            soft_timeout_seconds,
        )
    deadline_monotonic = (start_monotonic + soft_timeout_seconds) if (soft_timeout_seconds > 0 and not require_full) else None

    _logger.warning(
        "Planning ingest effective config. planning_document_id=%s require_full=%s docling_enabled=%s scanned_pdf_use_docling_first=%s ocr_fallback_enabled=%s soft_timeout_seconds=%s ocr_max_pages=%s",
        payload.planning_document_id,
        require_full,
        docling_enabled,
        scanned_pdf_use_docling_first,
        _resolve_pdf_ocr_fallback_enabled(),
        soft_timeout_seconds,
        _resolve_pdf_ocr_max_pages(),
    )

    page_maps = _build_page_line_maps(page_texts)
    total_pages = len(page_maps)
    timed_out = False

    structural_chunks: list[dict[str, Any]] = []
    if fmt == "pdf":
        max_docling_pages = _resolve_docling_max_pages()
        max_docling_pdf_bytes = _resolve_docling_max_pdf_bytes()
        is_large_by_pages = max_docling_pages > 0 and total_pages > max_docling_pages
        is_large_by_bytes = max_docling_pdf_bytes > 0 and len(source_bytes) > max_docling_pdf_bytes

        if force_docling_ocr:
            if docling_enabled and scanned_pdf_use_docling_first:
                _logger.warning(
                    "PDF has no text layer from pypdf; forcing Docling OCR path. total_pages=%s, pdf_bytes=%s",
                    total_pages,
                    len(source_bytes),
                )
                structural_chunks, timed_out = _docling_structural_chunks_batched(
                    source_bytes,
                    total_pages,
                    deadline_monotonic=deadline_monotonic,
                )
            else:
                _logger.warning(
                    "PDF has no text layer from pypdf; bypassing Docling and using OCR fallback path. total_pages=%s, pdf_bytes=%s docling_enabled=%s scanned_pdf_use_docling_first=%s",
                    total_pages,
                    len(source_bytes),
                    docling_enabled,
                    scanned_pdf_use_docling_first,
                )

        elif is_large_by_pages:
            _logger.warning(
                "Skipping Docling for large PDF and using fallback chunking. total_pages=%s, max_docling_pages=%s",
                total_pages,
                max_docling_pages,
            )
        elif is_large_by_bytes:
            _logger.warning(
                "Skipping Docling for large PDF payload and using fallback chunking. pdf_bytes=%s, max_docling_pdf_bytes=%s",
                len(source_bytes),
                max_docling_pdf_bytes,
            )
        else:
            structural_chunks, timed_out = _docling_structural_chunks_batched(
                source_bytes,
                total_pages,
                deadline_monotonic=deadline_monotonic,
            )
            if structural_chunks and not _is_docling_output_reliable(structural_chunks, total_pages):
                _logger.warning(
                    "Docling output considered unreliable; switching to fallback chunking. total_pages=%s, docling_chunks=%s",
                    total_pages,
                    len(structural_chunks),
                )
                structural_chunks = []

    if not structural_chunks:
        structural_chunks = _fallback_structural_chunks(cleaned, page_maps)

    if fmt == "pdf" and not structural_chunks and force_docling_ocr:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            timed_out = True
            _logger.warning("Skipping OCR fallback due to elapsed soft timeout before OCR start.")
        else:
            _logger.warning(
                "No extractable text from PDF + Docling unavailable/failed. Trying OCR fallback. total_pages=%s",
                total_pages,
            )
            structural_chunks, ocr_timed_out = _ocr_structural_chunks(
                source_bytes,
                total_pages,
                deadline_monotonic=deadline_monotonic,
                require_full=require_full,
            )
            timed_out = timed_out or ocr_timed_out

    if timed_out:
        _logger.warning(
            "Planning ingest reached soft timeout. Returning available chunks only. planning_document_id=%s",
            payload.planning_document_id,
        )

    if not structural_chunks and not cleaned:
        if require_full:
            raise RuntimeError(
                "Failed to extract content from scanned PDF after all OCR strategies. "
                "Try reducing OCR render scale (PLANNING_PDF_OCR_RENDER_SCALE), or process with higher memory."
            )
        elapsed_ms = int((time.monotonic() - start_monotonic) * 1000)
        _logger.warning(
            "Planning document extraction completed with no content. planning_document_id=%s format=%s pages=%s timed_out=%s elapsed_ms=%s",
            payload.planning_document_id,
            fmt,
            total_pages,
            timed_out,
            elapsed_ms,
        )
        return [], {"textChunks": 0, "tableChunks": 0, "timedOut": 1 if timed_out else 0}

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

    elapsed_ms = int((time.monotonic() - start_monotonic) * 1000)
    _logger.info(
        "Planning document extraction completed. planning_document_id=%s format=%s pages=%s chunks=%s text_chunks=%s table_chunks=%s timed_out=%s elapsed_ms=%s",
        payload.planning_document_id,
        fmt,
        total_pages,
        len(docs),
        text_count,
        table_count,
        timed_out,
        elapsed_ms,
    )

    return docs, {"textChunks": text_count, "tableChunks": table_count, "timedOut": 1 if timed_out else 0}
