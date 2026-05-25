from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.documents import Document
from pypdf import PdfReader

from .ingestion_config import (
    PLANNING_CHUNKING_BASELINE_FIXED,
    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD,
    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT,
    is_hierarchical_chunking_mode as _is_hierarchical_chunking_mode,
    resolve_http_verify as _resolve_http_verify,
    resolve_ingest_require_full as _resolve_ingest_require_full,
    resolve_ingest_soft_timeout_seconds as _resolve_ingest_soft_timeout_seconds,
    resolve_ocr_progress_every_pages as _resolve_ocr_progress_every_pages,
    resolve_pdf_force_ocr_on_low_quality as _resolve_pdf_force_ocr_on_low_quality,
    resolve_pdf_ocr_fallback_enabled as _resolve_pdf_ocr_fallback_enabled,
    resolve_pdf_ocr_max_pages as _resolve_pdf_ocr_max_pages,
    resolve_pdf_ocr_render_scale as _resolve_pdf_ocr_render_scale,
    resolve_pdf_text_quality_min_score as _resolve_pdf_text_quality_min_score,
    resolve_planning_chunking_mode as _resolve_planning_chunking_mode,
    resolve_ssl_allow_insecure_fallback as _resolve_ssl_allow_insecure_fallback,
)
from .metadata import canonicalize_planning_district
from ..utils.chunking import build_splitter

_splitter = None
_logger = logging.getLogger(__name__)


def _get_splitter():
    global _splitter
    if _splitter is None:
        _splitter = build_splitter(chunk_size=1500, chunk_overlap=120)
    return _splitter


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

    cleaned_lines: list[str] = []
    for raw_line in txts:
        if not isinstance(raw_line, str):
            continue
        cleaned = _clean_text(raw_line)
        if cleaned:
            cleaned_lines.append(cleaned)

    return cleaned_lines


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


def _extract_pdf_text_and_pages(binary: bytes) -> tuple[str, list[str]]:
    reader = PdfReader(io.BytesIO(binary))
    raw_pages: list[str] = []
    cleaned_pages: list[str] = []

    for page in reader.pages:
        raw_text = page.extract_text() or ""
        raw_pages.append(raw_text)
        cleaned_pages.append(_clean_text(raw_text))

    return "\n\n".join(raw_pages).strip(), cleaned_pages


def _clean_text(text: str) -> str:
    value = re.sub(r"\u0000", "", text or "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _normalize_for_match(text: str) -> str:
    value = _clean_text(text).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _estimate_pdf_text_quality(page_texts: list[str], cleaned_text: str) -> float:
    if not page_texts and not cleaned_text:
        return 0.0

    total_pages = max(1, len(page_texts))
    non_empty_pages = sum(1 for page in page_texts if len(_clean_text(page)) >= 40)
    coverage_score = min(1.0, non_empty_pages / total_pages)

    sample = (cleaned_text or "")[:120000]
    if not sample:
        return round(max(0.0, min(coverage_score * 0.25, 1.0)), 3)

    non_space_chars = max(1, sum(1 for ch in sample if not ch.isspace()))
    alpha_ratio = sum(1 for ch in sample if ch.isalpha()) / non_space_chars
    replacement_ratio = sample.count("\ufffd") / max(1, len(sample))
    control_ratio = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\n\r\t") / max(1, len(sample))

    avg_chars_per_page = len(cleaned_text) / max(1, total_pages)
    density_score = min(1.0, avg_chars_per_page / 900.0)
    alpha_score = min(1.0, alpha_ratio / 0.45)
    noise_penalty = min(1.0, (replacement_ratio * 18.0) + (control_ratio * 30.0))

    score = (coverage_score * 0.35) + (density_score * 0.35) + (alpha_score * 0.30)
    score *= (1.0 - (noise_penalty * 0.60))

    return round(max(0.0, min(score, 1.0)), 3)


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


def _looks_like_continuation_lead(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False

    if re.match(r"^(?:[-+*]|\d+[\.)]?)\s*", normalized):
        return True
    if re.match(r"^(?:tong|trong do|cu the|gom|la|dat|du an|cong trinh|ha)\b", normalized):
        return True
    return False


def _ends_with_continuation_signal(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False

    return any(
        normalized.endswith(marker)
        for marker in (
            "la",
            "gom",
            "gom:",
            "cu the",
            "cu the:",
            "trong do",
            "trong do:",
        )
    ) or normalized.endswith(":")


def _is_structurally_weak_chunk(item: dict[str, Any]) -> bool:
    content = _clean_text(str(item.get("content") or ""))
    if not content:
        return True

    normalized = _normalize_for_match(content)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return True

    if len(lines) == 1 and re.fullmatch(r"\d+", normalized):
        return True

    if len(lines) <= 2 and len(normalized.split()) <= 6:
        if not re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty)\b", normalized):
            if not any(marker in normalized for marker in ("du an", "cong trinh", "thu hoi", "quyet dinh", "dieu ")):
                return True

    return False


def _merge_continuation_chunks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []

    merged: list[dict[str, Any]] = []
    for item in items:
        content = _clean_text(str(item.get("content") or ""))
        if not content:
            continue

        current = dict(item)
        current["content"] = content

        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        prev_content = _clean_text(str(previous.get("content") or ""))
        prev_norm = _normalize_for_match(prev_content)
        current_norm = _normalize_for_match(content)
        if not prev_norm or not current_norm:
            merged.append(current)
            continue

        prev_chunk_type = previous.get("chunkType") or ("table" if _is_table_chunk(prev_content) else "text")
        current_chunk_type = current.get("chunkType") or ("table" if _is_table_chunk(content) else "text")
        same_chunk_type = prev_chunk_type == current_chunk_type == "text"
        same_page = (
            previous.get("pageNumber") is not None
            and previous.get("pageNumber") == current.get("pageNumber")
        )
        same_section = bool(previous.get("sectionHeading")) and previous.get("sectionHeading") == current.get("sectionHeading")
        same_hierarchy = bool(previous.get("hierarchyPath")) and previous.get("hierarchyPath") == current.get("hierarchyPath")
        current_short = len(content) <= 260 or len(content.splitlines()) <= 4
        current_numeric = re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty)\b", current_norm) is not None
        prev_missing_numeric = re.search(r"\b\d+(?:[\.,]\d+)?\s*(?:ha|m2|%|ty)\b", prev_norm) is None

        should_merge = (
            same_chunk_type
            and (same_page or same_section or same_hierarchy)
            and (
                _ends_with_continuation_signal(prev_content)
                or _looks_like_continuation_lead(lines := content.splitlines()[0])
                or (current_short and current_numeric and prev_missing_numeric)
            )
        )

        if not should_merge:
            merged.append(current)
            continue

        previous["content"] = _clean_text(f"{prev_content}\n{content}")
        previous["chunker"] = f"{previous.get('chunker') or 'unknown'}+ContinuationMerge"
        if previous.get("pageNumber") is None:
            previous["pageNumber"] = current.get("pageNumber")
        if not previous.get("sectionHeading"):
            previous["sectionHeading"] = current.get("sectionHeading")
        if not previous.get("hierarchyPath"):
            previous["hierarchyPath"] = current.get("hierarchyPath")

    return merged


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


def _hierarchy_line_level(line: str) -> int | None:
    compact = line.strip()
    if not compact:
        return None

    normalized = _normalize_for_match(compact)

    # Top hierarchy (Phần/Chương/A,B,C...)
    if re.match(r"^(phan|phần)\s+[ivxlcdm]+\b", normalized):
        return 1
    if re.match(r"^(chuong|chương)\s+[ivxlcdm0-9]+\b", normalized):
        return 1
    if re.match(r"^[A-ZĐ]\s+\S+", compact):
        return 1

    # Mid hierarchy (Mục / Roman numerals)
    if re.match(r"^(muc|mục)\s+\d+(?:\.\d+){0,3}\b", normalized):
        return 2
    if re.match(r"^[IVXLCDM]+\s+\S+", compact):
        return 2

    # Lower hierarchy (3.1, 3.1.2)
    if re.match(r"^\d+\.\d+(?:\.\d+){0,3}\b", normalized):
        return 3

    # Enumeration under active sections (1. / 1) / a. / a))
    if re.match(r"^\d+[\.)]\s+\S+", normalized):
        return 4
    if re.match(r"^[a-zđ][\.)]\s+\S+", normalized):
        return 5

    return None


def _is_new_item_marker(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False

    return bool(re.match(r"^(\d+[\.)]|[a-zđ][\.)]|[ivxlcdm]+[\.)])\s+\S+", normalized))


def _is_projectish_line(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False

    if any(token in normalized for token in ("du an", "cong trinh", "thu hoi", "dau gia", "gpmb")):
        return True

    if _looks_like_table_line(line):
        return True

    if re.search(r"\b\d+(?:[\.,]\d+)?\s*(ha|m2|m²)\b", normalized):
        return True

    return False


def _is_project_table_header_line(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if not normalized:
        return False

    markers = (
        "tt",
        "stt",
        "danh muc",
        "cong trinh",
        "du an",
        "ma loai dat",
        "dien tich ke hoach",
        "dien tich thu hoi",
        "dia danh",
        "phuong",
        "can cu phap ly",
        "ghi chu",
    )
    hit = sum(1 for marker in markers if marker in normalized)
    if hit >= 3:
        return True

    return bool(re.match(r"^(tt|stt)\b", normalized) and hit >= 2)


def _hierarchy_contextual_chunks(page_maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not page_maps:
        return []

    active_hierarchy: dict[int, str] = {}
    chunks: list[dict[str, Any]] = []

    for page in page_maps:
        page_number = page.get("pageNumber")
        lines: list[str] = page.get("lines") or []
        i = 0

        while i < len(lines):
            line = _clean_text(lines[i])
            if not line:
                i += 1
                continue

            if _is_project_table_header_line(line):
                i += 1
                continue

            level = _hierarchy_line_level(line)
            if level is not None:
                active_hierarchy[level] = line
                for deeper in [k for k in active_hierarchy if k > level]:
                    active_hierarchy.pop(deeper, None)
                i += 1
                continue

            block = [line]
            j = i + 1
            while j < len(lines):
                next_line = _clean_text(lines[j])
                if not next_line:
                    j += 1
                    continue

                if _hierarchy_line_level(next_line) is not None:
                    break

                if _is_new_item_marker(next_line) and len(block) >= 2 and _is_projectish_line(block[0]):
                    break

                block.append(next_line)
                if len(block) >= 14 or len("\n".join(block)) >= 1700:
                    break
                j += 1

            body = _clean_text("\n".join(block))
            if body:
                hierarchy_levels = sorted(active_hierarchy.keys())
                hierarchy_lines = [f"[h{lvl}={active_hierarchy[lvl]}]" for lvl in hierarchy_levels]
                hierarchy_path = " > ".join(active_hierarchy[lvl] for lvl in hierarchy_levels)
                merged = _clean_text("\n".join([*hierarchy_lines, body]))
                if merged:
                    chunks.append(
                        {
                            "content": merged,
                            "chunkType": "table" if _is_projectish_line(body) else "text",
                            "pageNumber": page_number if isinstance(page_number, int) else None,
                            "chunker": "HierarchyContextParser",
                            "sectionHeading": active_hierarchy[max(active_hierarchy.keys())] if active_hierarchy else None,
                            "hierarchyPath": hierarchy_path or None,
                        }
                    )

            i = max(j, i + 1)

    return _dedupe_structural_chunks(chunks)


def _hierarchy_signature(path: str, page_number: int | None = None) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", _normalize_for_match(path or "root")).strip("_")
    if not compact:
        compact = "root"
    suffix = f"_p{page_number}" if page_number is not None else ""
    return f"planning_hierarchy_{compact[:96]}{suffix}"


def _hierarchical_chunk_content(
    body: str,
    hierarchy_lines: list[str],
    hierarchy_path: str,
    mode: str,
) -> str:
    if mode != PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT:
        return body

    context_lines = [f"[hierarchy_path={hierarchy_path}]"] if hierarchy_path else []
    context_lines.extend(f"[h{idx + 1}={line}]" for idx, line in enumerate(hierarchy_lines))
    return _clean_text("\n".join([*context_lines, body]))


def _append_hierarchical_parent_chunks(
    chunks: list[dict[str, Any]],
    emitted_parent_ids: set[str],
    active_hierarchy: dict[int, str],
    *,
    page_number: int | None,
    sibling_index: int,
) -> str | None:
    if not active_hierarchy:
        return None

    levels = sorted(active_hierarchy.keys())
    parent_level = levels[-1]
    hierarchy_path = " > ".join(active_hierarchy[level] for level in levels)
    parent_id = _hierarchy_signature(hierarchy_path, page_number)
    if parent_id not in emitted_parent_ids:
        emitted_parent_ids.add(parent_id)
        parent_content = _clean_text(
            "\n".join(
                [
                    f"[hierarchy_path={hierarchy_path}]",
                    *[f"[h{level}={active_hierarchy[level]}]" for level in levels],
                ]
            )
        )
        if parent_content:
            chunks.append(
                {
                    "content": parent_content,
                    "chunkType": "text",
                    "pageNumber": page_number,
                    "chunker": "PlanningHierarchicalParentChunk",
                    "sectionHeading": active_hierarchy.get(parent_level),
                    "hierarchyPath": hierarchy_path,
                    "hierarchyLevel": parent_level,
                    "parentChunkId": None,
                    "siblingIndex": sibling_index,
                    "isParentChunk": True,
                }
            )

    return parent_id


def build_planning_hierarchical_chunks(
    text: str,
    page_maps: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """Build planning chunks by document hierarchy instead of fixed text length."""
    if mode not in _PLANNING_HIERARCHICAL_MODES:
        return []

    entries = _flatten_page_entries(page_maps)
    if not entries:
        entries = [(None, line) for line in text.splitlines() if _clean_text(line)]
    if not entries:
        return []

    active_hierarchy: dict[int, str] = {}
    active_table_header: list[str] = []
    chunks: list[dict[str, Any]] = []
    emitted_parent_ids: set[str] = set()
    sibling_counters: dict[str, int] = {}

    def _current_context(page_number: int | None) -> tuple[list[int], list[str], str, str | None, int, int]:
        levels = sorted(active_hierarchy.keys())
        hierarchy_lines = [active_hierarchy[level] for level in levels]
        hierarchy_path = " > ".join(hierarchy_lines)
        hierarchy_level = levels[-1] if levels else 0
        section_heading = active_hierarchy.get(hierarchy_level) if levels else None
        parent_id = (
            _append_hierarchical_parent_chunks(
                chunks,
                emitted_parent_ids,
                active_hierarchy,
                page_number=page_number,
                sibling_index=len(emitted_parent_ids),
            )
            if mode == PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD
            else None
        )
        counter_key = parent_id or hierarchy_path or "root"
        sibling_counters[counter_key] = sibling_counters.get(counter_key, 0) + 1
        return levels, hierarchy_lines, hierarchy_path, section_heading, hierarchy_level, sibling_counters[counter_key] - 1

    def _append_body_chunk(
        body_lines: list[str],
        *,
        page_number: int | None,
        chunk_type: str | None = None,
    ) -> None:
        clean_lines = [_clean_text(line) for line in body_lines if _clean_text(line)]
        if not clean_lines:
            return

        levels, hierarchy_lines, hierarchy_path, section_heading, hierarchy_level, sibling_index = _current_context(page_number)
        body = _clean_text("\n".join(clean_lines))
        if not body:
            return

        parent_id = _hierarchy_signature(hierarchy_path, page_number) if (
            mode == PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD and hierarchy_path
        ) else None
        content = _hierarchical_chunk_content(body, hierarchy_lines, hierarchy_path, mode)
        chunks.append(
            {
                "content": content,
                "chunkType": chunk_type or ("table" if _is_projectish_line(body) else "text"),
                "pageNumber": page_number,
                "chunker": {
                    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT: "PlanningHierarchicalParentContext",
                    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD: "PlanningHierarchicalParentChild",
                }[mode],
                "sectionHeading": section_heading,
                "hierarchyPath": hierarchy_path or None,
                "hierarchyLevel": hierarchy_level,
                "parentChunkId": parent_id,
                "siblingIndex": sibling_index,
                "isParentChunk": False,
            }
        )

    body_buffer: list[str] = []
    body_page: int | None = None

    def _flush_body() -> None:
        nonlocal body_buffer, body_page
        if body_buffer:
            _append_body_chunk(body_buffer, page_number=body_page)
        body_buffer = []
        body_page = None

    for page_number, raw_line in entries:
        line = _clean_text(raw_line)
        if not line:
            continue

        if _is_project_table_header_line(line):
            _flush_body()
            active_table_header = [line]
            continue

        raw_level = _hierarchy_line_level(line)
        level = raw_level if raw_level is not None and raw_level <= 3 else None
        if (
            level is None
            and _is_section_heading_line(line)
            and not _is_new_item_marker(line)
            and not _looks_like_table_line(line)
            and not _is_projectish_line(line)
        ):
            level = 3

        if level is not None:
            _flush_body()
            active_hierarchy[level] = line
            for deeper in [key for key in active_hierarchy if key > level]:
                active_hierarchy.pop(deeper, None)
            active_table_header = []
            continue

        if active_table_header and (_looks_like_table_line(line) or _is_projectish_line(line)):
            _flush_body()
            _append_body_chunk([*active_table_header, line], page_number=page_number, chunk_type="table")
            continue

        if body_buffer and _is_new_item_marker(line):
            _flush_body()

        if body_page is None:
            body_page = page_number
        body_buffer.append(line)

    _flush_body()

    meaningful_chunks = [
        item
        for item in chunks
        if item.get("isParentChunk") or not _is_structurally_weak_chunk(item)
    ]
    return _dedupe_structural_chunks(meaningful_chunks)


def _flatten_page_entries(page_maps: list[dict[str, Any]]) -> list[tuple[int | None, str]]:
    entries: list[tuple[int | None, str]] = []
    for page in page_maps:
        page_number = page.get("pageNumber") if isinstance(page.get("pageNumber"), int) else None
        for raw_line in page.get("lines") or []:
            line = _clean_text(raw_line)
            if not line:
                continue
            entries.append((page_number, line))
    return entries


def _detect_document_archetype(text: str, page_maps: list[dict[str, Any]]) -> str:
    entries = _flatten_page_entries(page_maps)
    sample_lines = [line for _, line in entries[:700]]
    sample_blob = "\n".join(sample_lines) if sample_lines else text
    sample_norm = _normalize_for_match(sample_blob[:140000])

    if not sample_norm:
        return "narrative_report"

    has_quyet_dinh = "quyet dinh" in sample_norm
    can_cu_count = sample_norm.count("can cu")
    article_count = len(re.findall(r"\bdieu\s+\d+\b", sample_norm))
    if has_quyet_dinh and can_cu_count >= 2 and article_count >= 1:
        return "decision"

    table_markers = (
        "danh muc cong trinh",
        "du an",
        "ma loai dat",
        "dien tich ke hoach",
        "dien tich thu hoi",
        "can cu phap ly",
        "ghi chu",
        "hdnd thanh pho",
    )
    table_hits = sum(1 for marker in table_markers if marker in sample_norm)
    roman_headers = sum(1 for line in sample_lines if re.match(r"^[IVXLCDM]+\s+\S+", line))
    alpha_headers = sum(1 for line in sample_lines if re.match(r"^[A-ZĐ]\s+\S+", line))
    letter_subgroups = sum(1 for line in sample_lines if re.match(r"^[a-zđ][\.)]\s+\S+", _normalize_for_match(line)))
    numeric_items = sum(1 for line in sample_lines if re.match(r"^\d+[\.)]\s+\S+", _normalize_for_match(line)))
    project_rows = sum(1 for line in sample_lines if _is_projectish_line(line) and (re.match(r"^\d+", _normalize_for_match(line)) is not None or " du an " in f" {_normalize_for_match(line)} "))
    table_header_lines = sum(1 for line in sample_lines if _is_project_table_header_line(line))
    hierarchy_levels = sum(1 for line in sample_lines if _hierarchy_line_level(line) is not None)

    hierarchy_table_like = (
        alpha_headers >= 1
        and roman_headers >= 1
        and (letter_subgroups >= 1 or table_header_lines >= 1)
        and project_rows >= 2
    )

    if hierarchy_table_like or (table_hits >= 2 and hierarchy_levels >= 5 and numeric_items >= 2):
        return "hierarchical_table"

    return "narrative_report"


def _decision_structural_chunks(page_maps: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    entries = [item for item in _flatten_page_entries(page_maps) if not _is_project_table_header_line(item[1])]
    if not entries:
        entries = [(None, line) for line in text.splitlines() if _clean_text(line)]

    if not entries:
        return []

    norms = [_normalize_for_match(line) for _, line in entries]
    total = len(entries)
    article_starts = [idx for idx, norm in enumerate(norms) if re.match(r"^dieu\s+\d+\b", norm)]
    first_article = article_starts[0] if article_starts else None
    article_end = total

    if first_article is not None:
        for idx in range(first_article, total):
            norm = norms[idx]
            if norm.startswith("noi nhan") or norm.startswith("tm.") or norm.startswith("kt."):
                article_end = idx
                break

    chunks: list[dict[str, Any]] = []

    def _append_chunk(block: list[tuple[int | None, str]], section: str, heading: str | None = None) -> None:
        if not block:
            return
        body = _clean_text("\n".join(line for _, line in block))
        if not body:
            return

        tags = [f"[decision_section={section}]"]
        if heading:
            tags.append(f"[decision_heading={heading}]")
        tagged = _clean_text("\n".join([*tags, body]))
        pieces = [tagged] if len(tagged) <= 2200 else _get_splitter().split_text(tagged)

        for piece in pieces:
            clean_piece = _clean_text(piece)
            if not clean_piece:
                continue
            chunks.append(
                {
                    "content": clean_piece,
                    "chunkType": "text",
                    "pageNumber": block[0][0],
                    "chunker": "DecisionStructureParser",
                    "sectionHeading": heading or section,
                }
            )

    legal_starts = [
        idx
        for idx, norm in enumerate(norms)
        if norm.startswith("can cu") and (first_article is None or idx < first_article)
    ]
    legal_ranges: list[tuple[int, int]] = []
    for pos, start in enumerate(legal_starts):
        end = legal_starts[pos + 1] if (pos + 1) < len(legal_starts) else (first_article if first_article is not None else total)
        legal_ranges.append((start, end))
        heading = entries[start][1][:140]
        _append_chunk(entries[start:end], section="legal_basis", heading=heading)

    front_end = first_article if first_article is not None else total
    legal_indices: set[int] = set()
    for start, end in legal_ranges:
        legal_indices.update(range(start, end))
    front_block = [entries[idx] for idx in range(0, front_end) if idx not in legal_indices]
    _append_chunk(front_block, section="front_matter", heading="Quyet dinh")

    if first_article is not None:
        article_span = [idx for idx in article_starts if idx < article_end]
        for pos, start in enumerate(article_span):
            end = article_span[pos + 1] if (pos + 1) < len(article_span) else article_end
            heading = entries[start][1][:180]
            _append_chunk(entries[start:end], section="article", heading=heading)

    if article_end < total:
        _append_chunk(entries[article_end:], section="closing", heading="Noi nhan")

    return _dedupe_structural_chunks(chunks)


def _section_table_fallback_chunks(text: str, page_maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_blocks, table_blocks = _split_text_and_table_blocks(text)
    chunks: list[dict[str, Any]] = []

    for block in text_blocks:
        section_items = _split_text_sections(block)
        for section in section_items:
            heading = section.get("heading")
            section_text = _clean_text(str(section.get("content") or ""))
            if not section_text:
                continue

            pieces = [section_text] if len(section_text) <= 1900 else _get_splitter().split_text(section_text)
            for index, piece in enumerate(pieces):
                clean_chunk = _clean_text(piece)
                if not clean_chunk:
                    continue

                if heading and index > 0:
                    heading_norm = _normalize_for_match(heading)
                    prefix_norm = _normalize_for_match(clean_chunk[: max(160, len(heading) + 20)])
                    if heading_norm and heading_norm not in prefix_norm:
                        clean_chunk = _clean_text(f"{heading}\n{clean_chunk}")

                chunks.append(
                    {
                        "content": clean_chunk,
                        "chunkType": "text",
                        "pageNumber": None,
                        "chunker": "SectionAwareSplitter",
                        "sectionHeading": heading,
                    }
                )

    for block in table_blocks:
        row_chunks = _chunk_table_block_rows(block, rows_per_chunk=4)
        for row_chunk in row_chunks:
            pieces = [row_chunk] if len(row_chunk) <= 2100 else _get_splitter().split_text(row_chunk)
            for piece in pieces:
                clean_chunk = _clean_text(piece)
                if clean_chunk:
                    chunks.append(
                        {
                            "content": clean_chunk,
                            "chunkType": "table",
                            "pageNumber": None,
                            "chunker": "TableRowSplitter",
                            "sectionHeading": None,
                        }
                    )

    if chunks:
        return chunks

    for page in page_maps:
        for chunk in _get_splitter().split_text("\n".join(page["lines"])):
            clean_chunk = _clean_text(chunk)
            if clean_chunk:
                chunks.append(
                    {
                        "content": clean_chunk,
                        "chunkType": "text",
                        "pageNumber": page["pageNumber"],
                        "chunker": "RecursiveCharacterTextSplitter",
                        "sectionHeading": None,
                    }
                )

    return chunks


def _fallback_structural_chunks(text: str, page_maps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    archetype = _detect_document_archetype(text, page_maps)

    if archetype == "decision":
        chunks = _decision_structural_chunks(page_maps, text)
        if chunks:
            return chunks
        return _section_table_fallback_chunks(text, page_maps)

    if archetype == "hierarchical_table":
        chunks = _hierarchy_contextual_chunks(page_maps)
        if chunks:
            return chunks
        return _section_table_fallback_chunks(text, page_maps)

    chunks = _section_table_fallback_chunks(text, page_maps)
    if chunks:
        return chunks

    return _hierarchy_contextual_chunks(page_maps)


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


def _is_section_heading_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    normalized = _normalize_for_match(compact)
    if re.match(r"^(chuong|phu luc|muc|phan|dieu)\b", normalized):
        return True

    if re.match(r"^\d+(?:\.\d+){0,3}\b", normalized):
        return True

    alpha_chars = [ch for ch in compact if ch.isalpha()]
    if len(alpha_chars) < 8:
        return False

    uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / max(1, len(alpha_chars))
    return uppercase_ratio >= 0.78 and len(compact) <= 150


def _split_text_sections(block: str) -> list[dict[str, str | None]]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return []

    sections: list[dict[str, str | None]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_heading, current_lines
        if not current_heading and not current_lines:
            return

        merged_lines: list[str] = []
        if current_heading:
            merged_lines.append(current_heading)
        merged_lines.extend(current_lines)

        content = _clean_text("\n".join(merged_lines))
        if content:
            sections.append({"heading": current_heading, "content": content})

        current_heading = None
        current_lines = []

    for line in lines:
        if _is_section_heading_line(line):
            _flush()
            current_heading = line
            continue
        current_lines.append(line)

    _flush()

    if not sections:
        fallback_content = _clean_text(block)
        if fallback_content:
            return [{"heading": None, "content": fallback_content}]
        return []

    return sections


def _chunk_table_block_rows(block: str, rows_per_chunk: int = 4) -> list[str]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return []

    header_lines: list[str] = []
    row_lines: list[str] = []
    header_markers = ("stt", "noi dung", "chi tieu", "don vi", "ghi chu", "tong cong", "ma")

    for index, line in enumerate(lines):
        normalized = _normalize_for_match(line)
        looks_header = any(marker in normalized for marker in header_markers)
        if index < 2 and (looks_header or not _looks_like_table_line(line)):
            header_lines.append(line)
            continue
        row_lines.append(line)

    if not row_lines:
        row_lines = lines
        header_lines = []

    out: list[str] = []
    stride = max(1, int(rows_per_chunk))
    for start in range(0, len(row_lines), stride):
        window = row_lines[start : start + stride]
        chunk_text = _clean_text("\n".join([*header_lines, *window]))
        if chunk_text:
            out.append(chunk_text)

    return out


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
    descriptor_parts = [
        f"document_type:{metadata.get('documentType') or 'unknown'}",
        f"district:{metadata.get('district') or 'unknown'}",
        f"plan_year:{metadata.get('planYear') or 'unknown'}",
        f"chunk_type:{metadata.get('chunkType') or 'text'}",
    ]

    title = str(metadata.get("title") or "").strip()
    if title:
        descriptor_parts.append(f"title:{title[:140]}")

    descriptor = " | ".join(descriptor_parts)
    return f"{descriptor}\n{content}" if descriptor else content


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
    chunking_mode = _resolve_planning_chunking_mode()
    source_bytes = await fetch_document_bytes(payload.source_url)

    fmt = (payload.format or "").lower().strip(".")
    if not fmt:
        fmt = "pdf" if payload.source_url.lower().endswith(".pdf") else "txt"

    if fmt == "pdf":
        raw_text, page_texts = _extract_pdf_text_and_pages(source_bytes)
    else:
        raw_text = source_bytes.decode("utf-8", errors="ignore")
        page_texts = [_clean_text(raw_text)] if raw_text else []

    cleaned = _clean_text(raw_text)
    pdf_text_quality_score: float | None = None
    low_quality_pdf_text = False

    force_ocr_extract = fmt == "pdf" and not cleaned
    force_ocr_reason = "no_text_layer" if force_ocr_extract else None

    if fmt == "pdf":
        pdf_text_quality_score = _estimate_pdf_text_quality(page_texts, cleaned)
        quality_threshold = _resolve_pdf_text_quality_min_score()
        force_ocr_on_low_quality = _resolve_pdf_force_ocr_on_low_quality()

        low_quality_pdf_text = (
            bool(cleaned)
            and force_ocr_on_low_quality
            and pdf_text_quality_score < quality_threshold
        )

        if low_quality_pdf_text:
            force_ocr_extract = True
            force_ocr_reason = "low_text_quality"

    start_monotonic = time.monotonic()
    soft_timeout_seconds = _resolve_ingest_soft_timeout_seconds()
    require_full = _resolve_ingest_require_full()
    if require_full and soft_timeout_seconds > 0:
        _logger.warning(
            "PLANNING_INGEST_REQUIRE_FULL=true ignores soft timeout. configured_timeout_seconds=%s",
            soft_timeout_seconds,
        )
    deadline_monotonic = (start_monotonic + soft_timeout_seconds) if (soft_timeout_seconds > 0 and not require_full) else None

    _logger.warning(
        "Planning ingest effective config. planning_document_id=%s chunking_mode=%s require_full=%s ocr_fallback_enabled=%s soft_timeout_seconds=%s ocr_max_pages=%s pdf_text_quality_score=%s force_ocr_extract=%s force_reason=%s",
        payload.planning_document_id,
        chunking_mode,
        require_full,
        _resolve_pdf_ocr_fallback_enabled(),
        soft_timeout_seconds,
        _resolve_pdf_ocr_max_pages(),
        pdf_text_quality_score,
        force_ocr_extract,
        force_ocr_reason,
    )

    page_maps = _build_page_line_maps(page_texts)
    total_pages = len(page_maps)
    timed_out = False

    structural_chunks: list[dict[str, Any]] = []
    if fmt == "pdf" and force_ocr_extract:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            timed_out = True
            _logger.warning("Skipping OCR fallback due to elapsed soft timeout before OCR start.")
        else:
            _logger.warning(
                "Triggering OCR fallback for PDF extraction. reason=%s total_pages=%s",
                force_ocr_reason,
                total_pages,
            )
            structural_chunks, ocr_timed_out = _ocr_structural_chunks(
                source_bytes,
                total_pages,
                deadline_monotonic=deadline_monotonic,
                require_full=require_full,
            )
            timed_out = timed_out or ocr_timed_out

    if not structural_chunks:
        if _is_hierarchical_chunking_mode(chunking_mode):
            structural_chunks = build_planning_hierarchical_chunks(cleaned, page_maps, chunking_mode)
            if not structural_chunks:
                structural_chunks = _fallback_structural_chunks(cleaned, page_maps)
                for item in structural_chunks:
                    item["chunkingFallback"] = True
        else:
            structural_chunks = _fallback_structural_chunks(cleaned, page_maps)

    used_hierarchical_chunks = (
        _is_hierarchical_chunking_mode(chunking_mode)
        and bool(structural_chunks)
        and not any(item.get("chunkingFallback") for item in structural_chunks)
    )
    if not used_hierarchical_chunks:
        structural_chunks = _merge_continuation_chunks(structural_chunks)
    structural_chunks = [
        item
        for item in structural_chunks
        if item.get("isParentChunk") or not _is_structurally_weak_chunk(item)
    ]

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

    canonical_district = canonicalize_planning_district(
        payload.district,
        title=payload.title,
        dossier_code=payload.dossier_code,
    )

    base_metadata: dict[str, Any] = {
        "documentScope": "planning",
        "planningDocumentId": payload.planning_document_id,
        "title": payload.title,
        "sourceUrl": payload.source_url,
        "format": payload.format,
        "documentType": payload.document_type,
        "dossierCode": payload.dossier_code,
        "city": payload.city,
        "district": canonical_district or payload.district,
        "districtCanonical": canonical_district,
        "districtRaw": payload.district,
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
        if page_number is None and isinstance(preferred_page, int) and not isinstance(preferred_page, bool):
            page_number = preferred_page

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
                "sectionHeading": item.get("sectionHeading"),
                "hierarchyPath": item.get("hierarchyPath"),
                "hierarchyLevel": item.get("hierarchyLevel"),
                "parentChunkId": item.get("parentChunkId"),
                "siblingIndex": item.get("siblingIndex"),
                "isParentChunk": bool(item.get("isParentChunk")),
                "chunkingMode": chunking_mode,
                "chunkingFallback": bool(item.get("chunkingFallback")),
                "chunkPreview": chunk[:400],
            }
        )
        docs.append(Document(page_content=_enrich_for_embedding(chunk, md), metadata=md))

    elapsed_ms = int((time.monotonic() - start_monotonic) * 1000)
    _logger.info(
        "Planning document extraction completed. planning_document_id=%s chunking_mode=%s format=%s pages=%s chunks=%s text_chunks=%s table_chunks=%s timed_out=%s elapsed_ms=%s",
        payload.planning_document_id,
        chunking_mode,
        fmt,
        total_pages,
        len(docs),
        text_count,
        table_count,
        timed_out,
        elapsed_ms,
    )

    return docs, {"textChunks": text_count, "tableChunks": table_count, "timedOut": 1 if timed_out else 0}
