from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

PLANNING_CHUNKING_BASELINE_FIXED = "planning_baseline_fixed"
PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT = "planning_hierarchical_parent_context"
PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD = "planning_hierarchical_parent_child"

_PLANNING_HIERARCHICAL_MODES = {
    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT,
    PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD,
}


def _resolve_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_planning_chunking_mode() -> str:
    raw = (
        os.getenv("PLANNING_CHUNKING_MODE") or PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT
    ).strip().lower()
    aliases = {
        "baseline": PLANNING_CHUNKING_BASELINE_FIXED,
        "fixed": PLANNING_CHUNKING_BASELINE_FIXED,
        "hierarchical_parent_context": PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT,
        "hierarchical_parent_child": PLANNING_CHUNKING_HIERARCHICAL_PARENT_CHILD,
    }
    resolved = aliases.get(raw, raw)
    valid = {PLANNING_CHUNKING_BASELINE_FIXED, *_PLANNING_HIERARCHICAL_MODES}
    if resolved not in valid:
        _logger.warning(
            "Unknown PLANNING_CHUNKING_MODE=%s. Falling back to %s",
            raw,
            PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT,
        )
        return PLANNING_CHUNKING_HIERARCHICAL_PARENT_CONTEXT
    return resolved


def is_hierarchical_chunking_mode(mode: str) -> bool:
    return mode in _PLANNING_HIERARCHICAL_MODES


def resolve_http_verify() -> bool | str:
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


def resolve_ssl_allow_insecure_fallback() -> bool:
    return _resolve_bool_env("PLANNING_INGEST_SSL_ALLOW_INSECURE_FALLBACK", False)


def resolve_pdf_ocr_fallback_enabled() -> bool:
    return _resolve_bool_env("PLANNING_PDF_OCR_FALLBACK_ENABLED", True)


def resolve_pdf_ocr_max_pages() -> int:
    raw = os.getenv("PLANNING_PDF_OCR_MAX_PAGES", "0").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def resolve_pdf_ocr_render_scale() -> float:
    raw = os.getenv("PLANNING_PDF_OCR_RENDER_SCALE", "1.25").strip()
    try:
        value = float(raw)
        return max(0.5, min(value, 3.0))
    except Exception:
        return 1.25


def resolve_pdf_text_quality_min_score() -> float:
    raw = os.getenv("PLANNING_PDF_TEXT_QUALITY_MIN_SCORE", "0.42").strip()
    try:
        value = float(raw)
        return max(0.0, min(value, 1.0))
    except Exception:
        return 0.42


def resolve_pdf_force_ocr_on_low_quality() -> bool:
    return _resolve_bool_env("PLANNING_PDF_FORCE_OCR_ON_LOW_QUALITY", True)


def resolve_ingest_soft_timeout_seconds() -> int:
    raw = os.getenv("PLANNING_INGEST_SOFT_TIMEOUT_SECONDS", "0").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 0


def resolve_ingest_require_full() -> bool:
    return _resolve_bool_env("PLANNING_INGEST_REQUIRE_FULL", False)


def resolve_ocr_progress_every_pages() -> int:
    raw = os.getenv("PLANNING_OCR_PROGRESS_EVERY_PAGES", "5").strip()
    try:
        value = int(raw)
        return max(0, value)
    except Exception:
        return 5
