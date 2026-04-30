from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from typing import Any

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever
from sqlalchemy import text

from ..db.pgvector import AsyncSessionLocal


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def _doc_identity(doc: Document) -> str:
    md = doc.metadata or {}
    chunk_index = md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex")
    return "|".join(
        [
            str(md.get("postId") or ""),
            str(md.get("propertyId") or ""),
            str(md.get("planningDocumentId") or ""),
            str(md.get("chunkType") or ""),
            str(chunk_index or ""),
        ]
    )

def build_pgvector_store(embeddings, collection_name: str | None = None) -> PGVector:
    pgvector_url = os.getenv("PGVECTOR_URL", "")
    if not pgvector_url:
        raise RuntimeError("PGVECTOR_URL is missing")

    resolved_collection = collection_name or os.getenv("PGVECTOR_COLLECTION", "post_embeddings__multilingual_e5_base_ghr1")

    return PGVector(
        connection=pgvector_url,
        embeddings=embeddings,
        collection_name=resolved_collection,
        use_jsonb=True,
        async_mode=True,
        create_extension=False,
    )

def build_metadata_filter(filters: dict) -> dict:
    """
    Convert extracted filters to PGVector metadata filter format.
    
    Chỉ filter theo những field có trong filters (được extract ra từ user query).
    
    Handles:
    - Exact match: city, district, postType, bedrooms
    - Range filters: priceMin/Max, areaMin/Max → convert sang $gte/$lte operators
    
    Args:
        filters: Extracted filters from user query
        
    Returns:
        PGVector-compatible filter dict
        
    Example:
        Input: {'city': 'Hà Nội', 'priceMax': 5000000000, 'bedrooms': 2}
        Output: {
            'city': 'Hà Nội',
            'bedrooms': 2,
            'price': {'$lte': 5000000000}
        }
    """
    if not filters:
        return {}
    
    clauses: list[dict] = []
    
    # Exact match filters - chi them neu co trong filters
    exact_match_fields = [
        'city',
        'district',
        'postType',
        'bedrooms',
        'postId',
        'planningDocumentId',
        'documentScope',
        'documentType',
        'dossierCode',
        'chunkType',
        'propertyId',
    ]
    for field in exact_match_fields:
        if field in filters:
            clauses.append({field: filters[field]})
    
    # Range filters for price
    # Metadata trong DB có field 'price' (giá trị thực của post)
    # User query có thể có 'priceMin' hoặc 'priceMax'
    if 'priceMin' in filters:
        clauses.append({'price': {'$gte': filters['priceMin']}})
    if 'priceMax' in filters:
        clauses.append({'price': {'$lte': filters['priceMax']}})
    
    # Range filters for area
    if 'areaMin' in filters:
        clauses.append({'area': {'$gte': filters['areaMin']}})
    if 'areaMax' in filters:
        clauses.append({'area': {'$lte': filters['areaMax']}})

    # Range/equality filter for plan year.
    if 'planYear' in filters:
        clauses.append({'planYear': filters['planYear']})

    # Optional IN filter for chunk types.
    if isinstance(filters.get('chunkTypes'), list) and filters['chunkTypes']:
        clauses.append({'chunkType': {'$in': filters['chunkTypes']}})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {'$and': clauses}


def _merge_pgvector_filters(*filters: dict | None) -> dict:
    clauses: list[dict] = []
    for filter_item in filters:
        if not filter_item:
            continue
        if len(filter_item) == 1 and '$and' in filter_item and isinstance(filter_item['$and'], list):
            clauses.extend(filter_item['$and'])
            continue
        clauses.append(filter_item)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {'$and': clauses}


class HybridRetriever:
    """Hybrid dense + lexical retriever with RRF fusion."""

    def __init__(
        self,
        vs: PGVector,
        k: int = 12,
        filters: dict | None = None,
        base_filter: dict | None = None,
    ):
        self.vs = vs
        self.k = max(1, int(k))
        self.filters = dict(filters or {})
        self.base_filter = dict(base_filter or {})

        # Environment-tunable knobs to avoid hardcoded per-case tuning.
        self.semantic_multiplier = max(1, int(os.getenv("RAG_HYBRID_SEM_MULTIPLIER", "3")))
        self.lexical_multiplier = max(1, int(os.getenv("RAG_HYBRID_LEX_MULTIPLIER", "3")))
        self.rrf_k = max(1, int(os.getenv("RAG_HYBRID_RRF_K", "60")))
        self.semantic_weight = float(os.getenv("RAG_HYBRID_SEM_WEIGHT", "1.0"))
        self.lexical_weight = float(os.getenv("RAG_HYBRID_LEX_WEIGHT", "0.9"))

        base_pg_filter = build_metadata_filter(self.base_filter) if self.base_filter else {}
        query_pg_filter = build_metadata_filter(self.filters) if self.filters else {}
        self.merged_filter = _merge_pgvector_filters(base_pg_filter, query_pg_filter)

    async def _semantic_search(self, query: str, k: int) -> list[tuple[Document, float | None]]:
        if not query:
            return []

        kwargs: dict[str, Any] = {"k": max(1, int(k))}
        if self.merged_filter:
            kwargs["filter"] = self.merged_filter

        # Prefer explicit relevance scores when backend supports it.
        if hasattr(self.vs, "asimilarity_search_with_relevance_scores"):
            try:
                scored_docs = await self.vs.asimilarity_search_with_relevance_scores(query, **kwargs)
                return [(doc, float(score)) for doc, score in scored_docs]
            except Exception:
                pass

        try:
            docs = await self.vs.asimilarity_search(query, **kwargs)
            return [(doc, None) for doc in docs]
        except Exception:
            pass

        # Final fallback for compatibility with vectorstore implementations.
        search_kwargs = {"k": kwargs["k"]}
        if self.merged_filter:
            search_kwargs["filter"] = self.merged_filter
        retriever = self.vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        docs = await retriever.ainvoke(query)
        return [(doc, None) for doc in docs]

    async def ainvoke_with_scores(self, query: str) -> tuple[list[Document], list[float | None]]:
        semantic_k = max(self.k, min(96, self.k * self.semantic_multiplier))
        lexical_k = max(self.k, min(96, self.k * self.lexical_multiplier))

        semantic_task = asyncio.create_task(self._semantic_search(query, semantic_k))
        lexical_task = asyncio.create_task(
            lexical_search_documents(
                self.vs,
                query=query,
                k=lexical_k,
                filters=self.filters,
                base_filter=self.base_filter,
            )
        )

        semantic_res, lexical_docs = await asyncio.gather(semantic_task, lexical_task)

        docs_by_identity: dict[str, Document] = {}
        fused_scores: dict[str, float] = {}

        for rank, (doc, semantic_score) in enumerate(semantic_res):
            identity = _doc_identity(doc)
            if not identity:
                continue
            if identity not in docs_by_identity:
                docs_by_identity[identity] = doc
            fused = self.semantic_weight * _rrf_score(rank, self.rrf_k)
            if semantic_score is not None:
                # Keep score contribution small to preserve rank robustness across backends.
                fused += max(-1.0, min(1.0, float(semantic_score))) * 0.025
            fused_scores[identity] = fused_scores.get(identity, 0.0) + fused

        for rank, doc in enumerate(lexical_docs):
            identity = _doc_identity(doc)
            if not identity:
                continue
            if identity not in docs_by_identity:
                docs_by_identity[identity] = doc
            fused_scores[identity] = fused_scores.get(identity, 0.0) + self.lexical_weight * _rrf_score(rank, self.rrf_k)

        ranked = sorted(
            docs_by_identity.items(),
            key=lambda item: fused_scores.get(item[0], 0.0),
            reverse=True,
        )

        top_items = ranked[: self.k]
        docs = [doc for _, doc in top_items]
        scores: list[float | None] = [fused_scores.get(identity) for identity, _ in top_items]
        return docs, scores

    async def ainvoke(self, query: str) -> list[Document]:
        docs, _ = await self.ainvoke_with_scores(query)
        return docs

def build_retriever(
    vs: PGVector,
    k: int = 12,
    filters: dict | None = None,
    base_filter: dict | None = None,
) -> Any:
    """
    Build retriever with dynamic metadata filtering.
    
    Chỉ apply filters cho những field được extract từ user query.
    
    Args:
        vs: PGVector store
        k: Number of results to retrieve
        filters: Extracted filters from user query (city, district, price, bedrooms, etc.)
        
    Returns:
        VectorStoreRetriever with appropriate filters applied
    """
    mode = os.getenv("RAG_RETRIEVER_MODE", "hybrid").strip().lower()
    if mode == "hybrid":
        return HybridRetriever(vs=vs, k=k, filters=filters, base_filter=base_filter)

    search_kwargs = {"k": k}
    
    base_pg_filter = build_metadata_filter(base_filter or {}) if base_filter else {}
    query_pg_filter = build_metadata_filter(filters or {}) if filters else {}
    merged_filter: dict = _merge_pgvector_filters(base_pg_filter, query_pg_filter)

    if merged_filter:
        search_kwargs["filter"] = merged_filter
        print(f"[Retriever] Applying PGVector filter: {merged_filter}")
    elif filters:
        print("[Retriever] No valid metadata filters to apply")
    
    return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


_LEXICAL_NUMERIC_FIELDS = {
    "postId",
    "propertyId",
    "planningDocumentId",
    "planYear",
    "chunkIndex",
    "globalChunkIndex",
    "pageNumber",
    "price",
    "area",
    "bedrooms",
}

_LEXICAL_STOPWORDS = {
    "va",
    "la",
    "cua",
    "cho",
    "trong",
    "theo",
    "ve",
    "tu",
    "tai",
    "voi",
    "duoc",
    "nam",
    "bao",
    "nhieu",
    "nao",
    "nhu",
    "the",
    "ra",
    "sao",
}


_VIETNAMESE_ASCII_FOLDS: dict[str, str] = {
    "a": "áàảãạăắằẳẵặâấầẩẫậ",
    "e": "éèẻẽẹêếềểễệ",
    "i": "íìỉĩị",
    "o": "óòỏõọôốồổỗộơớờởỡợ",
    "u": "úùủũụưứừửữự",
    "y": "ýỳỷỹỵ",
    "d": "đ",
}

_LEXICAL_SQL_TRANSLATE_FROM = "".join(chars for chars in _VIETNAMESE_ASCII_FOLDS.values())
_LEXICAL_SQL_TRANSLATE_TO = "".join(base * len(chars) for base, chars in _VIETNAMESE_ASCII_FOLDS.items())


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = lowered.replace("đ", "d")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _extract_lexical_terms(query: str, max_terms: int = 12) -> list[str]:
    if not query:
        return []

    raw_tokens = re.findall(r"[\w/\-\.]+", query.lower(), flags=re.UNICODE)
    norm_tokens = re.findall(r"[\w/\-\.]+", _normalize_text(query), flags=re.UNICODE)

    normalized_query = _normalize_text(query)
    phrase_tokens = [
        token
        for token in re.findall(r"[a-z0-9/.\-]+", normalized_query, flags=re.UNICODE)
        if token
        and token not in _LEXICAL_STOPWORDS
    ]

    phrase_candidates: list[str] = []
    for window_size in range(5, 1, -1):
        if len(phrase_tokens) < window_size:
            continue
        for start in range(0, len(phrase_tokens) - window_size + 1):
            phrase = " ".join(phrase_tokens[start : start + window_size]).strip()
            compact_phrase = re.sub(r"[^a-z0-9]", "", phrase)
            if len(compact_phrase) < 10:
                continue
            phrase_candidates.append(phrase)

    terms: list[str] = []
    seen: set[str] = set()
    prioritized_tokens = [*phrase_candidates[:6], *raw_tokens, *norm_tokens]
    for token in prioritized_tokens:
        clean = token.strip("._-")
        if not clean:
            continue
        if clean in _LEXICAL_STOPWORDS:
            continue
        if len(clean) < 3 and "/" not in clean and not clean.isdigit():
            continue
        if clean in seen:
            continue
        seen.add(clean)
        terms.append(clean)

        compact = re.sub(r"[^a-z0-9]", "", _normalize_text(clean))
        if len(compact) >= 10 and compact not in seen:
            seen.add(compact)
            terms.append(compact)

        # Include a plain numeric fragment for decimal values (e.g. 8,6915 -> 86915)
        # so lexical LIKE can still hit OCR/format variants.
        compact_numeric = re.sub(r"[^0-9]", "", clean)
        if len(compact_numeric) >= 4 and compact_numeric not in seen:
            seen.add(compact_numeric)
            terms.append(compact_numeric)

        if len(terms) >= max_terms:
            break

    return terms


def _sql_normalized_text_expr(expr: str, *, compact: bool = False) -> str:
    normalized = (
        "translate("
        f"lower(coalesce({expr}, '')),"
        f" '{_LEXICAL_SQL_TRANSLATE_FROM}',"
        f" '{_LEXICAL_SQL_TRANSLATE_TO}'"
        ")"
    )
    if compact:
        return f"regexp_replace({normalized}, '[^a-z0-9]+', '', 'g')"
    return f"regexp_replace({normalized}, '[^a-z0-9]+', ' ', 'g')"


def _coerce_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_raw_filters(base_filter: dict | None, filters: dict | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (base_filter or {}, filters or {}):
        for key, value in source.items():
            if key == "chunkTypes" and isinstance(value, list):
                merged["chunkType"] = {"$in": value}
                continue

            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
                continue

            merged[key] = value
    return merged


def _append_in_clause(
    clauses: list[str],
    params: dict[str, Any],
    key: str,
    values: list[Any],
    index_seed: int,
) -> int:
    if not values:
        return index_seed

    placeholders: list[str] = []
    is_numeric = key in _LEXICAL_NUMERIC_FIELDS

    for offset, item in enumerate(values):
        param_name = f"flt_{key}_{index_seed + offset}"
        if is_numeric:
            number = _coerce_number(item)
            if number is None:
                continue
            params[param_name] = number
        else:
            params[param_name] = str(item).lower().strip()
        placeholders.append(f":{param_name}")

    if not placeholders:
        return index_seed

    if is_numeric:
        clauses.append(f"(NULLIF(e.cmetadata->>'{key}', '')::double precision IN ({', '.join(placeholders)}))")
    else:
        clauses.append(f"(LOWER(COALESCE(e.cmetadata->>'{key}', '')) IN ({', '.join(placeholders)}))")

    return index_seed + len(placeholders)


def _build_metadata_where_clauses(filters: dict[str, Any], params: dict[str, Any]) -> list[str]:
    clauses: list[str] = []
    index_seed = 0

    for key, value in (filters or {}).items():
        if value is None:
            continue

        is_numeric = key in _LEXICAL_NUMERIC_FIELDS

        if isinstance(value, dict):
            if "$in" in value and isinstance(value["$in"], list):
                index_seed = _append_in_clause(clauses, params, key, value["$in"], index_seed)
                continue

            if "$gte" in value:
                number = _coerce_number(value.get("$gte"))
                if number is not None:
                    param_name = f"flt_{key}_gte_{index_seed}"
                    params[param_name] = number
                    clauses.append(f"(NULLIF(e.cmetadata->>'{key}', '')::double precision >= :{param_name})")
                    index_seed += 1

            if "$lte" in value:
                number = _coerce_number(value.get("$lte"))
                if number is not None:
                    param_name = f"flt_{key}_lte_{index_seed}"
                    params[param_name] = number
                    clauses.append(f"(NULLIF(e.cmetadata->>'{key}', '')::double precision <= :{param_name})")
                    index_seed += 1

            continue

        param_name = f"flt_{key}_{index_seed}"
        if is_numeric:
            number = _coerce_number(value)
            if number is None:
                continue
            params[param_name] = number
            clauses.append(f"(NULLIF(e.cmetadata->>'{key}', '')::double precision = :{param_name})")
        else:
            params[param_name] = str(value).lower().strip()
            clauses.append(f"(LOWER(COALESCE(e.cmetadata->>'{key}', '')) = :{param_name})")
        index_seed += 1

    return clauses


async def lexical_search_documents(
    vs: PGVector,
    query: str,
    k: int = 12,
    filters: dict | None = None,
    base_filter: dict | None = None,
    allow_empty_terms: bool = False,
) -> list[Document]:
    """Metadata-aware lexical retrieval using SQL LIKE scoring on document + title fields."""
    if _env_bool("RAG_LEXICAL_DISABLE"):
        return []

    terms = _extract_lexical_terms(query)
    if not terms and not allow_empty_terms:
        return []

    merged_filters = _merge_raw_filters(base_filter, filters)
    collection_name = getattr(vs, "collection_name", None) or os.getenv(
        "PGVECTOR_COLLECTION",
        "post_embeddings__multilingual_e5_base_ghr1",
    )

    params: dict[str, Any] = {
        "collection_name": collection_name,
        "limit": max(1, int(k)),
    }

    term_predicates: list[str] = []
    term_score_parts: list[str] = []
    term_count_expr = "0"
    for idx, term in enumerate(terms):
        term_lower = str(term or "").lower().strip()
        term_norm = _normalize_text(term_lower)
        term_compact = re.sub(r"[^a-z0-9]", "", term_norm)
        if not term_lower and not term_norm and not term_compact:
            continue

        raw_param_name = f"term_raw_{idx}"
        norm_param_name = f"term_norm_{idx}"
        compact_param_name = f"term_compact_{idx}"
        params[raw_param_name] = f"%{term_lower}%"
        params[norm_param_name] = f"%{term_norm}%"
        params[compact_param_name] = f"%{term_compact}%"
        term_weight = 1.0
        if " " in term_norm and len(term_compact) >= 10:
            term_weight = 2.4
        elif re.fullmatch(r"\d{4,}", term_norm):
            term_weight = 2.0
        elif "/" in term_norm or len(term_compact) >= 12:
            term_weight = 1.8
        elif len(term_norm) >= 9:
            term_weight = 1.4

        title_norm_expr = _sql_normalized_text_expr("e.cmetadata->>'title'")
        dossier_norm_expr = _sql_normalized_text_expr("e.cmetadata->>'dossierCode'")
        locator_norm_expr = _sql_normalized_text_expr("e.cmetadata->>'sourceLocator'")
        title_compact_expr = _sql_normalized_text_expr("e.cmetadata->>'title'", compact=True)
        dossier_compact_expr = _sql_normalized_text_expr("e.cmetadata->>'dossierCode'", compact=True)
        locator_compact_expr = _sql_normalized_text_expr("e.cmetadata->>'sourceLocator'", compact=True)

        match_expr = (
            f"(LOWER(COALESCE(e.document, '')) LIKE :{raw_param_name} "
            f"OR LOWER(COALESCE(e.cmetadata->>'title', '')) LIKE :{raw_param_name} "
            f"OR LOWER(COALESCE(e.cmetadata->>'dossierCode', '')) LIKE :{raw_param_name} "
            f"OR LOWER(COALESCE(e.cmetadata->>'sourceLocator', '')) LIKE :{raw_param_name} "
            f"OR {_sql_normalized_text_expr('e.document')} LIKE :{norm_param_name} "
            f"OR {title_norm_expr} LIKE :{norm_param_name} "
            f"OR {dossier_norm_expr} LIKE :{norm_param_name} "
            f"OR {locator_norm_expr} LIKE :{norm_param_name} "
            f"OR {_sql_normalized_text_expr('e.document', compact=True)} LIKE :{compact_param_name} "
            f"OR {title_compact_expr} LIKE :{compact_param_name} "
            f"OR {dossier_compact_expr} LIKE :{compact_param_name} "
            f"OR {locator_compact_expr} LIKE :{compact_param_name})"
        )
        term_predicates.append(match_expr)
        term_score_parts.append(f"CASE WHEN {match_expr} THEN {term_weight} ELSE 0 END")

    if term_predicates:
        term_count_expr = " + ".join(f"CASE WHEN {expr} THEN 1 ELSE 0 END" for expr in term_predicates)

    min_term_hits = 1
    raw_min_hits = (os.getenv("RAG_LEXICAL_MIN_TERM_HITS", "") or "").strip()
    if raw_min_hits:
        try:
            min_term_hits = max(1, int(raw_min_hits))
        except ValueError:
            min_term_hits = 1

    where_clauses: list[str] = ["c.name = :collection_name"]
    if term_predicates:
        where_clauses.append("(" + " OR ".join(term_predicates) + ")")
    where_clauses.extend(_build_metadata_where_clauses(merged_filters, params))

    score_expr = " + ".join(term_score_parts) if term_score_parts else "0"
    having_clause = ""
    if term_predicates and not allow_empty_terms:
        having_clause = "\n        AND lexical_term_hits >= :min_term_hits"

    params["min_term_hits"] = max(1, min_term_hits)
    stmt = text(
        f"""
        SELECT *
        FROM (
            SELECT
                e.document AS document,
                e.cmetadata AS cmetadata,
                ({score_expr}) AS lexical_score,
                ({term_count_expr}) AS lexical_term_hits,
                e.id AS embedding_id
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE {' AND '.join(where_clauses)}
        ) ranked
        WHERE 1=1{having_clause}
        ORDER BY lexical_score DESC, lexical_term_hits DESC, embedding_id ASC
        LIMIT :limit
        """
    )

    lexical_timeout_seconds = _env_float("RAG_LEXICAL_TIMEOUT_SECONDS", 0.0)

    try:
        async with AsyncSessionLocal() as session:
            if lexical_timeout_seconds > 0:
                # Keep timeout server-side too so cancelled lexical probes do not
                # continue consuming the DB while semantic results are ready.
                timeout_ms = max(1, int(lexical_timeout_seconds * 1000))
                await session.execute(
                    text("SELECT set_config('statement_timeout', :timeout_ms, true)"),
                    {"timeout_ms": f"{timeout_ms}ms"},
                )
                result = await asyncio.wait_for(
                    session.execute(stmt, params),
                    timeout=lexical_timeout_seconds + 1.0,
                )
            else:
                result = await session.execute(stmt, params)
            rows = result.fetchall()
    except (asyncio.TimeoutError, TimeoutError):
        print(
            f"[Retriever] lexical search timed out after {lexical_timeout_seconds}s; "
            "using semantic results only",
            flush=True,
        )
        return []
    except Exception as exc:
        message = str(exc).lower()
        if "statement timeout" in message or "querycanceled" in message or "query canceled" in message:
            print(
                f"[Retriever] lexical search hit statement_timeout after {lexical_timeout_seconds}s; "
                "using semantic results only",
                flush=True,
            )
            return []
        raise

    docs: list[Document] = []
    for row in rows:
        mapping = row._mapping if hasattr(row, "_mapping") else row
        content = str(mapping.get("document") or "").strip()
        if not content:
            continue
        metadata = mapping.get("cmetadata")
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append(Document(page_content=content, metadata=metadata))

    return docs
