from __future__ import annotations

import os
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever

def build_pgvector_store(embeddings, collection_name: str | None = None) -> PGVector:
    pgvector_url = os.getenv("PGVECTOR_URL", "")
    if not pgvector_url:
        raise RuntimeError("PGVECTOR_URL is missing")

    resolved_collection = collection_name or os.getenv("PGVECTOR_COLLECTION", "post_embeddings")

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

def build_retriever(
    vs: PGVector,
    k: int = 12,
    filters: dict | None = None,
    base_filter: dict | None = None,
) -> VectorStoreRetriever:
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
