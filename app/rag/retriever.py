from __future__ import annotations

import os
from langchain_postgres import PGVector
from langchain_core.vectorstores import VectorStoreRetriever

def build_pgvector_store(embeddings) -> PGVector:
    pgvector_url = os.getenv("PGVECTOR_URL", "")
    if not pgvector_url:
        raise RuntimeError("PGVECTOR_URL is missing")

    collection_name = os.getenv("PGVECTOR_COLLECTION", "post_embeddings")

    return PGVector(
        connection=pgvector_url,
        embeddings=embeddings,
        collection_name=collection_name,
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
    
    pgvector_filter = {}
    
    # Exact match filters - chỉ thêm nếu có trong filters
    exact_match_fields = ['city', 'district', 'postType', 'bedrooms']
    for field in exact_match_fields:
        if field in filters:
            pgvector_filter[field] = filters[field]
    
    # Range filters for price
    # Metadata trong DB có field 'price' (giá trị thực của post)
    # User query có thể có 'priceMin' hoặc 'priceMax'
    price_filter = {}
    if 'priceMin' in filters:
        price_filter['$gte'] = filters['priceMin']  # price >= priceMin
    if 'priceMax' in filters:
        price_filter['$lte'] = filters['priceMax']  # price <= priceMax
    
    if price_filter:
        pgvector_filter['price'] = price_filter
    
    # Range filters for area
    area_filter = {}
    if 'areaMin' in filters:
        area_filter['$gte'] = filters['areaMin']
    if 'areaMax' in filters:
        area_filter['$lte'] = filters['areaMax']
    
    if area_filter:
        pgvector_filter['area'] = area_filter
    
    return pgvector_filter

def build_retriever(vs: PGVector, k: int = 12, filters: dict | None = None) -> VectorStoreRetriever:
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
    
    if filters:
        pgvector_filter = build_metadata_filter(filters)
        
        if pgvector_filter:
            search_kwargs["filter"] = pgvector_filter
            print(f"[Retriever] Applying PGVector filter: {pgvector_filter}")
        else:
            print(f"[Retriever] No valid metadata filters to apply")
    
    return vs.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
