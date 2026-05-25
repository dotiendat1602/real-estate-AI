from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import text

from ..db.pgvector import AsyncSessionLocal
from ..planning.ingestion import PlanningIngestPayload, build_planning_documents
from ..rag.resources import initialize_planning_vector_store, planning_collection_name

router = APIRouter()
_logger = logging.getLogger(__name__)

async def initialize_vector_store():
    return await initialize_planning_vector_store()


class PlanningIngestDocument(BaseModel):
    planningDocumentId: int
    title: str
    sourceUrl: HttpUrl
    format: Optional[str] = None
    documentType: Optional[str] = None
    dossierCode: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    planYear: Optional[int] = None
    propertyId: Optional[int] = None
    rawMeta: dict[str, Any] = Field(default_factory=dict)


class PlanningIngestRequest(BaseModel):
    replaceExisting: bool = True
    skipIfExists: bool = True
    documents: list[PlanningIngestDocument] = Field(default_factory=list)


def _planning_collection_name() -> str:
    return planning_collection_name()


@router.get("/planning/ingested-documents")
async def list_ingested_planning_documents(limit: int = 50, offset: int = 0):
    collection_name = _planning_collection_name()

    async with AsyncSessionLocal() as session:
        q = text(
            """
            SELECT
              (e.cmetadata->>'planningDocumentId')::bigint AS planning_document_id,
              max(e.cmetadata->>'title') AS title,
              max(e.cmetadata->>'sourceUrl') AS source_url,
              max(e.cmetadata->>'documentType') AS document_type,
              max(e.cmetadata->>'city') AS city,
              max(e.cmetadata->>'district') AS district,
              max(e.cmetadata->>'planYear') AS plan_year,
              COUNT(*) AS total_chunks,
              SUM(CASE WHEN e.cmetadata->>'chunkType' = 'table' THEN 1 ELSE 0 END) AS table_chunks,
              SUM(CASE WHEN e.cmetadata->>'chunkType' = 'text' THEN 1 ELSE 0 END) AS text_chunks,
              min((e.cmetadata->>'pageNumber')::int) AS min_page,
              max((e.cmetadata->>'pageNumber')::int) AS max_page
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :collection_name
              AND e.cmetadata->>'documentScope' = 'planning'
            GROUP BY (e.cmetadata->>'planningDocumentId')::bigint
            ORDER BY planning_document_id DESC
            LIMIT :limit OFFSET :offset
            """
        )
        res = await session.execute(q, {"collection_name": collection_name, "limit": limit, "offset": offset})
        rows = [dict(r._mapping) for r in res.fetchall()]

    return {"ok": True, "collection": collection_name, "items": rows, "limit": limit, "offset": offset}


@router.get("/planning/ingested-documents/{planning_document_id}/chunks")
async def get_ingested_planning_document_chunks(planning_document_id: int, limit: int = 100, offset: int = 0):
    collection_name = _planning_collection_name()

    async with AsyncSessionLocal() as session:
        q = text(
            """
            SELECT
              e.id,
              (e.cmetadata->>'planningDocumentId')::bigint AS planning_document_id,
              e.cmetadata->>'title' AS title,
              e.cmetadata->>'sourceUrl' AS source_url,
              e.cmetadata->>'chunkType' AS chunk_type,
              (e.cmetadata->>'chunkIndex')::int AS chunk_index,
              (e.cmetadata->>'globalChunkIndex')::int AS global_chunk_index,
              (e.cmetadata->>'pageNumber')::int AS page_number,
              (e.cmetadata->>'lineStart')::int AS line_start,
              (e.cmetadata->>'lineEnd')::int AS line_end,
              e.cmetadata->>'sourceLocator' AS source_locator,
              e.cmetadata->>'chunker' AS chunker,
              left(e.document, 1200) AS chunk_text,
              e.cmetadata
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :collection_name
              AND e.cmetadata->>'documentScope' = 'planning'
              AND e.cmetadata->>'planningDocumentId' = :planning_document_id
            ORDER BY global_chunk_index ASC, chunk_index ASC, id ASC
            LIMIT :limit OFFSET :offset
            """
        )
        res = await session.execute(
            q,
            {
                "collection_name": collection_name,
                "planning_document_id": str(planning_document_id),
                "limit": limit,
                "offset": offset,
            },
        )
        rows = [dict(r._mapping) for r in res.fetchall()]

        q_total = text(
            """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = :collection_name
              AND e.cmetadata->>'documentScope' = 'planning'
              AND e.cmetadata->>'planningDocumentId' = :planning_document_id
            """
        )
        total = (
            await session.execute(
                q_total,
                {
                    "collection_name": collection_name,
                    "planning_document_id": str(planning_document_id),
                },
            )
        ).scalar_one()

    return {
        "ok": True,
        "collection": collection_name,
        "planningDocumentId": planning_document_id,
        "totalChunks": total,
        "items": rows,
        "limit": limit,
        "offset": offset,
    }


@router.post("/planning/ingest-documents")
async def ingest_planning_documents(req: PlanningIngestRequest):
    if not req.documents:
        _logger.info("Planning ingest request completed with no documents.")
        return {"ok": True, "ingestedChunks": 0, "items": []}

    request_started = time.monotonic()
    _logger.info(
        "Planning ingest request started. documents=%s replace_existing=%s skip_if_exists=%s",
        len(req.documents),
        req.replaceExisting,
        req.skipIfExists,
    )

    vs = await initialize_planning_vector_store()
    result_items: list[dict[str, Any]] = []
    total_ingested = 0
    failed_documents = 0

    for item in req.documents:
        doc_started = time.monotonic()
        if req.replaceExisting:
            async with AsyncSessionLocal() as session:
                deleted = await session.execute(
                    text(
                        "DELETE FROM langchain_pg_embedding "
                        "WHERE cmetadata->>'documentScope' = 'planning' "
                        "AND cmetadata->>'planningDocumentId' = :doc_id"
                    ),
                    {"doc_id": str(item.planningDocumentId)},
                )
                await session.commit()
                deleted_count = deleted.rowcount or 0
        else:
            deleted_count = 0

        if not req.replaceExisting and req.skipIfExists:
            async with AsyncSessionLocal() as session:
                existed = await session.execute(
                    text(
                        "SELECT 1 FROM langchain_pg_embedding "
                        "WHERE cmetadata->>'documentScope' = 'planning' "
                        "AND cmetadata->>'planningDocumentId' = :doc_id "
                        "LIMIT 1"
                    ),
                    {"doc_id": str(item.planningDocumentId)},
                )
                exists_row = existed.first()

            if exists_row:
                elapsed_ms = int((time.monotonic() - doc_started) * 1000)
                _logger.info(
                    "Planning ingest document skipped. planning_document_id=%s reason=%s deleted_chunks=%s elapsed_ms=%s",
                    item.planningDocumentId,
                    "already_ingested",
                    deleted_count,
                    elapsed_ms,
                )
                result_items.append(
                    {
                        "planningDocumentId": item.planningDocumentId,
                        "title": item.title,
                        "deletedChunks": deleted_count,
                        "ingestedChunks": 0,
                        "textChunks": 0,
                        "tableChunks": 0,
                        "skipped": True,
                        "reason": "already_ingested",
                    }
                )
                continue

        payload = PlanningIngestPayload(
            planning_document_id=item.planningDocumentId,
            title=item.title,
            source_url=str(item.sourceUrl),
            format=item.format,
            document_type=item.documentType,
            dossier_code=item.dossierCode,
            city=item.city,
            district=item.district,
            plan_year=item.planYear,
            property_id=item.propertyId,
            raw_meta=item.rawMeta,
        )

        try:
            docs, counts = await build_planning_documents(payload)
        except RuntimeError as exc:
            failed_documents += 1
            _logger.exception(
                "Planning ingest runtime error. planning_document_id=%s source_url=%s",
                item.planningDocumentId,
                str(item.sourceUrl),
            )
            result_items.append(
                {
                    "planningDocumentId": item.planningDocumentId,
                    "title": item.title,
                    "deletedChunks": deleted_count,
                    "ingestedChunks": 0,
                    "textChunks": 0,
                    "tableChunks": 0,
                    "skipped": True,
                    "reason": "runtime_error",
                    "error": str(exc),
                }
            )
            continue
        except Exception as exc:
            failed_documents += 1
            _logger.exception(
                "Planning ingest unexpected error. planning_document_id=%s source_url=%s",
                item.planningDocumentId,
                str(item.sourceUrl),
            )
            result_items.append(
                {
                    "planningDocumentId": item.planningDocumentId,
                    "title": item.title,
                    "deletedChunks": deleted_count,
                    "ingestedChunks": 0,
                    "textChunks": 0,
                    "tableChunks": 0,
                    "skipped": True,
                    "reason": "unexpected_error",
                    "error": str(exc),
                }
            )
            continue

        if docs:
            try:
                ids = await vs.aadd_documents(docs)
                ingested_count = len(ids)
            except Exception as exc:
                failed_documents += 1
                _logger.exception(
                    "Planning ingest failed while storing embeddings. planning_document_id=%s source_url=%s",
                    item.planningDocumentId,
                    str(item.sourceUrl),
                )
                result_items.append(
                    {
                        "planningDocumentId": item.planningDocumentId,
                        "title": item.title,
                        "deletedChunks": deleted_count,
                        "ingestedChunks": 0,
                        "textChunks": counts["textChunks"],
                        "tableChunks": counts["tableChunks"],
                        "skipped": True,
                        "reason": "vector_store_error",
                        "error": str(exc),
                    }
                )
                continue
        else:
            ingested_count = 0

        total_ingested += ingested_count
        timed_out = bool(counts.get("timedOut"))
        reason = (
            "processing_timeout"
            if ingested_count == 0 and timed_out
            else ("no_extractable_content" if ingested_count == 0 else None)
        )

        elapsed_ms = int((time.monotonic() - doc_started) * 1000)
        _logger.info(
            "Planning ingest document completed. planning_document_id=%s deleted_chunks=%s ingested_chunks=%s text_chunks=%s table_chunks=%s skipped=%s reason=%s elapsed_ms=%s",
            item.planningDocumentId,
            deleted_count,
            ingested_count,
            counts["textChunks"],
            counts["tableChunks"],
            ingested_count == 0,
            reason,
            elapsed_ms,
        )

        result_items.append(
            {
                "planningDocumentId": item.planningDocumentId,
                "title": item.title,
                "deletedChunks": deleted_count,
                "ingestedChunks": ingested_count,
                "textChunks": counts["textChunks"],
                "tableChunks": counts["tableChunks"],
                "skipped": ingested_count == 0,
                "reason": reason,
            }
        )

    ok = failed_documents == 0
    elapsed_ms = int((time.monotonic() - request_started) * 1000)
    _logger.info(
        "Planning ingest request completed. ok=%s documents=%s failed_documents=%s ingested_chunks=%s elapsed_ms=%s",
        ok,
        len(req.documents),
        failed_documents,
        total_ingested,
        elapsed_ms,
    )

    return {
        "ok": failed_documents == 0,
        "ingestedChunks": total_ingested,
        "failedDocuments": failed_documents,
        "items": result_items,
    }


