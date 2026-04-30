from __future__ import annotations

import argparse
import asyncio
import os
import hashlib
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from sqlalchemy import text


PLANNING_CHUNKING_MODES = (
    "planning_baseline_fixed",
    "planning_hierarchical_parent_context",
    "planning_hierarchical_parent_child",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-embed an existing planning collection into another collection without re-running OCR/chunking."
    )
    parser.add_argument("--source-collection", required=True)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clear-target", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing target rows and embed only source chunks not already present in the target collection.",
    )
    return parser.parse_args()


def _collection_chunking_mode(collection_name: str) -> str | None:
    for mode in PLANNING_CHUNKING_MODES:
        if collection_name.endswith(f"__{mode}"):
            return mode
    return None


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    stable_parts = [
        str(metadata.get("planningDocumentId") or ""),
        str(metadata.get("chunkingMode") or ""),
        str(metadata.get("globalChunkIndex") or ""),
        str(metadata.get("chunkType") or ""),
        str(metadata.get("isParentChunk") or ""),
        str(metadata.get("parentChunkId") or ""),
        str(metadata.get("siblingIndex") or ""),
    ]
    if stable_parts[0] and stable_parts[2]:
        return "|".join(stable_parts)
    digest = hashlib.sha1(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return "|".join([*stable_parts, digest])


async def _load_documents(source_collection: str) -> list[Document]:
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :collection_name
                ORDER BY e.cmetadata->>'planningDocumentId',
                         (e.cmetadata->>'globalChunkIndex')::int NULLS LAST,
                         e.id
                """
            ),
            {"collection_name": source_collection},
        )
        docs: list[Document] = []
        for document, metadata in result.all():
            md: dict[str, Any]
            if isinstance(metadata, dict):
                md = dict(metadata)
            else:
                md = {}
            docs.append(Document(page_content=str(document or ""), metadata=md))
        return docs


async def _load_document_keys(collection_name: str) -> set[str]:
    docs = await _load_documents(collection_name)
    return {_doc_key(doc) for doc in docs}


async def _clear_collection(collection_name: str) -> None:
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(
            text(
                """
                DELETE FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name
                )
                """
            ),
            {"collection_name": collection_name},
        )
        await session.commit()


async def _collection_count(collection_name: str) -> int:
    from app.db.pgvector import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                WHERE c.name = :collection_name
                """
            ),
            {"collection_name": collection_name},
        )
        return int(result.scalar_one() or 0)


async def main() -> None:
    args = parse_args()
    service_root = Path(__file__).resolve().parents[2]
    load_dotenv(service_root / ".env")
    os.chdir(service_root)

    source_mode = _collection_chunking_mode(args.source_collection)
    target_mode = _collection_chunking_mode(args.target_collection)
    if source_mode != target_mode:
        raise RuntimeError(
            "Refusing to re-embed across different planning chunking modes: "
            f"source={source_mode} target={target_mode}"
        )
    if args.clear_target and args.resume:
        raise RuntimeError("--clear-target and --resume are mutually exclusive.")

    os.environ["EMBED_MODEL"] = args.embedding_model
    if args.embedding_model.startswith("intfloat/"):
        os.environ.setdefault("EMBED_QUERY_PREFIX", "query: ")
        os.environ.setdefault("EMBED_DOCUMENT_PREFIX", "passage: ")
    else:
        os.environ.pop("EMBED_QUERY_PREFIX", None)
        os.environ.pop("EMBED_DOCUMENT_PREFIX", None)

    from app.rag.embedder import build_embeddings
    from app.rag.retriever import build_pgvector_store

    docs = await _load_documents(args.source_collection)
    if not docs:
        raise RuntimeError(f"Source collection is empty: {args.source_collection}")

    if args.clear_target:
        await _clear_collection(args.target_collection)

    existing = await _collection_count(args.target_collection)
    if existing:
        if args.resume:
            existing_keys = await _load_document_keys(args.target_collection)
            before = len(docs)
            docs = [doc for doc in docs if _doc_key(doc) not in existing_keys]
            print(
                f"[Reembed] resume target_existing={existing} source_docs={before} remaining={len(docs)}",
                flush=True,
            )
            if not docs:
                print(
                    f"[Reembed] done target_count={await _collection_count(args.target_collection)}",
                    flush=True,
                )
                return
        else:
            print(
                f"[Reembed] target already has {existing} rows. "
                "Use --clear-target to rebuild it or --resume to continue it.",
                flush=True,
            )
            return

    print(
        f"[Reembed] source={args.source_collection} target={args.target_collection} "
        f"model={args.embedding_model} docs={len(docs)} batch_size={args.batch_size}",
        flush=True,
    )
    vs = build_pgvector_store(build_embeddings(), collection_name=args.target_collection)
    await vs.__apost_init__()
    total = 0
    for start in range(0, len(docs), args.batch_size):
        batch = docs[start : start + args.batch_size]
        ids = await vs.aadd_documents(batch)
        total += len(ids)
        print(f"[Reembed] {total}/{len(docs)}", flush=True)

    print(
        f"[Reembed] done target_count={await _collection_count(args.target_collection)}",
        flush=True,
    )


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
