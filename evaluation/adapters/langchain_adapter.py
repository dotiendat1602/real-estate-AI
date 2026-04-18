from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.api.chat import _has_planning_intent, _retrieve_planning_docs_for_nl_query
from app.api.chat import build_llm
from app.rag.chain import (
    build_retrieval_query,
    detect_lang,
    postprocess_answer,
    prepare_docs_for_context,
    sanitize_llm_text,
)
from app.rag.embedder import build_embeddings
from app.rag.filter_extractor import extract_filters_from_query
from app.rag.prompt import prompt
from app.rag.retriever import build_metadata_filter, build_pgvector_store, build_retriever


@dataclass
class TurnTrace:
    input: str
    conversation_history_used: list[dict[str, str]]
    rewritten_query: str | None
    retrieval_context: list[str]
    retrieved_doc_ids: list[str]
    retrieval_scores: list[float | None]
    actual_output: str
    metadata: dict[str, Any]


@dataclass
class ConversationTrace:
    conversation_id: str
    turns: list[TurnTrace]


class LangChainEvalAdapter:
    """Adapter that runs the existing RAG pipeline and emits eval-friendly traces."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = settings or {}
        self.top_k = int(self.settings.get("runtime", {}).get("top_k", 12))
        self.max_history_turns = int(self.settings.get("runtime", {}).get("max_history_turns", 6))
        self.eval_context_max_docs = int(self.settings.get("runtime", {}).get("eval_context_max_docs", 4))
        self.eval_context_max_chars_per_doc = int(
            self.settings.get("runtime", {}).get("eval_context_max_chars_per_doc", 1400)
        )
        self.planning_eval_context_max_docs = int(
            self.settings.get("runtime", {}).get("planning_eval_context_max_docs", 8)
        )
        self.eval_explanatory_max_docs = int(
            self.settings.get("runtime", {}).get("eval_explanatory_max_docs", 6)
        )
        self.eval_explanatory_max_chars_per_doc = int(
            self.settings.get("runtime", {}).get("eval_explanatory_max_chars_per_doc", 2200)
        )

        self._embeddings = None
        self._vs = None
        self._planning_vs = None
        self._llm = None

    async def initialize(self) -> None:
        if self._embeddings is None:
            self._embeddings = build_embeddings()
        if self._vs is None:
            self._vs = build_pgvector_store(self._embeddings)
            if hasattr(self._vs, "__apost_init__"):
                await self._vs.__apost_init__()
        if self._planning_vs is None:
            planning_collection = os.getenv("PGVECTOR_COLLECTION_PLANNING", "planning_documents")
            self._planning_vs = build_pgvector_store(self._embeddings, collection_name=planning_collection)
            if hasattr(self._planning_vs, "__apost_init__"):
                await self._planning_vs.__apost_init__()
        if self._llm is None:
            self._llm = build_llm()

    async def run_single_turn(
        self,
        input_text: str,
        conversation_history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        target_metadata: dict[str, Any] | None = None,
    ) -> TurnTrace:
        await self.initialize()

        history_payload = (conversation_history or [])[-self.max_history_turns :]
        history_messages = self._to_langchain_messages(history_payload)
        target_metadata = target_metadata or {}
        force_planning_document_id: int | None = None
        raw_planning_doc_id = target_metadata.get("planningDocumentId")
        if raw_planning_doc_id is not None:
            try:
                force_planning_document_id = int(raw_planning_doc_id)
            except (TypeError, ValueError):
                force_planning_document_id = None

        k = top_k or self.top_k
        retrieval_query = build_retrieval_query(input_text, history_payload)
        rewritten_query = (
            retrieval_query
            if self._normalize_text(retrieval_query) != self._normalize_text(input_text)
            else None
        )
        filter_query = retrieval_query if rewritten_query else input_text
        extracted_filters = await extract_filters_from_query(filter_query, self._llm)
        filters = self._merge_target_metadata_filters(extracted_filters, target_metadata)

        use_planning_mode = _has_planning_intent(input_text) or self._target_metadata_suggests_planning(target_metadata)
        is_explanatory_query = self._is_explanatory_question(input_text)
        retrieval_strategy = "planning_nl" if use_planning_mode else "exact_filters"
        retrieval_started = time.perf_counter()
        retrieval_augmentation_docs_added = 0

        if use_planning_mode:
            docs = await _retrieve_planning_docs_for_nl_query(
                self._planning_vs,
                retrieval_query,
                k,
                history_messages=history_payload,
                force_planning_document_id=force_planning_document_id,
            )
            scores = [None] * len(docs)
        else:
            docs, scores, retrieval_strategy = await self._retrieve_with_fallbacks(
                retrieval_query,
                filters=filters,
                base_filters=extracted_filters,
                top_k=k,
                prefer_broader_context=is_explanatory_query,
            )
            if is_explanatory_query:
                original_doc_count = len(docs)
                docs, scores = await self._augment_explanatory_target_docs(
                    question=retrieval_query,
                    raw_input=input_text,
                    filters=filters,
                    docs=docs,
                    scores=scores,
                    top_k=k,
                    retrieval_strategy=retrieval_strategy,
                )
                retrieval_augmentation_docs_added = max(0, len(docs) - original_doc_count)
        retrieval_seconds = round(time.perf_counter() - retrieval_started, 3)

        context_selection_query = retrieval_query or input_text
        planning_max_docs = (
            max(self.eval_context_max_docs, self.planning_eval_context_max_docs)
            if use_planning_mode or _has_planning_intent(context_selection_query)
            else self.eval_context_max_docs
        )
        if is_explanatory_query and not use_planning_mode:
            planning_max_docs = max(planning_max_docs, self.eval_explanatory_max_docs)

        max_chars_per_doc = self.eval_context_max_chars_per_doc
        if is_explanatory_query and not use_planning_mode:
            max_chars_per_doc = max(max_chars_per_doc, self.eval_explanatory_max_chars_per_doc)

        context_docs = prepare_docs_for_context(
            context_selection_query,
            docs,
            max_docs=planning_max_docs,
            max_chars_per_doc=max_chars_per_doc,
        )
        eval_context_docs = self._prepare_eval_context_docs(
            context_selection_query,
            context_docs,
            use_planning_mode=use_planning_mode,
            is_explanatory_query=is_explanatory_query,
        )

        context_text = self._format_docs(context_docs)

        answer_started = time.perf_counter()
        answer = await self._generate_answer(
            question=input_text,
            context=context_text,
            history=history_messages,
            context_docs=context_docs,
        )
        answer_generation_seconds = round(time.perf_counter() - answer_started, 3)

        score_map = {
            self._doc_metadata_identity(doc): score
            for doc, score in zip(docs, scores)
        }
        context_scores = [score_map.get(self._doc_metadata_identity(doc)) for doc in context_docs]
        eval_context_scores = [score_map.get(self._doc_metadata_identity(doc)) for doc in eval_context_docs]
        context_chars = sum(len(d.page_content or "") for d in context_docs)
        eval_context_chars = sum(len(d.page_content or "") for d in eval_context_docs)

        metadata = {
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "judge_phase": self.settings.get("runtime", {}).get("judge_phase", "offline_eval"),
            "prompt_version": self.settings.get("versions", {}).get("prompt_version", "rag_prompt_v1"),
            "retriever_version": self.settings.get("versions", {}).get("retriever_version", "pgvector_similarity_v1"),
            "chunking_version": self.settings.get("versions", {}).get("chunking_version", "chunking_v1"),
            "top_k": k,
            "filters": filters,
            "extracted_filters": extracted_filters,
            "retrieval_query": retrieval_query,
            "context_selection_query": context_selection_query,
            "retrieval_strategy": retrieval_strategy,
            "raw_retrieved_docs_count": len(docs),
            "context_docs_count": len(context_docs),
            "context_chars": context_chars,
            "eval_context_docs_count": len(eval_context_docs),
            "eval_context_chars": eval_context_chars,
            "context_max_chars_per_doc": max_chars_per_doc,
            "retrieval_seconds": retrieval_seconds,
            "answer_generation_seconds": answer_generation_seconds,
            "planning_auto_context_used": use_planning_mode,
            "is_explanatory_query": is_explanatory_query,
            "retrieval_augmentation_docs_added": retrieval_augmentation_docs_added,
            "target_planning_document_id": force_planning_document_id,
            "planning_docs_count": len(docs) if use_planning_mode else 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return TurnTrace(
            input=input_text,
            conversation_history_used=history_payload,
            rewritten_query=rewritten_query,
            retrieval_context=[d.page_content for d in eval_context_docs],
            retrieved_doc_ids=self._extract_doc_ids(eval_context_docs),
            retrieval_scores=eval_context_scores,
            actual_output=answer,
            metadata=metadata,
        )

    async def run_conversation(
        self,
        conversation_id: str,
        turns: list[dict[str, str]],
    ) -> ConversationTrace:
        await self.initialize()

        history: list[dict[str, str]] = []
        traces: list[TurnTrace] = []

        for turn in turns:
            role = (turn.get("role") or "").lower().strip()
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                turn_trace = await self.run_single_turn(
                    input_text=content,
                    conversation_history=history,
                    top_k=self.top_k,
                )
                traces.append(turn_trace)
                history.append({"role": "user", "content": content})
                history.append({"role": "assistant", "content": turn_trace.actual_output})
            elif role == "assistant":
                # Keep explicit scripted assistant turns if present in the golden conversation.
                history.append({"role": "assistant", "content": content})

        return ConversationTrace(conversation_id=conversation_id, turns=traces)

    async def _retrieve(
        self,
        question: str,
        filters: dict[str, Any],
        top_k: int,
    ) -> tuple[list[Document], list[float | None]]:
        retriever = build_retriever(self._vs, k=top_k, filters=filters)

        if hasattr(retriever, "ainvoke_with_scores"):
            docs, scores = await retriever.ainvoke_with_scores(question)
            return docs, scores

        docs: list[Document] = await retriever.ainvoke(question)

        # Prefer explicit relevance scores when vector store supports it.
        scores: list[float | None]
        scores = [None for _ in docs]

        if hasattr(self._vs, "asimilarity_search_with_relevance_scores"):
            try:
                kwargs: dict[str, Any] = {"k": top_k}
                metadata_filter = build_metadata_filter(filters or {})
                if metadata_filter:
                    kwargs["filter"] = metadata_filter
                scored = await self._vs.asimilarity_search_with_relevance_scores(question, **kwargs)
                score_map = {
                    self._stable_doc_key(doc): float(score)
                    for doc, score in scored
                }
                scores = [score_map.get(self._stable_doc_key(doc)) for doc in docs]
            except Exception:
                # Keep None scores when backend cannot provide relevance scores.
                pass

        return docs, scores

    async def _retrieve_with_fallbacks(
        self,
        question: str,
        filters: dict[str, Any],
        base_filters: dict[str, Any],
        top_k: int,
        prefer_broader_context: bool = False,
    ) -> tuple[list[Document], list[float | None], str]:
        attempts: list[tuple[str, dict[str, Any]]] = []
        exact_filters = dict(filters or {})
        attempts.append(("exact_filters", exact_filters))

        target_only_filters = {
            key: exact_filters[key]
            for key in ("postId", "propertyId", "planningDocumentId")
            if key in exact_filters
        }
        if target_only_filters and target_only_filters != exact_filters:
            attempts.append(("target_only", target_only_filters))

        relaxed_exact = dict(exact_filters)
        relaxed_exact.pop("district", None)
        relaxed_exact.pop("city", None)
        if relaxed_exact and relaxed_exact != exact_filters and relaxed_exact != target_only_filters:
            attempts.append(("relaxed_exact_filters", relaxed_exact))

        if base_filters != exact_filters:
            attempts.append(("base_filters", dict(base_filters or {})))

        relaxed_base = dict(base_filters or {})
        relaxed_base.pop("district", None)
        if relaxed_base and relaxed_base != base_filters and relaxed_base != exact_filters:
            attempts.append(("relaxed_no_district", relaxed_base))

        attempts.append(("semantic_only", {}))

        seen_signature: set[tuple[tuple[str, Any], ...]] = set()
        for strategy, attempt_filters in attempts:
            signature = tuple(sorted((k, repr(v)) for k, v in (attempt_filters or {}).items()))
            if signature in seen_signature:
                continue
            seen_signature.add(signature)

            attempt_top_k = top_k
            if prefer_broader_context and strategy in {"exact_filters", "target_only"}:
                attempt_top_k = max(top_k, 16)

            docs, scores = await self._retrieve(question, filters=attempt_filters, top_k=attempt_top_k)
            if docs:
                return docs, scores, strategy

        return [], [], "empty"

    async def _augment_explanatory_target_docs(
        self,
        question: str,
        raw_input: str,
        filters: dict[str, Any],
        docs: list[Document],
        scores: list[float | None],
        top_k: int,
        retrieval_strategy: str,
    ) -> tuple[list[Document], list[float | None]]:
        if retrieval_strategy not in {"exact_filters", "target_only"}:
            return docs, scores

        post_id = self._coerce_int((filters or {}).get("postId"))
        property_id = self._coerce_int((filters or {}).get("propertyId"))
        if post_id is None and property_id is None:
            return docs, scores

        current_chars = sum(len(doc.page_content or "") for doc in docs)
        if len(docs) >= 3 and current_chars >= 1600:
            return docs, scores

        target_docs = list(docs)
        target_scores = list(scores)
        seen = {self._doc_metadata_identity(doc) for doc in target_docs}
        hard_cap_docs = max(top_k, 8)
        hard_cap_chars = 3200

        probe_filters: list[dict[str, Any]] = []
        if post_id is not None:
            probe_filters.append({"postId": post_id})
        if property_id is not None:
            probe_filters.append({"propertyId": property_id})
        if post_id is not None and property_id is not None:
            probe_filters.append({"postId": post_id, "propertyId": property_id})

        probe_queries = [
            question,
            raw_input,
            f"{raw_input} dac diem vi tri tien ich noi that",
        ]

        for attempt_filters in probe_filters:
            for probe_query in probe_queries:
                extra_docs, extra_scores = await self._retrieve(
                    probe_query,
                    filters=attempt_filters,
                    top_k=max(top_k, 16),
                )
                for doc, score in zip(extra_docs, extra_scores):
                    identity = self._doc_metadata_identity(doc)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    target_docs.append(doc)
                    target_scores.append(score)
                    current_chars += len(doc.page_content or "")
                    if len(target_docs) >= hard_cap_docs or current_chars >= hard_cap_chars:
                        return target_docs, target_scores

        return target_docs, target_scores

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _merge_target_metadata_filters(
        self,
        extracted_filters: dict[str, Any],
        target_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(extracted_filters or {})

        target_post_id = self._coerce_int((target_metadata or {}).get("postId"))
        if target_post_id is not None:
            merged["postId"] = target_post_id

        target_property_id = self._coerce_int((target_metadata or {}).get("propertyId"))
        if target_property_id is not None:
            merged["propertyId"] = target_property_id

        return merged

    @staticmethod
    def _target_metadata_suggests_planning(target_metadata: dict[str, Any] | None) -> bool:
        if not target_metadata:
            return False

        if target_metadata.get("postId") is not None or target_metadata.get("propertyId") is not None:
            return False

        planning_keys = {
            "planningDocumentId",
            "dossierCode",
            "planYear",
            "totalProjects2025",
            "approvedProjects2025",
            "approvedArea2025Ha",
            "totalArea2025Ha",
            "naturalAreaHa",
            "administrativeUnits",
            "agriculturalRecovery2025Ha",
            "nonAgriculturalRecovery2025Ha",
            "unusedLandToUse2025Ha",
            "totalRecovery2025Ha",
            "unused2024Ha",
            "project",
            "projectName",
            "ward",
            "areaHa",
            "article67ProjectCount",
        }
        return any(key in target_metadata for key in planning_keys)

    async def _generate_answer(
        self,
        question: str,
        context: str,
        history: list[Any],
        context_docs: list[Document],
    ) -> str:
        safe_question = sanitize_llm_text(question, max_len=3000)
        safe_context = sanitize_llm_text(context)
        safe_history = self._sanitize_history_messages(history)
        answer_language = detect_lang(safe_question)
        chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
                "history": lambda x: x["history"],
                "answer_language": lambda x: x["answer_language"],
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )
        raw_answer = await chain.ainvoke(
            {
                "question": safe_question,
                "context": safe_context,
                "history": safe_history,
                "answer_language": answer_language,
            }
        )
        return postprocess_answer(safe_question, raw_answer, context_docs)

    @staticmethod
    def _sanitize_history_messages(history: list[Any]) -> list[Any]:
        sanitized: list[Any] = []
        for message in history or []:
            if isinstance(message, HumanMessage):
                sanitized.append(HumanMessage(content=sanitize_llm_text(message.content)))
            elif isinstance(message, AIMessage):
                sanitized.append(AIMessage(content=sanitize_llm_text(message.content)))
            else:
                sanitized.append(message)
        return sanitized

    @staticmethod
    def _to_langchain_messages(history: list[dict[str, str]]) -> list[Any]:
        messages: list[Any] = []
        for item in history:
            role = (item.get("role") or "").lower().strip()
            content = sanitize_llm_text(item.get("content") or "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def _extract_doc_ids(docs: list[Document]) -> list[str]:
        ids: list[str] = []
        for doc in docs:
            md = doc.metadata or {}
            parts = [
                str(md.get("postId") or md.get("planningDocumentId") or "unknown"),
                str(md.get("chunkType") or "chunk"),
                str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or "na"),
            ]
            ids.append("_".join(parts))
        return ids

    @staticmethod
    def _stable_doc_key(doc: Document) -> str:
        md = doc.metadata or {}
        return "|".join(
            [
                str(md.get("postId") or ""),
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(md.get("globalChunkIndex") or md.get("chunkIndex") or ""),
                (doc.page_content or "")[:120],
            ]
        )

    @staticmethod
    def _doc_metadata_identity(doc: Document) -> str:
        md = doc.metadata or {}
        return "|".join(
            [
                str(md.get("postId") or ""),
                str(md.get("propertyId") or ""),
                str(md.get("planningDocumentId") or ""),
                str(md.get("chunkType") or ""),
                str(md.get("globalChunkIndex") if md.get("globalChunkIndex") is not None else md.get("chunkIndex") or ""),
            ]
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = (text or "").lower()
        lowered = unicodedata.normalize("NFD", lowered)
        lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
        lowered = lowered.replace("đ", "d")
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    def _prepare_eval_context_docs(
        self,
        question: str,
        docs: list[Document],
        use_planning_mode: bool = False,
        is_explanatory_query: bool = False,
    ) -> list[Document]:
        max_docs = self.eval_context_max_docs
        if use_planning_mode or _has_planning_intent(question):
            max_docs = max(max_docs, self.planning_eval_context_max_docs)
        if is_explanatory_query and not use_planning_mode:
            max_docs = max(max_docs, self.eval_explanatory_max_docs)
        return docs[:max_docs]

    def _is_explanatory_question(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        if not normalized:
            return False

        return any(
            marker in normalized
            for marker in (
                "vi sao",
                "ly do",
                "nhu the nao",
                "duoc xem la",
                "phu hop",
                "ket hop o va",
                "uu diem",
                "han che",
            )
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs if d.page_content)

def turn_trace_to_dict(trace: TurnTrace) -> dict[str, Any]:
    return asdict(trace)


def conversation_trace_to_dict(trace: ConversationTrace) -> dict[str, Any]:
    return {
        "conversation_id": trace.conversation_id,
        "turns": [asdict(turn) for turn in trace.turns],
    }
