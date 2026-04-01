from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.api.chat import _has_planning_intent, _retrieve_planning_docs_for_nl_query
from app.api.chat import build_llm
from app.rag.chain import detect_lang, postprocess_answer
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
    ) -> TurnTrace:
        await self.initialize()

        history_payload = (conversation_history or [])[-self.max_history_turns :]
        history_messages = self._to_langchain_messages(history_payload)

        filters = await extract_filters_from_query(input_text, self._llm)
        k = top_k or self.top_k

        use_planning_mode = _has_planning_intent(input_text)

        if use_planning_mode:
            docs = await _retrieve_planning_docs_for_nl_query(
                self._planning_vs,
                input_text,
                k,
                history_messages=history_payload,
            )
            scores = [None] * len(docs)
        else:
            docs, scores = await self._retrieve(input_text, filters=filters, top_k=k)

        context_text = self._format_docs(docs)

        answer = await self._generate_answer(
            question=input_text,
            context=context_text,
            history=history_messages,
        )

        metadata = {
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "judge_phase": self.settings.get("runtime", {}).get("judge_phase", "offline_eval"),
            "prompt_version": self.settings.get("versions", {}).get("prompt_version", "rag_prompt_v1"),
            "retriever_version": self.settings.get("versions", {}).get("retriever_version", "pgvector_similarity_v1"),
            "chunking_version": self.settings.get("versions", {}).get("chunking_version", "chunking_v1"),
            "top_k": k,
            "filters": filters,
            "planning_auto_context_used": use_planning_mode,
            "planning_docs_count": len(docs) if use_planning_mode else 0,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return TurnTrace(
            input=input_text,
            conversation_history_used=history_payload,
            rewritten_query=None,
            retrieval_context=[d.page_content for d in docs],
            retrieved_doc_ids=self._extract_doc_ids(docs),
            retrieval_scores=scores,
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

    async def _generate_answer(self, question: str, context: str, history: list[Any]) -> str:
        answer_language = detect_lang(question)
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
                "question": question,
                "context": context,
                "history": history,
                "answer_language": answer_language,
            }
        )
        return postprocess_answer(question, raw_answer)

    @staticmethod
    def _to_langchain_messages(history: list[dict[str, str]]) -> list[Any]:
        messages: list[Any] = []
        for item in history:
            role = (item.get("role") or "").lower().strip()
            content = (item.get("content") or "").strip()
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
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs if d.page_content)


def turn_trace_to_dict(trace: TurnTrace) -> dict[str, Any]:
    return asdict(trace)


def conversation_trace_to_dict(trace: ConversationTrace) -> dict[str, Any]:
    return {
        "conversation_id": trace.conversation_id,
        "turns": [asdict(turn) for turn in trace.turns],
    }
