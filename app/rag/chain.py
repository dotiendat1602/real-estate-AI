from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import time

from .answer_processing import detect_lang, postprocess_answer
from .context_preparation import (
    build_citations as _build_citations,
    format_docs as _format_docs,
    prepare_docs_for_context,
)
from .llm_usage import extract_token_usage, message_content_to_text
from .prompt import prompt
from .query_rewrite import build_retrieval_query
from .text_utils import normalize_text as _normalize_text, sanitize_llm_text


@dataclass
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]
    token_usage: dict[str, Any] | None = None
    timings: dict[str, float] | None = None


class RagChain:
    def __init__(self, llm, retriever):
        """
        llm: LangChain chat model (ChatOpenAI)
        retriever: VectorStoreRetriever
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
                "history": lambda x: x["history"],
                "answer_language": lambda x: x["answer_language"],
            }
            | prompt
        )

    async def run(
        self,
        question: str,
        history: list | None = None,
        extra_context: str = "",
        *,
        max_docs: int | None = None,
        max_chars_per_doc: int | None = None,
    ) -> ChatResult:
        if history is None:
            history = []

        retrieval_query = build_retrieval_query(question, history)
        retrieval_started = time.perf_counter()
        docs = await self.retriever.ainvoke(retrieval_query or question)
        retrieval_seconds = round(time.perf_counter() - retrieval_started, 3)

        resolved_max_docs = int(max_docs) if max_docs is not None else int(os.getenv("RAG_CONTEXT_MAX_DOCS", "4"))
        resolved_max_chars_per_doc = (
            int(max_chars_per_doc)
            if max_chars_per_doc is not None
            else int(os.getenv("RAG_CONTEXT_MAX_CHARS_PER_DOC", "1400"))
        )
        context_selection_query = retrieval_query or question
        docs_for_context = prepare_docs_for_context(
            context_selection_query,
            docs,
            max_docs=resolved_max_docs,
            max_chars_per_doc=resolved_max_chars_per_doc,
        )
        context = _format_docs(docs_for_context)
        if extra_context:
            context = f"{context}\n\n{extra_context}" if context else extra_context

        payload = {
            "question": question,
            "context": context,
            "history": history,
            "answer_language": detect_lang(question),
        }
        answer_started = time.perf_counter()
        prompt_value = await self.prompt_chain.ainvoke(payload)
        ai_message = await self.llm.ainvoke(prompt_value)
        answer_generation_seconds = round(time.perf_counter() - answer_started, 3)

        answer = message_content_to_text(getattr(ai_message, "content", ""))
        answer = postprocess_answer(question, answer)

        return ChatResult(
            answer=answer,
            citations=_build_citations(docs_for_context),
            token_usage=extract_token_usage(ai_message),
            timings={
                "retrieval_seconds": retrieval_seconds,
                "answer_generation_seconds": answer_generation_seconds,
                "runtime_seconds": round(retrieval_seconds + answer_generation_seconds, 3),
            },
        )
