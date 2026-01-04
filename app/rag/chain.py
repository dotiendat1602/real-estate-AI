from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app.rag.prompt import prompt


@dataclass
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]


def detect_lang(text: str) -> str:
    """
    Very lightweight language detection:
    - If Vietnamese diacritics appear -> Vietnamese
    - Else -> English
    """
    t = (text or "").strip()
    if not t:
        return "English"

    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(c in t.lower() for c in vi_chars):
        return "Vietnamese"

    return "English"


def _build_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append(
            {
                "postId": md.get("postId"),
                # retriever thường không trả score mặc định => có thể None
                "score": md.get("score"),
                "metadata": md,
                "snippet": (d.page_content or "")[:240],
            }
        )
    return out


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)


class RagChain:
    def __init__(self, llm, retriever):
        """
        llm: LangChain chat model (ChatOpenAI)
        retriever: VectorStoreRetriever
        """
        self.llm = llm
        self.retriever = retriever

        # Build chain once (reuse per request)
        self.chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"],
                "answer_language": lambda x: x["answer_language"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def run(self, question: str) -> ChatResult:
        docs: list[Document] = await self.retriever.ainvoke(question)
        context = _format_docs(docs)

        answer_language = detect_lang(question)

        answer: str = await self.chain.ainvoke(
            {
                "question": question,
                "context": context,
                "answer_language": answer_language,
            }
        )

        return ChatResult(answer=answer, citations=_build_citations(docs))
