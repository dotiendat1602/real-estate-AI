from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.rag.prompt import prompt


@dataclass
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]


def _build_citations(docs: list[Document]) -> list[dict[str, Any]]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append(
            {
                "postId": md.get("postId"),
                "score": md.get("score"),
                "metadata": md,
                "snippet": (d.page_content or "")[:240],
            }
        )
    return out


class RagChain:
    def __init__(self, llm, retriever):
        """
        llm: LangChain chat model (OpenAI/local)
        retriever: VectorStoreRetriever (ưu tiên loại có .ainvoke(query) -> list[Document])
        """
        self.llm = llm
        self.retriever = retriever

        # LCEL chain:
        # question -> retriever lấy docs -> prompt nhét context -> llm -> parse string
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "context": self.retriever,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def run(self, question: str) -> ChatResult:
        # Lấy docs riêng để làm citations (vì chain trên chỉ output string)
        # Retriever mới thường support .ainvoke
        docs: list[Document] = await self.retriever.ainvoke(question)

        answer: str = await self.chain.ainvoke(question)
        return ChatResult(answer=answer, citations=_build_citations(docs))
