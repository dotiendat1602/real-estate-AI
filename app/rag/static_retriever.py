from __future__ import annotations

from langchain_core.documents import Document


class StaticDocumentsRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    async def ainvoke(self, query: str) -> list[Document]:
        return self.docs
