from __future__ import annotations

from langchain_core.documents import Document


def choose_better_planning_fallback(
    current_docs: list[Document],
    candidate_docs: list[Document],
) -> list[Document]:
    """Keep the richer fallback candidate while preserving vector-ranked order."""
    if not candidate_docs:
        return current_docs

    if not current_docs or len(candidate_docs) > len(current_docs):
        return list(candidate_docs)

    return current_docs
