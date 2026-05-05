from typing import Any


def build_splitter(chunk_size: int = 2000, chunk_overlap: int = 200) -> Any:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n=== ", "\n\n---", "\n\n", "\n", ". ", " ", ""],
    )
