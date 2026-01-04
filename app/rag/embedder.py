from __future__ import annotations

import os
from langchain_huggingface import HuggingFaceEmbeddings

def build_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    device = os.getenv("EMBED_DEVICE", "cpu")

    # normalize_embeddings=True để cosine similarity ổn
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
