from __future__ import annotations

import os
from typing import Any


def build_llm() -> Any:
    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
