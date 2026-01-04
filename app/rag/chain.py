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
    Robust language detection for Vietnamese (including no-diacritic text).

    Strategy:
    1) Use langid to classify. If it says 'vi' -> Vietnamese.
    2) If uncertain or very short, use heuristics:
       - Vietnamese diacritics -> Vietnamese
       - Common Vietnamese function words (no diacritics) -> Vietnamese
       - Else -> English
    """
    t = (text or "").strip()
    if not t:
        return "English"

    # 1) Try langid (local)
    try:
        import langid
        lang, score = langid.classify(t)
        # langid returns ISO 639-1 like "vi", "en"
        if lang == "vi":
            return "Vietnamese"
        if lang == "en":
            # vẫn có thể sai nếu câu rất ngắn, xử lý ở heuristic bên dưới
            pass
    except Exception:
        # nếu lib chưa cài hoặc lỗi runtime -> fallback heuristic
        pass

    lower = t.lower()

    # 2) Heuristic: diacritics
    vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    if any(c in lower for c in vi_chars):
        return "Vietnamese"

    # 3) Heuristic: Vietnamese stop-words without diacritics (very common)
    # NOTE: choose words that rarely appear in English.
    vi_markers = {
        "toi", "ban", "minh", "muon", "can", "tim", "kiem", "nha", "dat", "canho",
        "gia", "bao", "nhieu", "o", "tai", "quan", "huyen", "phuong", "duong",
        "dien", "tich", "phong", "ngu", "phap", "ly", "so", "do", "hop", "dong",
        "gan", "trung", "tam", "khoang", "trieu", "ty",
        "co", "khong", "neu", "thi", "va", "voi", "cho", "xin", "giup",
    }

    # tokenize đơn giản theo whitespace + strip punctuation
    tokens = []
    for w in lower.replace("\n", " ").split():
        w = "".join(ch for ch in w if ch.isalnum())
        if w:
            tokens.append(w)

    if not tokens:
        return "English"

    hit = sum(1 for w in tokens if w in vi_markers)
    # nếu câu ngắn, chỉ cần 1-2 marker là đủ
    if hit >= 2 or (len(tokens) <= 6 and hit >= 1):
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
