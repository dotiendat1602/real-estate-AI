from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import unicodedata

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ..rag.prompt import prompt


@dataclass
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]


_PLANNING_FACT_MARKERS = [
    "ke hoach su dung dat",
    "quy hoach",
    "ma ho so",
    "dossier",
    "nam nao",
    "ap dung",
    "khu vuc nao",
    "quan nao",
    "huyen nao",
]


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = unicodedata.normalize("NFD", lowered)
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _is_planning_fact_question(question: str) -> bool:
    q = _normalize_text(question)
    return any(marker in q for marker in _PLANNING_FACT_MARKERS)


def _looks_uncertain_or_no_data(answer: str) -> bool:
    a = _normalize_text(answer)
    signals = [
        "không biết",
        "khong biet",
        "không có trong context",
        "khong co trong context",
        "không đủ dữ liệu",
        "khong du du lieu",
        "i don't know",
        "insufficient",
        "not enough information",
    ]
    return any(signal in a for signal in signals)


def postprocess_answer(question: str, answer: str) -> str:
    if not answer:
        return answer

    cleaned = answer.strip()

    # Remove common generic trailing invites that hurt factual relevancy metrics.
    generic_tail_patterns = [
        r"\s*Nếu bạn cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*Nếu cần thêm[^.!?]*[.!?]?\s*$",
        r"\s*If you need more information[^.!?]*[.!?]?\s*$",
    ]
    for pattern in generic_tail_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if not _is_planning_fact_question(question):
        return cleaned

    if _looks_uncertain_or_no_data(cleaned):
        return cleaned

    return cleaned


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
    seen_post_ids = set()
    
    for d in docs:
        md = d.metadata or {}
        post_id = md.get("postId")
        
        # Deduplicate by postId (avoid showing same post multiple times)
        if post_id and post_id in seen_post_ids:
            continue
        
        if post_id:
            seen_post_ids.add(post_id)
        
        out.append({
            "postId": post_id,
            "propertyId": md.get("propertyId"),
            "planningDocumentId": md.get("planningDocumentId"),
            "title": md.get("title"),
            "sourceUrl": md.get("sourceUrl"),
            "format": md.get("format"),
            "documentScope": md.get("documentScope"),
            "documentType": md.get("documentType"),
            "dossierCode": md.get("dossierCode"),
            "planYear": md.get("planYear"),
            "chunkType": md.get("chunkType"),
            "chunkIndex": md.get("chunkIndex"),
            "globalChunkIndex": md.get("globalChunkIndex"),
            "pageNumber": md.get("pageNumber"),
            "lineStart": md.get("lineStart"),
            "lineEnd": md.get("lineEnd"),
            "sourceLocator": md.get("sourceLocator"),
            "chunker": md.get("chunker"),
            "postType": md.get("postType"),
            "categoryName": md.get("categoryName"),
            "city": md.get("city"),
            "district": md.get("district"),
            "ward": md.get("ward"),
            "price": md.get("price"),
            "area": md.get("area"),
            "bedrooms": md.get("bedrooms"),
            "amenities": md.get("amenities", []),
            "snippet": (d.page_content or "")[:300],
        })
    
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
                "history": lambda x: x["history"],
                "answer_language": lambda x: x["answer_language"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def run(self, question: str, history: list = None, extra_context: str = "") -> ChatResult:
        if history is None:
            history = []
            
        docs: list[Document] = await self.retriever.ainvoke(question)
        context = _format_docs(docs)
        if extra_context:
            context = f"{context}\n\n{extra_context}" if context else extra_context

        answer_language = detect_lang(question)

        answer: str = await self.chain.ainvoke(
            {
                "question": question,
                "context": context,
                "history": history,
                "answer_language": answer_language,
            }
        )

        answer = postprocess_answer(question, answer)

        return ChatResult(answer=answer, citations=_build_citations(docs))
