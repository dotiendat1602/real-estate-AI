from langchain_core.prompts import ChatPromptTemplate

SYSTEM = """You are Estatein's real-estate assistant.
Rules:
- Answer using ONLY the provided context. If insufficient, ask 1-2 clarification questions.
- Be concise, practical.
- When citing listings, include post_id and a short reason.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "USER MESSAGE:\n{question}\n\nCONTEXT:\n{context}"),
    ]
)
