from __future__ import annotations

from typing import Any

from .langchain_adapter import ConversationTrace, TurnTrace


def build_single_turn_test_case(trace: TurnTrace, golden: dict[str, Any]):
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=trace.input,
        actual_output=trace.actual_output,
        retrieval_context=trace.retrieval_context,
        context=golden.get("context"),
        expected_output=golden.get("expected_output") or "\n".join(golden.get("expected_output_outline", [])),
    )


def build_conversation_test_case(trace: ConversationTrace, golden: dict[str, Any]):
    from deepeval.test_case import ConversationalTestCase, Turn

    turns: list[Turn] = []
    for item in trace.turns:
        turns.append(Turn(role="user", content=item.input))
        turns.append(Turn(role="assistant", content=item.actual_output))

    return ConversationalTestCase(
        scenario=golden.get("scenario", "Conversation quality check"),
        expected_outcome=golden.get("expected_outcome", "Assistant stays consistent and grounded."),
        turns=turns,
        context=golden.get("context"),
        chatbot_role=golden.get(
            "chatbot_role",
            "RAG chatbot for real-estate and land-use-planning assistance",
        ),
    )
