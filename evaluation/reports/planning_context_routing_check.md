# Planning Context Routing Check

## Conclusion
- Yes, the chatbot currently auto-attaches planning vector context mainly when planning intent is detected from query text.
- Planning context is also attached when `planningContexts` is provided explicitly by backend payload.

## Code-path summary
1. Always-on retriever:
- Main retriever (`vs`) is executed for every chat request.
- This retriever uses general vector collection configured by `PGVECTOR_COLLECTION`.

2. Planning branch A (explicit):
- If `planningContexts` is present in request body, planning retriever is executed and planning context is appended.

3. Planning branch B (auto):
- If `planningContexts` is absent and `_has_planning_intent(message)` is true, planning retriever is executed and planning context is appended.

## Update applied
- Expanded planning intent detection beyond narrow phrase matching.
- Added more aliases/synonyms for planning language (including variations for ke hoach su dung dat / quy hoach).
- Added structural heuristic: if message contains planning-structural terms and also has district hint or plan year, it is treated as planning intent.

## Practical meaning
- "Kế hoạch sử dụng đất" and "quy hoạch" are now both covered as planning intent signals.
- Queries without exact keywords can still route to planning context when they contain strong structural planning cues.

## Remaining limitation
- Auto planning context still depends on intent inference from user text unless `planningContexts` is provided.
- If a planning-related query is phrased too ambiguously, it may not trigger planning branch.

## Recommendation
- For high-stakes planning UX, keep sending structured `planningContexts` from backend whenever available.
- Keep auto-intent branch as fallback for natural-language planning queries.
