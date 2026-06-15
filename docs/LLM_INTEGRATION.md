# LLM Integration

OpenMatchER uses a single provider interface:

```python
class LLMProvider:
    async def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult:
        ...
```

Supported providers:

- OpenAI
- Anthropic
- Gemini
- Ollama
- OpenRouter

API keys are stored encrypted in the backend database. Provider responses are validated with Pydantic structured output models before they can affect a match decision.

When `use_llm` is enabled on a run, OpenMatchER first runs the deterministic matching engine. It then sends only uncertain candidate pairs to the selected provider. The default review band is `0.70` to `0.88`, which keeps obvious matches and obvious non-matches out of the LLM path.

For reviewed pairs, the LLM returns:

```json
{
  "match": true,
  "confidence": 0.93,
  "reasoning": "The names and addresses refer to the same company."
}
```

The final score becomes the LLM confidence for a match, or `1 - confidence` for a non-match. Clusters and run metrics are recalculated after the LLM-adjusted scores are applied.

LLM review should be reserved for uncertain candidate pairs because deterministic features are cheaper, faster, and easier to audit.
