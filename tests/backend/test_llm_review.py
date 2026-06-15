import asyncio

from app.services.resolution import review_uncertain_matches
from openmatcher_llm import AdjudicationRequest, AdjudicationResult, LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test-key", model="test-model")
        self.requests: list[AdjudicationRequest] = []

    async def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult:
        self.requests.append(request)
        return AdjudicationResult(match=True, confidence=0.93, reasoning="LLM sees the same address and similar name.")


async def _review(result: dict, provider: FakeProvider) -> dict:
    return await review_uncertain_matches(
        result=result,
        provider=provider,
        threshold=0.9,
        min_score=0.70,
        max_score=0.88,
    )


def test_llm_review_only_reviews_uncertain_pairs():
    provider = FakeProvider()
    result = {
        "records": [{"name": "Acme"}, {"name": "ACME"}, {"name": "Globex"}],
        "matches": [
            {
                "left_index": 0,
                "right_index": 1,
                "entity_a": {"name": "Acme"},
                "entity_b": {"name": "ACME"},
                "final_score": 0.82,
                "contributing_features": {"jaro_winkler": 0.82},
                "reasoning": "deterministic",
            },
            {
                "left_index": 0,
                "right_index": 2,
                "entity_a": {"name": "Acme"},
                "entity_b": {"name": "Globex"},
                "final_score": 0.42,
                "contributing_features": {"jaro_winkler": 0.42},
                "reasoning": "deterministic",
            },
        ],
        "clusters": [],
        "metrics": {},
    }

    reviewed = asyncio.run(_review(result, provider))

    assert len(provider.requests) == 1
    assert reviewed["matches"][0]["final_score"] == 0.93
    assert reviewed["matches"][0]["llm_review"]["reviewed"] is True
    assert reviewed["metrics"]["llm_reviewed_count"] == 1
    assert reviewed["metrics"]["llm_match_count"] == 1
    assert reviewed["metrics"]["match_count"] == 1
    assert reviewed["matches"][1]["llm_review"]["reason"] == "outside_review_band"
