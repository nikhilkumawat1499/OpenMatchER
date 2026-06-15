# LLM Assisted Record Linkage

LLMs are most useful for ambiguous pairs, not for every candidate pair. OpenMatchER first computes deterministic features, then routes only uncertain pairs to an adjudication provider.

This keeps cost bounded, preserves auditability, and gives human reviewers natural-language reasoning for difficult decisions.

