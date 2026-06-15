# Threat Model

## Assets

- Uploaded datasets.
- Provider API keys.
- Match results and review decisions.
- Audit logs and experiment artifacts.

## Trust Boundaries

- Browser to backend API.
- Backend to database.
- Backend to filesystem/object storage.
- Backend to LLM providers.
- Worker execution environment.

## Key Risks

- Plaintext secret leakage.
- Malicious uploads.
- Prompt injection through record content.
- Over-broad CORS in production.
- Excessive LLM spend from unbounded adjudication.
- Unauthorized review or export access.

## Controls

- Provider keys are encrypted before storage.
- Upload size is limited and file types are validated.
- Pydantic validates API payloads.
- Rate limiting is enabled.
- LLM review is opt-in and score-band limited.
- GitHub CodeQL and Dependabot run in CI.
- Audit logs capture dataset upload and review actions.

## Production Recommendations

- Replace development secrets before deployment.
- Add SSO-backed RBAC middleware.
- Use private object storage with short-lived signed URLs.
- Enable TLS and strict CORS.
- Add per-project authorization checks.
- Set LLM budget limits per workspace.

