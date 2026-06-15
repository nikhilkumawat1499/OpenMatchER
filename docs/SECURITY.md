# Security

OpenMatchER includes security foundations suitable for a public open-source project:

- Pydantic request validation.
- Encrypted API key storage with Fernet.
- CORS allow-list configuration.
- Rate limiting middleware.
- GitHub CodeQL workflow.
- Dependabot for dependency updates.
- Pre-commit lint and format hooks.

Production deployments should rotate secrets, configure TLS at the ingress, restrict database network access, and add organization-specific SSO or RBAC policies.

