# Deployment

## Docker Compose

```bash
docker compose up --build
```

The default stack starts PostgreSQL, Redis, OpenSearch, the FastAPI backend, and the Next.js frontend.

## Environment variables

- `OPENMATCHER_DATABASE_URL`
- `OPENMATCHER_SECRET_KEY`
- `OPENMATCHER_ENCRYPTION_KEY`
- `OPENMATCHER_ALLOWED_ORIGINS`
- `OPENMATCHER_UPLOAD_DIR`
- `NEXT_PUBLIC_API_BASE`

For production, set a generated Fernet encryption key and a strong application secret.

