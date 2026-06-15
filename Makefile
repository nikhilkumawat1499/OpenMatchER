.PHONY: start stop test lint format benchmark

start:
	docker compose up --build

stop:
	docker compose down

test:
	PYTHONPATH=backend:matching:pipelines:llm:embeddings pytest --cov

lint:
	ruff check backend matching pipelines llm embeddings tests examples benchmarks
	cd frontend && npm run lint

format:
	ruff format backend matching pipelines llm embeddings tests examples benchmarks
	ruff check --fix backend matching pipelines llm embeddings tests examples benchmarks

benchmark:
	PYTHONPATH=backend:matching:pipelines:llm:embeddings python benchmarks/run_benchmarks.py

