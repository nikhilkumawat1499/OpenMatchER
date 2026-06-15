# Screenshot Guide

Capture these screenshots from the local UI before publishing the first release:

| File | What to show |
| --- | --- |
| `dashboard.png` | Project workspace with populated summary cards. |
| `dataset-upload-preview.png` | Upload panel plus a loaded demo dataset preview. |
| `pipeline-configuration.png` | Threshold, selected name column, and LLM uncertain-pair controls. |
| `cluster-review.png` | Matched pairs with scores, reasoning, and deterministic/LLM review labels. |
| `benchmark-leaderboard.png` | Benchmark leaderboard with micro and macro F1 columns. |
| `benchmark-report.png` | `reports/benchmark_results.html` with curves and scalability table. |

Suggested local setup:

```bash
PYTHONPATH=backend:matching:pipelines:llm:embeddings uvicorn app.main:app --reload --app-dir backend
cd frontend && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Then open `http://localhost:3000` and use one of the demo datasets from `examples/`.
