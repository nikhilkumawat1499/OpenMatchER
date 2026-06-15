# Building Entity Resolution at Scale

Entity resolution at scale is a search problem before it is a scoring problem. The central design challenge is to avoid comparing every record with every other record while preserving enough recall for downstream scoring.

OpenMatchER approaches this with normalization, blocking, explainable feature generation, and late-stage clustering. At higher volumes, the same semantics can move from local execution to Spark through the execution engine interface.

