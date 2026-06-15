# Entity Resolution with Spark

Spark is useful for normalization, blocking-key generation, candidate-pair joins, and feature-vector materialization. OpenMatchER keeps business logic separate from Spark by placing distributed execution behind an adapter.

This lets local tests stay fast while production jobs use Spark DataFrames for partitioned candidate generation and scalable scoring.

