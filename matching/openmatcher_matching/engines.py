from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .pipeline import MatchingConfig, run_resolution


@dataclass(frozen=True)
class EngineRunRequest:
    records: list[dict[str, Any]]
    config: MatchingConfig


class ExecutionEngine(ABC):
    name: str

    @abstractmethod
    def run(self, request: EngineRunRequest) -> dict[str, Any]:
        raise NotImplementedError


class LocalEngine(ExecutionEngine):
    name = "local"

    def run(self, request: EngineRunRequest) -> dict[str, Any]:
        result = run_resolution(request.records, request.config)
        result["engine"] = self.name
        return result


class SparkEngine(ExecutionEngine):
    name = "spark"

    def __init__(self, spark_session=None) -> None:
        self.spark = spark_session

    def run(self, request: EngineRunRequest) -> dict[str, Any]:
        if self.spark is None:
            try:
                from pyspark.sql import SparkSession
            except ImportError as exc:
                raise RuntimeError("Spark execution requires pyspark to be installed.") from exc
            self.spark = SparkSession.builder.appName("OpenMatchER").getOrCreate()

        frame = self.spark.createDataFrame(request.records)
        # Adapter boundary: Spark owns partitioning and filtering, the matching core owns scoring semantics.
        local_records = [row.asDict(recursive=True) for row in frame.collect()]
        result = run_resolution(local_records, request.config)
        result["engine"] = self.name
        result["spark_partitions"] = frame.rdd.getNumPartitions()
        return result


def engine_from_name(name: str) -> ExecutionEngine:
    if name == "local":
        return LocalEngine()
    if name == "spark":
        return SparkEngine()
    raise ValueError(f"Unsupported execution engine: {name}")

