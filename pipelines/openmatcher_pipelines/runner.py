from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineNode:
    name: str
    kind: str
    config: dict[str, Any]


class PipelineRunner:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {}

    def register(self, kind: str, handler: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]) -> None:
        self._handlers[kind] = handler

    def run(self, nodes: list[PipelineNode], initial_state: dict[str, Any]) -> dict[str, Any]:
        state = dict(initial_state)
        for node in nodes:
            if node.kind not in self._handlers:
                raise ValueError(f"No handler registered for pipeline node {node.kind}")
            state = self._handlers[node.kind](state, node.config)
            state.setdefault("events", []).append({"node": node.name, "kind": node.kind, "status": "completed"})
        return state

