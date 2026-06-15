import networkx as nx


def connected_components(record_count: int, matches: list[dict], threshold: float) -> list[dict]:
    graph = nx.Graph()
    graph.add_nodes_from(range(record_count))
    for match in matches:
        if match["final_score"] >= threshold:
            graph.add_edge(match["left_index"], match["right_index"], weight=match["final_score"])
    return [
        {"cluster_id": f"cluster_{idx + 1}", "record_indices": sorted(component)}
        for idx, component in enumerate(nx.connected_components(graph))
    ]

