"""Tests for sigint.graph -- Supply-chain knowledge graph."""

from __future__ import annotations

from sigint.graph import SupplyChainGraph
from sigint.models import RelationType, SupplyChainEdge


class TestSupplyChainGraph:
    """Tests for the SupplyChainGraph class."""

    def test_construction(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        assert graph.edge_count == 4

    def test_nodes(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        nodes = graph.nodes
        assert "AAPL" in nodes
        assert "TSMC" in nodes
        assert "NVDA" in nodes

    def test_suppliers_of(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        suppliers = graph.suppliers_of("AAPL")
        targets = {s["target"] for s in suppliers}
        assert "TSMC" in targets
        assert "Foxconn" in targets

    def test_customers_of(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        customers = graph.customers_of("TSMC")
        sources = {c["source"] for c in customers}
        assert "AAPL" in sources
        assert "NVDA" in sources
        assert "AMD" in sources

    def test_exposure(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        exp = graph.exposure("TSMC")
        assert "AAPL" in exp["direct_dependents"]
        assert "NVDA" in exp["direct_dependents"]
        assert exp["total_exposed"] >= 3

    def test_most_connected(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        top = graph.most_connected(top_n=3)
        # TSMC has the most connections (3 incoming)
        assert top[0][0] == "TSMC"

    def test_shortest_path_exists(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        path = graph.shortest_path("AAPL", "TSMC")
        assert path is not None
        assert path == ["AAPL", "TSMC"]

    def test_shortest_path_not_found(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        path = graph.shortest_path("TSMC", "Foxconn")
        assert path is None

    def test_serialisation_roundtrip(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph(sample_edges)
        data = graph.to_dict()
        restored = SupplyChainGraph.from_dict(data)
        assert restored.edge_count == graph.edge_count
        assert set(restored.nodes) == set(graph.nodes)

    def test_empty_graph(self) -> None:
        graph = SupplyChainGraph()
        assert graph.edge_count == 0
        assert graph.nodes == []

    def test_add_edges(self, sample_edges: list[SupplyChainEdge]) -> None:
        graph = SupplyChainGraph()
        graph.add_edges(sample_edges[:2])
        assert graph.edge_count == 2
        graph.add_edges(sample_edges[2:])
        assert graph.edge_count == 4

    def test_duplicate_edge_keeps_higher_confidence(
        self, sample_edges: list[SupplyChainEdge]
    ) -> None:
        graph = SupplyChainGraph(sample_edges)
        # Add a duplicate with lower confidence
        low_conf = sample_edges[0].model_copy(update={"confidence": 0.1})
        graph.add_edges([low_conf])
        # Original should be preserved (higher confidence)
        suppliers = graph.suppliers_of("AAPL")
        tsmc = [s for s in suppliers if s["target"] == "TSMC"]
        assert tsmc[0]["confidence"] == 0.95

    def test_distinct_relations_between_same_nodes_are_preserved(
        self, sample_edges: list[SupplyChainEdge]
    ) -> None:
        graph = SupplyChainGraph(sample_edges)
        partner_edge = sample_edges[0].model_copy(
            update={
                "relation": RelationType.PARTNERS_WITH,
                "context": "packaging collaboration",
            }
        )
        graph.add_edges([partner_edge])

        suppliers = graph.suppliers_of("AAPL")
        tsmc_edges = [s for s in suppliers if s["target"] == "TSMC"]
        assert len(tsmc_edges) == 2
        assert {edge["relation"] for edge in tsmc_edges} == {
            "depends_on",
            "partners_with",
        }
