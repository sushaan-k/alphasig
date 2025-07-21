"""Supply-chain knowledge graph built on NetworkX.

Provides traversal, exposure analysis, and plotting for the supplier /
customer / partner graph extracted by the supply-chain engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import structlog

from sigint.models import SupplyChainEdge

logger = structlog.get_logger()


class SupplyChainGraph:
    """Directed graph of supply-chain relationships.

    Each node is a company (identified by ticker or name) and each edge
    represents a dependency extracted from an SEC filing.

    Args:
        edges: Initial set of supply-chain edges.
    """

    def __init__(self, edges: Sequence[SupplyChainEdge] | None = None) -> None:
        self._graph = nx.MultiDiGraph()
        if edges:
            self.add_edges(edges)

    # -- Graph construction ---------------------------------------------------

    def add_edges(self, edges: Sequence[SupplyChainEdge]) -> None:
        """Add edges to the graph, merging duplicates by confidence.

        If an edge (source -> target, relation) already exists, we keep
        the one with higher confidence.
        """
        for edge in edges:
            relation_key = edge.relation.value
            existing = self._graph.get_edge_data(
                edge.source,
                edge.target,
                key=relation_key,
            )
            if (
                existing is not None
                and existing.get("confidence", 0) >= edge.confidence
            ):
                continue
            self._graph.add_edge(
                edge.source,
                edge.target,
                key=relation_key,
                relation=edge.relation.value,
                context=edge.context,
                confidence=edge.confidence,
                filing_type=edge.filing_type.value,
                filed_date=edge.filed_date.isoformat(),
            )

    # -- Queries --------------------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        """All company identifiers in the graph."""
        return list(self._graph.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return int(self._graph.number_of_edges())

    def suppliers_of(self, ticker: str) -> list[dict[str, Any]]:
        """Return all entities that *ticker* depends on.

        Args:
            ticker: Company whose suppliers to retrieve.

        Returns:
            List of dicts with ``target``, ``relation``, ``context``,
            ``confidence``.
        """
        results: list[dict[str, Any]] = []
        for _, target, _, data in self._graph.out_edges(ticker, keys=True, data=True):
            results.append({"target": target, **data})
        return sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

    def customers_of(self, ticker: str) -> list[dict[str, Any]]:
        """Return all entities that depend on *ticker*.

        This is the reverse query: who lists *ticker* as a dependency?
        """
        results: list[dict[str, Any]] = []
        for source, _, _, data in self._graph.in_edges(ticker, keys=True, data=True):
            results.append({"source": source, **data})
        return sorted(results, key=lambda r: r.get("confidence", 0), reverse=True)

    def exposure(self, ticker: str) -> dict[str, Any]:
        """Analyse which companies are exposed to *ticker*.

        Computes direct and transitive (1-hop) dependents.

        Args:
            ticker: The supplier/entity to check exposure for.

        Returns:
            Dict with ``direct_dependents``, ``transitive_dependents``,
            and ``exposure_score``.
        """
        direct = [src for src, _ in self._graph.in_edges(ticker)]

        transitive: set[str] = set()
        for dep in direct:
            for src, _ in self._graph.in_edges(dep):
                if src != ticker:
                    transitive.add(src)

        total = len(direct) + len(transitive)
        score = min(1.0, total / max(self._graph.number_of_nodes(), 1))

        return {
            "entity": ticker,
            "direct_dependents": sorted(direct),
            "transitive_dependents": sorted(transitive),
            "total_exposed": total,
            "exposure_score": round(score, 4),
        }

    def most_connected(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the most connected nodes by total degree.

        Args:
            top_n: Number of results.

        Returns:
            List of (node, degree) tuples in descending order.
        """
        degrees: list[tuple[str, int]] = sorted(
            self._graph.degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        return degrees[:top_n]

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find the shortest dependency path between two entities.

        Returns:
            List of node identifiers forming the path, or ``None``
            if no path exists.
        """
        try:
            path: list[str] = nx.shortest_path(self._graph, source, target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a JSON-compatible dict."""
        data: dict[str, Any] = nx.node_link_data(self._graph)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupplyChainGraph:
        """Reconstruct a graph from :meth:`to_dict` output."""
        g = cls()
        g._graph = nx.node_link_graph(data)
        return g

    # -- Visualisation --------------------------------------------------------

    def plot(
        self,
        output_path: str | None = None,
        figsize: tuple[int, int] = (16, 12),
    ) -> Any:
        """Render the graph using matplotlib.

        Args:
            output_path: If given, save the figure to this path instead
                of showing it interactively.
            figsize: Figure dimensions in inches.

        Returns:
            The matplotlib figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            ) from exc

        fig, ax = plt.subplots(figsize=figsize)

        pos = nx.spring_layout(self._graph, seed=42, k=2.0)

        # Node sizing by degree
        degrees = dict(self._graph.degree())
        node_sizes = [300 + 200 * degrees.get(n, 0) for n in self._graph.nodes]

        nx.draw_networkx_nodes(
            self._graph,
            pos,
            node_size=node_sizes,
            node_color="#4A90D9",
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_labels(
            self._graph, pos, font_size=8, font_weight="bold", ax=ax
        )

        # Colour edges by relation type
        edge_colors = []
        for _, _, data in self._graph.edges(data=True):
            relation = data.get("relation", "depends_on")
            if relation == "depends_on":
                edge_colors.append("#E74C3C")
            elif relation == "supplies_to":
                edge_colors.append("#2ECC71")
            else:
                edge_colors.append("#95A5A6")

        nx.draw_networkx_edges(
            self._graph,
            pos,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=15,
            alpha=0.6,
            ax=ax,
        )

        ax.set_title("Supply Chain Dependency Graph", fontsize=14)
        ax.axis("off")
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("graph_saved", path=output_path)

        return fig
