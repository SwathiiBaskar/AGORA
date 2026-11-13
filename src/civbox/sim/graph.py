"""The social graph

A :class:`SocialGraph` is an immutable, directed, trust-weighted graph over ``N`` agents,
stored as a compressed sparse row (CSR) matrix ``T`` in canonical form:

* ``T[i, j] > 0`` means agent ``i`` listens to agent ``j`` with trust ``T[i, j]``, so row
  ``i`` lists the senders ``i`` listens to. This is the orientation of
  :class:`civbox.sim.pooling.InfluenceEdges`: receiver is the row, sender the column.
* Within each row the column indices are strictly increasing, so the stored order is
  ``(receiver, sender)`` ascending. There are no duplicate edges, no self-edges, and no
  stored zeros.
* Trust values lie in ``(0, 1]``.

Canonical form means the same graph always has the same arrays, whatever order its edges
were supplied in, so graphs can be compared and hashed by their bytes. The arrays are
read-only.

The graph is plain arrays and a SciPy matrix; nothing here iterates over Python objects per
node or edge, so it is safe to use inside a simulation step.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from civbox.sim.beliefs import FloatArray, IntArray, require_array
from civbox.sim.pooling import InfluenceEdges


def _readonly[A: np.ndarray](array: A) -> A:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SocialGraph:
    """Directed trust graph in canonical CSR form. See the module documentation."""

    n_agents: int
    indptr: IntArray
    indices: IntArray
    trust: FloatArray
    communities: IntArray | None = None

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def matrix(self) -> "sparse.csr_array":
        """The trust matrix as a SciPy CSR array sharing this graph's (read-only) arrays."""
        matrix = sparse.csr_array(
            (self.trust, self.indices, self.indptr),
            shape=(self.n_agents, self.n_agents),
            copy=False,
        )
        return matrix

    @classmethod
    def from_edges(
        cls,
        n_agents: int,
        receivers: IntArray,
        senders: IntArray,
        trust: FloatArray,
        communities: IntArray | None = None,
    ) -> Self:
        """Build a graph from edges given in any order.

        Raises ``ValueError`` for self-edges, repeated edges, endpoints outside
        ``[0, n_agents)``, or trust outside ``(0, 1]``.
        """
        if n_agents < 1:
            raise ValueError(f"n_agents must be at least 1, got {n_agents}")
        for name, array, dtype in (
            ("receivers", receivers, np.int64),
            ("senders", senders, np.int64),
            ("trust", trust, np.float64),
        ):
            require_array(name, array, ndim=1, dtype=dtype)
        count = receivers.shape[0]
        if senders.shape[0] != count or trust.shape[0] != count:
            raise ValueError("receivers, senders and trust must have the same length")
        if count:
            low = min(int(receivers.min()), int(senders.min()))
            high = max(int(receivers.max()), int(senders.max()))
            if low < 0 or high >= n_agents:
                raise ValueError(f"edge endpoints must lie in [0, {n_agents})")
            if (receivers == senders).any():
                raise ValueError("self-edges are not allowed")
        if not np.isfinite(trust).all() or (trust <= 0.0).any() or (trust > 1.0).any():
            raise ValueError("trust must lie in (0, 1]")

        order = np.lexsort((senders, receivers))
        rows, cols = receivers[order], senders[order]
        if count > 1 and ((rows[1:] == rows[:-1]) & (cols[1:] == cols[:-1])).any():
            raise ValueError("each (receiver, sender) pair may appear only once")

        indptr = np.zeros(n_agents + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n_agents), out=indptr[1:])
        return cls(
            n_agents=n_agents,
            indptr=_readonly(indptr),
            indices=_readonly(cols.astype(np.int64)),
            trust=_readonly(trust[order].astype(np.float64)),
            communities=_validated_communities(communities, n_agents),
        )

    def rows(self) -> IntArray:
        """The receiver of each stored edge, aligned with :attr:`indices` and :attr:`trust`."""
        expanded: IntArray = np.repeat(
            np.arange(self.n_agents, dtype=np.int64), np.diff(self.indptr)
        )
        return expanded

    def to_edges(self) -> InfluenceEdges:
        """The edges in the form the pooling step consumes."""
        return InfluenceEdges.from_arrays(
            self.n_agents, self.rows(), self.indices.copy(), self.trust.copy()
        )

    # --- structure -------------------------------------------------------------------------------

    def listening_degree(self) -> IntArray:
        """How many agents each agent listens to (row sizes)."""
        degree: IntArray = np.diff(self.indptr)
        return degree

    def audience_size(self) -> IntArray:
        """How many agents listen to each agent (column counts)."""
        sizes: IntArray = np.bincount(self.indices, minlength=self.n_agents).astype(np.int64)
        return sizes

    def _reverse_edge_exists(self) -> np.ndarray:
        """For each stored edge ``(i, j)``, whether the edge ``(j, i)`` is also stored.

        Edges are identified by the integer ``receiver * n + sender``. Because the storage
        order is canonical those keys are already ascending, so a reverse edge is found by
        binary search.
        """
        if self.n_edges == 0:
            return np.empty(0, dtype=np.bool_)
        receivers = self.rows()
        keys = receivers * self.n_agents + self.indices
        reverse = self.indices * self.n_agents + receivers
        position = np.minimum(np.searchsorted(keys, reverse), keys.size - 1)
        return np.asarray(keys[position] == reverse)

    def is_symmetric(self) -> bool:
        """Whether ``i`` listens to ``j`` exactly when ``j`` listens to ``i`` (trust may differ)."""
        return bool(self._reverse_edge_exists().all())

    def reciprocity(self) -> float:
        """Fraction of edges whose reverse edge also exists; 1.0 for an empty graph."""
        if self.n_edges == 0:
            return 1.0
        return float(self._reverse_edge_exists().mean())

    def n_weak_components(self) -> int:
        """Number of connected components, ignoring edge direction."""
        count, _ = csgraph.connected_components(self.matrix, directed=True, connection="weak")
        return int(count)

    # --- consistency -----------------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` unless the arrays are in canonical form."""
        n, e = self.n_agents, self.n_edges
        if self.indptr.shape != (n + 1,) or self.trust.shape != (e,):
            raise ValueError("invalid graph: array shapes disagree")
        if self.indptr[0] != 0 or self.indptr[-1] != e or (np.diff(self.indptr) < 0).any():
            raise ValueError("invalid graph: indptr is not a valid row pointer")
        if e:
            if int(self.indices.min()) < 0 or int(self.indices.max()) >= n:
                raise ValueError("invalid graph: an endpoint is out of range")
            rows = self.rows()
            if (rows == self.indices).any():
                raise ValueError("invalid graph: self-edge")
            same_row = rows[1:] == rows[:-1]
            if (same_row & (self.indices[1:] <= self.indices[:-1])).any():
                raise ValueError("invalid graph: columns are not strictly increasing within a row")
            if (
                not np.isfinite(self.trust).all()
                or (self.trust <= 0).any()
                or (self.trust > 1).any()
            ):
                raise ValueError("invalid graph: trust outside (0, 1]")
        _validated_communities(self.communities, n)


def _validated_communities(communities: IntArray | None, n_agents: int) -> IntArray | None:
    if communities is None:
        return None
    require_array("communities", communities, ndim=1, dtype=np.int64)
    if communities.shape != (n_agents,):
        raise ValueError("communities must have one label per agent")
    if communities.size and int(communities.min()) < 0:
        raise ValueError("community labels must be non-negative")
    return _readonly(communities.copy())
