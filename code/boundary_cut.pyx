from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph

cdef class py_BoundaryCut:
    cdef shared_ptr[BoundaryCut] inner_BoundaryCut
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_BoundaryCut = shared_ptr[BoundaryCut](new BoundaryCut())

    cdef _fill_graph(self, _g):
        self.inner_Graph = shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).edge_list = _g.edge_list
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).nodes_weight = _g.nodes_weight
        deref(self.inner_Graph).total_nodes_weight = _g.total_nodes_weight

    def getSolution(self, _g, double gcc_threshold=0.01):
        self._fill_graph(_g)
        return deref(self.inner_BoundaryCut).getSolution(self.inner_Graph, gcc_threshold)

    def getSolutionHD(self, _g, double gcc_threshold=0.01):
        self._fill_graph(_g)
        return deref(self.inner_BoundaryCut).getSolutionHD(self.inner_Graph, gcc_threshold)

    def getSolutionDegree(self, _g, double gcc_threshold=0.01):
        self._fill_graph(_g)
        return deref(self.inner_BoundaryCut).getSolutionDegree(self.inner_Graph, gcc_threshold)

    def getSolutionDegreeHD(self, _g, double gcc_threshold=0.01):
        self._fill_graph(_g)
        return deref(self.inner_BoundaryCut).getSolutionDegreeHD(self.inner_Graph, gcc_threshold)

    def getQFSolution(self, _g, double gcc_threshold=0.01):
        # 返回 (individual, remaining, need_finder)
        self._fill_graph(_g)
        out = deref(self.inner_BoundaryCut).getQFSolution(self.inner_Graph, gcc_threshold)
        individual = list(out[0])
        remaining = list(out[1])
        need_finder = bool(out[2][0]) if len(out[2]) > 0 else False
        return individual, remaining, need_finder
