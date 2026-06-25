from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from graph cimport Graph

cdef extern from "./lib/boundary_cut.h":
    cdef cppclass BoundaryCut:
        BoundaryCut()
        vector[int] getSolution(shared_ptr[Graph] graph, double gcc_threshold) except+
        vector[int] getSolutionHD(shared_ptr[Graph] graph, double gcc_threshold) except+
        vector[int] getSolutionDegree(shared_ptr[Graph] graph, double gcc_threshold) except+
        vector[int] getSolutionDegreeHD(shared_ptr[Graph] graph, double gcc_threshold) except+
        vector[vector[int]] getQFSolution(shared_ptr[Graph] graph, double gcc_threshold) except+
