from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},

    #################for ubuntu compile
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['PrepareBatchGraph.pyx','lib/PrepareBatchGraph.cpp','lib/graph.cpp','lib/graph_struct.cpp',  'lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['graph.pyx', 'lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['mvc_env.pyx', 'lib/mvc_env.cpp', 'lib/graph.cpp','lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['utils.pyx', 'lib/utils.cpp', 'lib/graph.cpp', 'lib/graph_utils.cpp', 'lib/disjoint_set.cpp', 'lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['nstep_replay_mem.pyx', 'lib/nstep_replay_mem.cpp', 'lib/graph.cpp', 'lib/mvc_env.cpp', 'lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['nstep_replay_mem_prioritized.pyx', 'lib/nstep_replay_mem_prioritized.cpp','lib/graph.cpp', 'lib/mvc_env.cpp', 'lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['graph_struct.pyx', 'lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('boundary_cut', sources=['boundary_cut.pyx', 'lib/boundary_cut.cpp', 'lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11'])
                   ])

