#ifndef BOUNDARY_CUT_H
#define BOUNDARY_CUT_H

#include <vector>
#include <memory>
#include "graph.h"

// BoundaryCut: C++ 实现的 "boundarycut"(QF) 网络瓦解算法。
// 该算法反复利用 k-core 分解找出 coreness==1 的边界(枝)节点,
// 用并查集(merge2 语义)统计每个非 1-core 边界节点上挂载的枝结构规模,
// 以 规模/成本 作为优先级移除边界节点; 当最大连通片比例足够小后,
// 对剩余子图使用 HDA(自适应最高度) 完成瓦解。
class BoundaryCut
{
public:
    BoundaryCut();

    // boundarycut: QF 边界剥离 + HDA 收尾。
    // 输入: graph(邻接表 + 节点成本权重), gcc_threshold(最大连通片比例阈值, 默认 0.01)
    // 输出: 长度为 num_nodes 的完整移除顺序(0..n-1 的一个排列)
    std::vector<int> getSolution(std::shared_ptr<Graph> graph, double gcc_threshold = 0.01);

    // boundarycutHD: QF 边界剥离 + HD(非自适应最高度) 收尾。
    std::vector<int> getSolutionHD(std::shared_ptr<Graph> graph, double gcc_threshold = 0.01);

    // boundary_degree_cut: QF 边界剥离(优先级 = 节点原始度数) + HDA 收尾。
    std::vector<int> getSolutionDegree(std::shared_ptr<Graph> graph, double gcc_threshold = 0.01);

    // boundary_degree_cut_HD: QF 边界剥离(优先级 = 节点原始度数) + HD(非自适应最高度) 收尾。
    std::vector<int> getSolutionDegreeHD(std::shared_ptr<Graph> graph, double gcc_threshold = 0.01);

    // RENDCut 用: 仅执行 QF 边界剥离阶段, 把 HDA 收尾交给上层(如 CoE-FINDER)。
    // 返回 [individual, remaining, {need_finder}]。
    std::vector<std::vector<int> > getQFSolution(std::shared_ptr<Graph> graph, double gcc_threshold = 0.01);
};

#endif
