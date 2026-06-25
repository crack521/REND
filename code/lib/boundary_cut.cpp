#include "boundary_cut.h"
#include <vector>
#include <set>
#include <algorithm>
#include <climits>

BoundaryCut::BoundaryCut() {}

// 并查集: 带路径压缩的查根
static int findRoot(std::vector<int>& uni, int node)
{
    int root = node;
    while (uni[root] != root) root = uni[root];
    while (uni[node] != root)
    {
        int nxt = uni[node];
        uni[node] = root;
        node = nxt;
    }
    return root;
}

// 计算 present 子图诱导子图的 core number(coreness)。
// 完全复刻 networkx.core_number 的 Batagelj-Zaversnik 算法,
// 返回长度为 n 的数组, 非 present 节点取 -1。
static std::vector<int> computeCoreNumber(int n,
                                          const std::vector<std::vector<int> >& adj,
                                          const std::vector<char>& present)
{
    std::vector<int> deg(n, -1);
    int maxdeg = 0;
    int numPresent = 0;
    for (int v = 0; v < n; ++v)
    {
        if (!present[v]) continue;
        ++numPresent;
        int d = 0;
        for (size_t k = 0; k < adj[v].size(); ++k)
            if (present[adj[v][k]]) ++d;
        deg[v] = d;
        if (d > maxdeg) maxdeg = d;
    }

    // bin[d] = 度数 < d 的节点个数(即度数为 d 的分组在有序数组中的起始下标)
    std::vector<int> bin(maxdeg + 2, 0);
    for (int v = 0; v < n; ++v)
        if (present[v]) bin[deg[v] + 1]++;
    for (int d = 1; d <= maxdeg + 1; ++d)
        bin[d] += bin[d - 1];
    // 现在 bin[d] = 度数 < d 的节点数 = 度数为 d 分组的起始下标

    std::vector<int> order(numPresent);
    std::vector<int> pos(n, -1);
    {
        std::vector<int> cursor = bin; // 复制一份用于填充
        for (int v = 0; v < n; ++v)
        {
            if (!present[v]) continue;
            int p = cursor[deg[v]]++;
            order[p] = v;
            pos[v] = p;
        }
    }

    for (int i = 0; i < numPresent; ++i)
    {
        int v = order[i];
        for (size_t k = 0; k < adj[v].size(); ++k)
        {
            int u = adj[v][k];
            if (!present[u]) continue;
            if (deg[u] > deg[v])
            {
                int du = deg[u];
                int pu = pos[u];
                int pw = bin[du];
                int w = order[pw];
                if (u != w)
                {
                    order[pu] = w; pos[w] = pu;
                    order[pw] = u; pos[u] = pw;
                }
                bin[du]++;
                deg[u]--;
            }
        }
    }
    return deg;
}

// present 子图中最大连通片的节点数
static int largestCC(int n,
                     const std::vector<std::vector<int> >& adj,
                     const std::vector<char>& present)
{
    std::vector<char> vis(n, 0);
    std::vector<int> stk;
    int best = 0;
    for (int s = 0; s < n; ++s)
    {
        if (!present[s] || vis[s]) continue;
        int sz = 0;
        stk.clear();
        stk.push_back(s);
        vis[s] = 1;
        while (!stk.empty())
        {
            int v = stk.back();
            stk.pop_back();
            ++sz;
            for (size_t k = 0; k < adj[v].size(); ++k)
            {
                int u = adj[v][k];
                if (present[u] && !vis[u])
                {
                    vis[u] = 1;
                    stk.push_back(u);
                }
            }
        }
        if (sz > best) best = sz;
    }
    return best;
}

// 桶排序加速的自适应最高度算法(HDA), 平局时选最小编号节点。
// 在 present 子图上移除全部节点, 返回移除顺序。
static std::vector<int> hdaBucket(int n,
                                  const std::vector<std::vector<int> >& adj,
                                  const std::vector<char>& present_in)
{
    std::vector<int> deg(n, 0);
    std::vector<char> removed(n, 0);
    int maxdeg = 0;
    int cnt = 0;
    for (int v = 0; v < n; ++v)
    {
        if (!present_in[v]) { removed[v] = 1; continue; }
        ++cnt;
        int d = 0;
        for (size_t k = 0; k < adj[v].size(); ++k)
            if (present_in[adj[v][k]]) ++d;
        deg[v] = d;
        if (d > maxdeg) maxdeg = d;
    }

    std::vector<std::set<int> > buckets(maxdeg + 1);
    for (int v = 0; v < n; ++v)
        if (present_in[v]) buckets[deg[v]].insert(v);

    std::vector<int> order;
    order.reserve(cnt);
    int curMax = maxdeg;
    int removedCount = 0;
    while (removedCount < cnt)
    {
        while (curMax >= 0 && buckets[curMax].empty()) --curMax;
        if (curMax < 0) break;
        int node = *buckets[curMax].begin(); // 平局选最小编号
        buckets[curMax].erase(buckets[curMax].begin());
        removed[node] = 1;
        ++removedCount;
        order.push_back(node);
        for (size_t k = 0; k < adj[node].size(); ++k)
        {
            int u = adj[node][k];
            if (!present_in[u] || removed[u]) continue;
            int old = deg[u];
            buckets[old].erase(u);
            deg[u] = old - 1;
            buckets[old - 1].insert(u);
        }
    }
    return order;
}

// HD(最高度, 非自适应): 一次性按 (度数, 节点编号) 降序排列 present 子图中的全部节点。
// 与 boundarycut.py 的 HD(): sorted(key=(deg, node), reverse=True) 行为一致。
static std::vector<int> hdStatic(int n,
                                 const std::vector<std::vector<int> >& adj,
                                 const std::vector<char>& present_in)
{
    std::vector<std::pair<int, int> > nodes; // (度数, 节点编号)
    for (int v = 0; v < n; ++v)
    {
        if (!present_in[v]) continue;
        int d = 0;
        for (size_t k = 0; k < adj[v].size(); ++k)
            if (present_in[adj[v][k]]) ++d;
        nodes.push_back(std::make_pair(d, v));
    }
    // 度数降序; 平局时节点编号降序(reverse=True 作用于整个元组)
    std::sort(nodes.begin(), nodes.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second > b.second;
              });
    std::vector<int> order;
    order.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
        order.push_back(nodes[i].second);
    return order;
}

// 把 QF 移除顺序与收尾移除顺序拼接成完整解(0..n-1 的排列)。
static std::vector<int> assembleSolution(int n,
                                         const std::vector<int>& individual,
                                         const std::vector<int>& remove_list)
{
    std::vector<int> solution;
    solution.reserve(n);
    std::vector<char> inSol(n, 0);
    for (size_t i = 0; i < individual.size(); ++i)
    {
        int v = individual[i];
        if (!inSol[v]) { solution.push_back(v); inSol[v] = 1; }
    }
    for (size_t i = 0; i < remove_list.size(); ++i)
    {
        int v = remove_list[i];
        if (!inSol[v]) { solution.push_back(v); inSol[v] = 1; }
    }
    for (int v = 0; v < n; ++v)
        if (!inSol[v]) { solution.push_back(v); inSol[v] = 1; }
    return solution;
}

// 由 Graph 构造去自环、去重边的简单图邻接表 + 节点首次出现顺序
static void buildGraphData(std::shared_ptr<Graph> graph,
                           std::vector<std::vector<int> >& adj,
                           std::vector<int>& appearOrder)
{
    int n = graph->num_nodes;
    const std::vector<std::vector<int> >& adjRaw = graph->adj_list;
    adj.assign(n, std::vector<int>());
    for (int v = 0; v < n; ++v)
    {
        std::vector<int> tmp;
        tmp.reserve(adjRaw[v].size());
        for (size_t k = 0; k < adjRaw[v].size(); ++k)
        {
            int u = adjRaw[v][k];
            if (u != v) tmp.push_back(u);
        }
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        adj[v] = tmp;
    }

    // 节点首次出现顺序(与 networkx G.nodes() 的插入顺序一致), 用于排序平局时的稳定决胜。
    const std::vector<std::pair<int, int> >& el = graph->edge_list;
    appearOrder.assign(n, INT_MAX);
    int appearCounter = 0;
    for (size_t i = 0; i < el.size(); ++i)
    {
        int a = el[i].first, b = el[i].second;
        if (a >= 0 && a < n && appearOrder[a] == INT_MAX) appearOrder[a] = appearCounter++;
        if (b >= 0 && b < n && appearOrder[b] == INT_MAX) appearOrder[b] = appearCounter++;
    }
    for (int v = 0; v < n; ++v)
        if (appearOrder[v] == INT_MAX) appearOrder[v] = appearCounter++;
}

// QF(boundarycut)边界剥离主循环。
// 输入: adj(简单图邻接), weight(节点成本), appearOrder, gcc_threshold
//       priorityMode - 0: 优先级 = rankCount/cost (boundarycut 默认);
//                      1: 优先级 = 节点原始度数 (boundary_degree_cut)
//       degree       - priorityMode==1 时使用的节点原始度数(简单图)
// 输出: individual(QF 移除顺序), present(收尾时仍保留的节点标记)
static void runQF(int n,
                  const std::vector<std::vector<int> >& adj,
                  const std::vector<double>& weight,
                  const std::vector<int>& appearOrder,
                  double gcc_threshold,
                  int priorityMode,
                  const std::vector<int>& degree,
                  std::vector<int>& individual,
                  std::vector<char>& present)
{
    present.assign(n, 1);
    individual.clear();
    std::vector<char> inIndividual(n, 0);

    while (true)
    {
        std::vector<int> core = computeCoreNumber(n, adj, present);

        std::vector<char> isCore1(n, 0);
        for (int v = 0; v < n; ++v)
            if (present[v] && core[v] == 1) isCore1[v] = 1;

        // 并查集(merge2 语义): 把 1-core 节点合并到非 1-core 根上
        std::vector<int> uni(n);
        std::vector<int> rankCount(n, 0);
        for (int v = 0; v < n; ++v)
        {
            uni[v] = v;
            if (present[v]) rankCount[v] = 1;
        }
        for (int v = 0; v < n; ++v)
        {
            if (!present[v] || !isCore1[v]) continue;
            for (size_t k = 0; k < adj[v].size(); ++k)
            {
                int u = adj[v][k];
                if (!present[u]) continue;
                int r1 = findRoot(uni, v); // node1 必须是 1-core 节点
                int r2 = findRoot(uni, u);
                if (r1 == r2) continue;
                if (!isCore1[r1])
                {
                    uni[r2] = r1;
                    rankCount[r1] += rankCount[r2];
                    rankCount[r2] = 0;
                }
                else
                {
                    uni[r1] = r2;
                    rankCount[r2] += rankCount[r1];
                    rankCount[r1] = 0;
                }
            }
        }

        // 候选: present 且 非 1-core 且 rankCount 不为 {0,1}; 优先级 = rankCount / cost
        // 每个候选保存 (优先级, 首次出现序号, 节点编号)
        std::vector<std::pair<double, std::pair<int, int> > > cand;
        for (int v = 0; v < n; ++v)
        {
            if (!present[v] || isCore1[v]) continue;
            int rc = rankCount[v];
            if (rc == 0 || rc == 1) continue;
            double pri;
            if (priorityMode == 1)
            {
                pri = (double)degree[v]; // boundary_degree_cut: 优先级 = 节点原始度数
            }
            else
            {
                double w = weight[v];
                pri = (w != 0.0) ? ((double)rc / w) : (double)rc;
            }
            cand.push_back(std::make_pair(pri, std::make_pair(appearOrder[v], v)));
        }
        // 优先级降序; 平局按节点首次出现顺序(与 networkx 稳定排序行为一致)
        std::sort(cand.begin(), cand.end(),
                  [](const std::pair<double, std::pair<int, int> >& a,
                     const std::pair<double, std::pair<int, int> >& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second.first < b.second.first;
                  });

        int n_before = 0;
        for (int v = 0; v < n; ++v) if (present[v]) ++n_before;

        for (size_t i = 0; i < cand.size(); ++i)
        {
            int v = cand[i].second.second;
            if (!inIndividual[v])
            {
                individual.push_back(v);
                inIndividual[v] = 1;
            }
        }
        for (size_t i = 0; i < individual.size(); ++i)
            present[individual[i]] = 0;

        int n_left = 0;
        for (int v = 0; v < n; ++v) if (present[v]) ++n_left;

        if (n_left == 0) break;
        int gcc = largestCC(n, adj, present);
        if ((double)gcc / (double)n_left < gcc_threshold) break; // QF DONE
        if (n_before == n_left) break;                            // 本轮无节点被移除
    }
}

// boundarycut: QF 边界剥离 + HDA 收尾, 返回完整移除顺序。
std::vector<int> BoundaryCut::getSolution(std::shared_ptr<Graph> graph, double gcc_threshold)
{
    int n = graph->num_nodes;
    const std::vector<double>& weight = graph->nodes_weight;
    std::vector<std::vector<int> > adj;
    std::vector<int> appearOrder;
    buildGraphData(graph, adj, appearOrder);

    std::vector<int> individual;
    std::vector<char> present;
    std::vector<int> noDegree;
    runQF(n, adj, weight, appearOrder, gcc_threshold, 0, noDegree, individual, present);

    // ---------------- HDA 收尾 ----------------
    std::vector<int> remove_list;
    int gccFinal = largestCC(n, adj, present);
    if ((double)gccFinal / (double)n > gcc_threshold) // 注意: 除以原始节点数
        remove_list = hdaBucket(n, adj, present);

    return assembleSolution(n, individual, remove_list);
}

// boundary_degree_cut: QF 边界剥离(优先级 = 节点原始度数) + HDA 收尾。
std::vector<int> BoundaryCut::getSolutionDegree(std::shared_ptr<Graph> graph, double gcc_threshold)
{
    int n = graph->num_nodes;
    const std::vector<double>& weight = graph->nodes_weight;
    std::vector<std::vector<int> > adj;
    std::vector<int> appearOrder;
    buildGraphData(graph, adj, appearOrder);

    // 节点原始度数(简单图, 去自环去重边后)
    std::vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v)
        degree[v] = (int)adj[v].size();

    std::vector<int> individual;
    std::vector<char> present;
    runQF(n, adj, weight, appearOrder, gcc_threshold, 1, degree, individual, present);

    // ---------------- HDA 收尾 ----------------
    std::vector<int> remove_list;
    int gccFinal = largestCC(n, adj, present);
    if ((double)gccFinal / (double)n > gcc_threshold) // 注意: 除以原始节点数
        remove_list = hdaBucket(n, adj, present);

    return assembleSolution(n, individual, remove_list);
}

// boundary_degree_cut_HD: QF 边界剥离(优先级 = 节点原始度数) + HD(非自适应最高度) 收尾。
std::vector<int> BoundaryCut::getSolutionDegreeHD(std::shared_ptr<Graph> graph, double gcc_threshold)
{
    int n = graph->num_nodes;
    const std::vector<double>& weight = graph->nodes_weight;
    std::vector<std::vector<int> > adj;
    std::vector<int> appearOrder;
    buildGraphData(graph, adj, appearOrder);

    // 节点原始度数(简单图, 去自环去重边后)
    std::vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v)
        degree[v] = (int)adj[v].size();

    std::vector<int> individual;
    std::vector<char> present;
    runQF(n, adj, weight, appearOrder, gcc_threshold, 1, degree, individual, present);

    // ---------------- HD 收尾 ----------------
    std::vector<int> remove_list;
    int gccFinal = largestCC(n, adj, present);
    if ((double)gccFinal / (double)n > gcc_threshold) // 注意: 除以原始节点数
        remove_list = hdStatic(n, adj, present);

    return assembleSolution(n, individual, remove_list);
}

// boundarycutHD: QF 边界剥离 + HD(非自适应最高度) 收尾, 返回完整移除顺序。
std::vector<int> BoundaryCut::getSolutionHD(std::shared_ptr<Graph> graph, double gcc_threshold)
{
    int n = graph->num_nodes;
    const std::vector<double>& weight = graph->nodes_weight;
    std::vector<std::vector<int> > adj;
    std::vector<int> appearOrder;
    buildGraphData(graph, adj, appearOrder);

    std::vector<int> individual;
    std::vector<char> present;
    std::vector<int> noDegree;
    runQF(n, adj, weight, appearOrder, gcc_threshold, 0, noDegree, individual, present);

    // ---------------- HD 收尾 ----------------
    std::vector<int> remove_list;
    int gccFinal = largestCC(n, adj, present);
    if ((double)gccFinal / (double)n > gcc_threshold) // 注意: 除以原始节点数
        remove_list = hdStatic(n, adj, present);

    return assembleSolution(n, individual, remove_list);
}

// RENDCut 用: 仅执行 QF 边界剥离阶段。
// 返回 [individual, remaining, {need_finder}]:
//   individual    - QF 阶段的移除顺序
//   remaining     - QF 后仍保留的节点(升序)
//   {need_finder} - 1 表示剩余子图最大连通片/原始节点数 > 阈值(需要进一步瓦解), 否则 0
std::vector<std::vector<int> > BoundaryCut::getQFSolution(std::shared_ptr<Graph> graph, double gcc_threshold)
{
    int n = graph->num_nodes;
    const std::vector<double>& weight = graph->nodes_weight;
    std::vector<std::vector<int> > adj;
    std::vector<int> appearOrder;
    buildGraphData(graph, adj, appearOrder);

    std::vector<int> individual;
    std::vector<char> present;
    std::vector<int> noDegree;
    runQF(n, adj, weight, appearOrder, gcc_threshold, 0, noDegree, individual, present);

    std::vector<int> remaining;
    for (int v = 0; v < n; ++v)
        if (present[v]) remaining.push_back(v);

    int gccFinal = largestCC(n, adj, present);
    int need_finder = ((double)gccFinal / (double)n > gcc_threshold) ? 1 : 0;

    std::vector<std::vector<int> > out;
    out.push_back(individual);
    out.push_back(remaining);
    out.push_back(std::vector<int>(1, need_finder));
    return out;
}
