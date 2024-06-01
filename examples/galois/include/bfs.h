#include "BFS_SSSP.h"

const int NSLOTS = 8;

struct NodeData {
    unsigned distances[NSLOTS];
};

//using Graph = galois::graphs::LC_CSR_Graph<unsigned, void>::with_no_lockable<true>::type;

using Graph = galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;

using GNode = Graph::GraphNode;

void init_galois(int nThreads);

void delete_galois();

Graph* load_file(const std::string& filename);

void bfs(Graph* pGraph, int iSource, int slot);

unsigned int read_distance(Graph* pGraph, int node, int slot);