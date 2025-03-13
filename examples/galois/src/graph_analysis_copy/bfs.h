#ifndef GRAPH_ANALYSIS_BFS_H
#define GRAPH_ANALYSIS_BFS_H

#include "graph.h"

void init_bfs(int num_cores);
void bfs(Graph* pGraph, int iSource, int slot);
unsigned int read_distance(Graph* pGraph, int node, int slot);

#endif