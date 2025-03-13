#ifndef GRAPH_ANALYSIS_PAGERANK_H
#define GRAPH_ANALYSIS_PAGERANK_H

#include "graph.h"

void init_pagerank(int num_cores);
int pagerank(Graph* pGraph);
Graph* load_graph(const std::string filename);

#endif