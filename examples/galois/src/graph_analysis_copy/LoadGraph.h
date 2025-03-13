#ifndef GRAPH_ANALYSIS_LOADGRAPH_H
#define GRAPH_ANALYSIS_LOADGRAPH_H

#include "graph.h"

void init_load_graph(int num_cores);
Graph* load_graph(const std::string filename);

#endif