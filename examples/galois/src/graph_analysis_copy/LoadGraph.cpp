#include <iostream>
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

#include "LoadGraph.h"

galois::SharedMemSys* load_graph_G = NULL;

void init_load_graph(int num_cores) {
  load_graph_G = new galois::SharedMemSys();
  galois::setActiveThreads(num_cores);
  std::cout << "initialize load_graph with " << galois::getActiveThreads() << " cores avaliable" << std::endl;
}

Graph* load_graph(const std::string filename) {
  Graph* graph = new Graph();
  std::cout << "Reading from file: " << filename << std::endl;
  galois::graphs::readGraph(*graph, filename);
  std::cout << "LoadGraph starts" << std::endl;
  std::cout << "Read " << graph->size() << " nodes, " << graph->sizeEdges()
            << " edges" << std::endl;
  return graph;
}