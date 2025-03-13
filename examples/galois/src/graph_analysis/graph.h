#ifndef GRAPH_ANALYSIS_GRAPH_H
#define GRAPH_ANALYSIS_GRAPH_H

#include <iostream>
#include <atomic>
#include "galois/Galois.h"
#include "galois/graphs/LCGraph.h"

const int NSLOTS = 8;

const int NUM_CORES = 100; // use all cores by default

//! All PageRank algorithm variants use the same constants for ease of
//! comparison.
constexpr static const float ALPHA         = 0.85;
constexpr static const float INIT_RESIDUAL = 1 - ALPHA;

constexpr static const float TOLERANCE   = 1.0e-1;
constexpr static const unsigned MAX_ITER = 5;

constexpr static const unsigned PRINT_TOP = 3;

//! Type definitions.
typedef float PRTy;

struct NodeData {
  unsigned distances[NSLOTS];
  PRTy value;
  std::atomic<PRTy> residual;

  void init() {
    value    = 0.0;
    residual = INIT_RESIDUAL;
  }

  friend std::ostream& operator<<(std::ostream& os, const NodeData& n) {
    os << "{PR " << n.value << ", residual " << n.residual << "}";
    return os;
  }

};

using Graph = galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;

using GNode = Graph::GraphNode;

#endif