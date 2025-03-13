/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#ifndef LONESTAR_PAGERANK_CONSTANTS_H
#define LONESTAR_PAGERANK_CONSTANTS_H

#include "graph.h"
#include <map>

#define DEBUG 0

template <typename GNode>
struct TopPair {
  float value;
  GNode id;

  TopPair(float v, GNode i) : value(v), id(i) {}

  bool operator<(const TopPair& b) const {
    if (value == b.value)
      return id > b.id;
    return value < b.value;
  }
};

//! Helper functions.

PRTy atomicAdd(std::atomic<PRTy>& v, PRTy delta) {
  PRTy old;
  do {
    old = v;
  } while (!v.compare_exchange_strong(old, old + delta));
  return old;
}

template <typename Graph>
void printTop(Graph& graph, unsigned topn = PRINT_TOP) {

  using GNode = typename Graph::GraphNode;
  typedef TopPair<GNode> Pair;
  typedef std::map<Pair, GNode> TopMap;

  TopMap top;

  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ++ii) {
    GNode src  = *ii;
    auto& n    = graph.getData(src);
    PRTy value = n.value;
    Pair key(value, src);

    if (top.size() < topn) {
      top.insert(std::make_pair(key, src));
      continue;
    }

    if (top.begin()->first < key) {
      top.erase(top.begin());
      top.insert(std::make_pair(key, src));
    }
  }

  int rank = 1;
  std::cout << "Rank PageRank Id\n";
  for (auto ii = top.rbegin(), ei = top.rend(); ii != ei; ++ii, ++rank) {
    std::cout << rank << ": " << ii->first.value << " " << ii->first.id << "\n";
  }
}

#if DEBUG
template <typename Graph>
void printPageRank(Graph& graph) {
  std::cout << "Id\tPageRank\n";
  int counter = 0;
  for (auto ii = graph.begin(), ei = graph.end(); ii != ei; ii++) {
    auto& sd = graph.getData(*ii);
    std::cout << counter << " " << sd.value << "\n";
    counter++;
  }
}
#endif

#endif