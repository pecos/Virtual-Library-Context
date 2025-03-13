#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <iostream>
#include <thread>
#include <sched.h>
#include <vector>

#include "VLC/runtime.h"
#include "VLC/tuning.h"

// sharing memory make it easy to sync like this way
pthread_barrier_t barrier;

// shared data between stages
std::vector<void *> graphs;  // set by stage 1 once a graph is loaded
std::vector<int> top_nodes;  // set by stage 1 once pagerank is finished
std::vector<bool> pagerank_done; // set by stage 1 once pagerank is finished

// function types
typedef void* (*load_graph_t)(const std::string filename);

typedef void (*init_pagerank_t)(int num_cores);
typedef int (*pagerank_t)(void* pGraph);

typedef void (*init_bfs_t)(int num_cores);
typedef void (*bfs_t)(void* pGraph, int iSource, int slot);
typedef unsigned int (*read_distance_t)(void* pGraph, int node, int slot);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"load_graph", "_Z10load_graphNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
      {"init_bfs", "_Z8init_bfsi"},
      {"bfs", "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
      {"read_distance", "_Z13read_distancePN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
      {"init_pagerank", "_Z13init_pageranki"},
      {"pagerank", "_Z8pagerankPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEE"}};
   VLC::Loader::register_func_names(names);
}

void launch_pagerank(std::string filename, int round, int vlc_id, const char* cpu_str, int core_count) {
    std::cout << "VLC " << vlc_id << "(PageRank) is created." << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu(cpu_str);
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libpagerank.so", vlc_id, false);

    auto init_pagerank = loader.load_func<init_pagerank_t>("init_pagerank");
    auto pagerank = loader.load_func<pagerank_t>("pagerank");
    auto load_graph = loader.load_func<load_graph_t>("load_graph");

    init_pagerank(core_count);

    // stag 1: Find the top nodes
    for (int rd = 0; rd < round + 1; rd++) {
        pthread_barrier_wait(&barrier);
        if (rd == round) {
            continue; // do nothing at the last round
        }

        auto start_time = std::chrono::system_clock::now();
        std::cout << "pagerank start round " << rd << std::endl;
        void *graph = load_graph(filename);

        // int top = pagerank(graph);

        graphs[rd] = graph;
        top_nodes[rd] = 0;
        pagerank_done[rd] = true;
        
        auto end_time = std::chrono::system_clock::now();
        std::cout << "pagerank round " << rd << " end time: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
    }
}

void launch_bfs(int round, int vlc_id, const char* cpu_str, int core_count) {
    std::cout << "VLC " << vlc_id << "(BFS) is created." << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu(cpu_str);
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libbfs.so", vlc_id, false);

    auto init_bfs = loader.load_func<init_bfs_t>("init_bfs");
    auto bfs = loader.load_func<bfs_t>("bfs");
    auto read_distance = loader.load_func<read_distance_t>("read_distance");

    init_bfs(core_count);

    // stag 2: Find the SSSP from 0 to top node
    for (int rd = 0; rd < round+1; rd++) {
        pthread_barrier_wait(&barrier);
        if (rd == 0) {
            continue; // do nothing at the first round
        }

        auto start_time = std::chrono::system_clock::now();
        std::cout << "bfs start round " << rd << std::endl;
        int slot = 0;
        int source = 3;
        int dst = 1131506;
        bfs(graphs[rd-1], source, slot);
        unsigned int d = read_distance(graphs[rd-1], dst, slot);
        std::cout << "SSSP: distance from " << source << " to " << dst << " is " << d << std::endl;
        // delete graphs[rd-1];

        auto end_time = std::chrono::system_clock::now();
        std::cout << "bfs round " << rd << " end time: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
    }
}

int main(int argc, char * argv[]) {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    register_functions();

    if (argc != 5) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <gr file path> <rounds> <cpu1> <cpu2>");
        exit(0);
    }

    std::string filename(argv[1]);
    int rounds = std::stoi(argv[2]);

    pthread_barrier_init(&barrier, NULL, 2);

    graphs.resize(rounds, NULL);
    top_nodes.resize(rounds, 0);
    pagerank_done.resize(rounds, false);

    auto pagerank = std::thread(launch_pagerank, filename, rounds, 1, argv[3], 4);
    auto bfs = std::thread(launch_bfs, rounds, 2, argv[4], 24);
    
    pagerank.join();
    bfs.join();

    return 0;
}