#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <iostream>
#include <thread>
#include <sched.h>

#include "VLC/runtime.h"
#include "VLC/tuning.h"

// global state of each stage
// sharing memory make it easy to sync like this way
bool load_graph_ready = false;
bool pagerank_ready = true;
bool bfs_ready = true;
int load_graph_round = 0;
int pagerank_round = 0;
int bfs_round = 0;

// shared data between stages
void * pagerank_input = NULL;  // set by stage 1 once a graph is loaded
void * bfs_input = NULL;  // set by stage 1 once a graph is loaded
int top_node = 0;  // set by stage 2
unsigned int distance = 0; // set by stage 3, this is the final result of a round

// function types
typedef void (*init_load_graph_t)(int num_cores);
typedef void* (*load_graph_t)(const std::string filename);

typedef void (*init_pagerank_t)(int num_cores);
typedef int (*pagerank_t)(void* pGraph);

typedef void (*init_bfs_t)(int num_cores);
typedef void (*bfs_t)(void* pGraph, int iSource, int slot);
typedef unsigned int (*read_distance_t)(void* pGraph, int node, int slot);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"init_load_graph", "_Z15init_load_graphi"},
      {"load_graph", "_Z10load_graphNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
      {"init_bfs", "_Z8init_bfsi"},
      {"bfs", "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
      {"read_distance", "_Z13read_distancePN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
      {"init_pagerank", "_Z13init_pageranki"},
      {"pagerank", "_Z8pagerankPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEE"}};
   VLC::Loader::register_func_names(names);
}

void launch_load_graph(std::string filename, int round, int vlc_id, const char* cpu_str, int core_count) {
    std::cout << "VLC " << vlc_id << "(LoadGraph) is created." << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu(cpu_str);
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libloadgraph.so", vlc_id, false);

    auto init_load_graph = loader.load_func<init_load_graph_t>("init_load_graph");
    auto load_graph = loader.load_func<load_graph_t>("load_graph");

    init_load_graph(core_count);

    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        std::cerr << "APP: unable to determine cpu set, " << strerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // stag 1: parsing graph file
    while (load_graph_round < round) {
        if (!pagerank_ready || (load_graph_round == pagerank_round)) {
            load_graph_ready = false;
            std::cout << "load graph start round " << load_graph_round << std::endl;
            void *graph_ptr = load_graph(filename);
            std::cout << "load graph finish round " << load_graph_round << std::endl;

            while(true) {
                if (pagerank_ready && (load_graph_round == pagerank_round)) {  // prepare graph for pagerank
                    pagerank_input = graph_ptr;

                    load_graph_round++;
                    load_graph_ready = true;
                    break;
                }
            }
        }
    }
}

void launch_pagerank(int round, int vlc_id, const char* cpu_str, int core_count) {
    std::cout << "VLC " << vlc_id << "(PageRank) is created." << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu(cpu_str);
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libpagerank.so", vlc_id, false);

    auto init_pagerank = loader.load_func<init_pagerank_t>("init_pagerank");
    auto pagerank = loader.load_func<pagerank_t>("pagerank");

    init_pagerank(core_count);

    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        std::cerr << "APP: unable to determine cpu set, " << strerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }


    // stag 2: Find the top nodes
    while (pagerank_round < round) {
        if (load_graph_ready && (load_graph_round == pagerank_round + 1)) {  // waiting next graph loaded and last graph passed to bfs
            void *graph = pagerank_input;

            pagerank_ready = false;
            std::cout << "pagerank start round " << pagerank_round << std::endl;
            int top = pagerank(graph);
            std::cout << "pagerank finish round " << pagerank_round << std::endl;

            while (true) {
                if (bfs_ready) { // waiting for bfs finished last graph
                    bfs_input = pagerank_input;
                    top_node = top;

                    pagerank_round++;
                    pagerank_ready = true;
                    break;
                }
            }
        }
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

    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        std::cerr << "APP: unable to determine cpu set, " << strerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // stag 3: Find the SSSP from 0 to top node
    while (bfs_round < round) {
        if (pagerank_ready && (pagerank_round >= bfs_round + 1)) {
            int top = top_node;
            void *graph = bfs_input;
            bfs_ready = false;
            std::cout << "bfs start round " << bfs_round << std::endl;

            int slot = 0;
            int source = 0;
            bfs(graph, source, slot);
            unsigned int d = read_distance(graph, 9915159, slot);
            std::cout << "SSSP: distance from " << source << " to " << 9915159 << " is " << d << std::endl;
            free(bfs_input);

            bfs_round++;
            bfs_ready = true;
        }
    }
}

int main(int argc, char * argv[]) {
    if (argc != 6) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <gr file path> <rounds> <cpu1> <cpu2> <cpu3>");
        exit(0);
    }

    std::string filename(argv[1]);
    int rounds = std::stoi(argv[2]);

    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    register_functions();

    auto load_graph = std::thread(launch_load_graph, filename, rounds, 1, argv[3], 20);
    auto pagerank = std::thread(launch_pagerank, rounds, 2, argv[4], 4);
    // auto bfs = std::thread(launch_bfs, rounds, 3, argv[5], 24);
    
    load_graph.join();
    pagerank.join();
    // bfs.join();

    return 0;
}