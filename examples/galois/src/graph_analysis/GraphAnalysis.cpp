#include <iostream>
#include <chrono>
#include "PageRank.h"
#include "bfs.h"


int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <gr file path> <rounds>");
        exit(0);
    }

    std::string filename(argv[1]);
    int rounds = std::stoi(argv[2]);

    std::cout << "Initialize Galois Runtime" << std::endl;
    init_pagerank(24);

    for (int rd = 0; rd < rounds; rd++) {  
        auto start_time = std::chrono::system_clock::now();
        std::cout << "LoadGraph starts" << std::endl;
        Graph * graph_ptr = load_graph(filename);
        auto mid_time = std::chrono::system_clock::now();
        std::cout << "round " << rd << " load time: " << ((std::chrono::duration<double>) (mid_time - start_time)).count() << "s\n";

        std::cout << "PageRank starts" << std::endl;
        // int top = pagerank(graph_ptr);

        int source = 0;

        std::cout << "SSSP starts" << std::endl;
        bfs(graph_ptr, 3, 0);
        unsigned int d = read_distance(graph_ptr, 1131506, 0);
        std::cout << "round " << rd << ": distance from " << 3 << " to " << 1131506 << " is " << d << std::endl;

        // free(graph_ptr);
        auto end_time = std::chrono::system_clock::now();
        std::cout << "round " << rd << " end time: " << ((std::chrono::duration<double>) (end_time - mid_time)).count() << "s\n";
    }

    return 0;
}