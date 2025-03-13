#include <iostream>
#include "bfs.h"
#include "LoadGraph.h"
#include "PageRank.h"

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <gr file path> <rounds>");
        exit(0);
    }

    std::string filename(argv[1]);
    int rounds = std::stoi(argv[2]);

    std::cout << "Initialize Galois Runtime" << std::endl;
    init_load_graph(24);

    for (int rd = 0; rd < rounds; rd++) {
        std::cout << "LoadGraph starts" << std::endl;
        Graph * graph_ptr = load_graph(filename);
        
        std::cout << "PageRank starts" << std::endl;
        int top = pagerank(graph_ptr);

        int source = 0;

        std::cout << "SSSP starts" << std::endl;
        bfs(graph_ptr, source, 0);

        unsigned int d = read_distance(graph_ptr, 9915159, 0);
        std::cout << "round " << rd + 1 << ": distance from " << source << " to " << 9915159 << " is " << d << std::endl;
        free(graph_ptr);
    }

    return 0;
}