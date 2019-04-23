#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "../app/shard/skelecton.cuh"
#include "../query.cuh"
#include "../utils.hpp"

int main(int argc, char *argv[]){
    if(argc < 2){
        std::cout << "Usage: " << argv[1] << " [file].\n";
        exit(EXIT_FAILURE);
    }

    device_list props;
    query_device(&props);

    FILE *fp = open_file_access(argv[1], "r");
    graph_shard graph;

    graph_construct(fp, graph, &(props.devices[0]), false, true);

    return 0;
}