#include "../defs.hpp"
#include "../utils.hpp"

int main(int argc, char *argv[]){
    if(argc < 2) {
        printf("Usage: %s [file]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    FILE *fp = open_file_access(argv[1], "r");

    csr_graph graph;
    
    parse_graph(fp, graph, false, false, true);

    for(int i = 0; i < graph.n; i++){
        printf("%d -> %d\n", graph.labels[i], graph.row_offsets[i + 1]);
    }

    graph.destroy();
    return 0;
}