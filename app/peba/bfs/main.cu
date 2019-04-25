#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "../skelecton.cuh"
#include "../../../query.cuh"
#include "../../../utils.hpp"

void usage(){
    std:: cout << "Required command line arguments:" << std::endl;
    std:: cout << "\t[-i] input file. E.g -i tmp.txt\n";
    std:: cout << "Optional arguments:\n";
    std:: cout << "\t[-o] output file. defailt out.txt\n";
    std:: cout << "\t[-s] blocksize, number of threads in a block\n";
    std:: cout << "\t[-d] device number, custom select which device to run\n";
    std:: cout << "\t[-a] an arbitrary number, select the sssp source vertex.\n";
    std:: cout << "\t[-v] verbose, whether print some useful information\n";
    std:: cout << "\t[-b] wheter the graph is direct, default direct\n";
}

void parse_args(config_t &conf, int argc, char** argv){
    int c;
    bool valid = false;

    while((c = getopt(argc, argv, "i:o:d:s:a:vb")) != -1){
        switch(c){
            case 'i':
                valid = true;
                strcpy(conf.input_path, optarg);
                break;
            case 'o':
                strcpy(conf.output_path, optarg);
                break;
            case 's':
                conf.blocksize = atoi(optarg);
                break;
            case 'd':
                conf.device = atoi(optarg);
                break;
            case 'a':
                conf.source = atoi(optarg);
                break;
            case 'v':
                conf.verbose = false;
                break;
            case 'b':
                conf.undirected = false;
                break;
            case '?':
                //std::cerr << "unknow argument option : " << optopt << std::endl;
                usage();
                exit(EXIT_FAILURE);
        }
    }

    if(!valid){
        usage();
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char* argv[]){
    
    // set application configuration
    config_t conf;
    global_config_init(&conf);
    parse_args(conf, argc, argv);

    /* query device info */
    device_list props;
    query_device(&props);

    if(conf.verbose){
        std::cout << "query device count: " << props.count << std::endl;
        display_device_info(props, conf.device);
    }

    FILE *fp = open_file_access(conf.input_path, "r");

    /* construct graph shard representation */
    csr_graph csr;
    parse_graph(fp, csr, conf.undirected, false, conf.verbose);

    graph_structure<uint> graph;

    graph.graph_construct(csr);

    graph_problem<uint> bfs;

    bfs.execute(graph, &conf);

    fclose(fp);

    return 0;
}