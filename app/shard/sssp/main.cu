#include "compute.cuh"
#include "../../../utils.hpp"
#include <iostream>
#include <cstdio>
#include <unistd.h>

#include "../skelecton.cuh"
#include "../../../query.cuh"

void usage(){
    std:: cout << "Required command line arguments:" << std::endl;
    std:: cout << "\t[-i] input file. E.g -i tmp.txt\n";
    std:: cout << "Optional arguments:\n";
    std:: cout << "\t[-o] output file. defailt out.txt\n";
    std:: cout << "\t[-d] device number, custom select which device to run\n";
    std:: cout << "\t[-a] an arbitrary number, select the sssp source vertex.\n";
    std:: cout << "\t[-v] verbose, whether print some useful information\n";
    std:: cout << "\t[-b] wheter the graph is direct, default direct\n";
}

void parse_args(config_t &conf, int argc, char** argv){
    int c;
    bool valid = false;

    while((c = getopt(argc, argv, "i:o:d:a:vb")) != -1){
        switch(c){
            case 'i':
                valid = true;
                strcpy(conf.input_path, optarg);
                break;
            case 'o':
                strcpy(conf.output_path, optarg);
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
                conf.undirected = true;
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

int main(int argc, char *argv[]){
    
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
    graph_shard<uint> graph;
    graph_construct(fp, graph, &(props.devices[conf.device]), &conf);

    execute(graph, &conf);

    return 0;
}