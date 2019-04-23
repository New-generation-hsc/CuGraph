#include "compute.cuh"
#include "../../../utils.hpp"
#include <iostream>
#include <cstdio>
#include <unistd.h>

#include "../skelecton.cuh"
#include "../../../query.cuh"

void usage(){
    std::cout << "Required Command Line Arguments:" << std::endl;
    std::cout << "\t[-i] input file. E.g -i input.txt" << std::endl;
    std::cout << "Optional Arguments:" << std::endl;
    std::cout << "\t[-o] output file. E.g default out.txt" << std::endl;
    std::cout << "\t[-r] initial pr value for pr problem" << std::endl;
    std::cout << "\t[-t] maximum number of iteration for pr" << std::endl;
    std::cout << "\t[-f] damping factor, default 0.85" << std::endl;
    std::cout << "\t[-e] tolerant error" << std::endl;
    std::cout << "\t[-d] choose a specific device number, default 0" << std::endl;
    std::cout << "\t[-b] whether the graph is directed" << std::endl;
    std::cout << "\t[-v] print some useful information" << std::endl;
}

void parse_args(config_t *config, int argc, char **argv){
    int c;
    bool valid = false; // indicate the input arguments is valid

    while((c = getopt(argc, argv, "i:o:r:t:f:e:d:bv")) != -1){
        switch(c){
            case 'i':
                valid = true;
                strcpy(config->input_path, optarg);
                break;
            case 'o':
                strcpy(config->output_path, optarg);
                break;
            case 'r':
                config->init_prval    = strtod(optarg, NULL);
                break;
            case 't':
                config->maximum_iterations = atoi(optarg);
                break;
            case 'f':
                config->factor        = strtod(optarg, NULL);
                break;
            case 'e':
                config->threshold     = strtod(optarg, NULL);
                break;
            case 'd':
                config->device         = atoi(optarg);
                break;
            case 'b':
                config->undirected     = false;
                break;
            case 'v':
                config->verbose        = false;
                break;
            case '?':
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
    parse_args(&conf, argc, argv);

    /* query device info */
    device_list props;
    query_device(&props);

    if(conf.verbose){
        std::cout << "query device count: " << props.count << std::endl;
        display_device_info(props, conf.device);
    }

    FILE *fp = open_file_access(conf.input_path, "r");

    /* construct graph shard representation */
    graph_shard<float> graph;
    graph_construct(fp, graph, &(props.devices[conf.device]), &conf);

    execute(graph, &conf);

    return 0;
}