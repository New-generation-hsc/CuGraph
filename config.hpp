#ifndef __GRAPH_CONFIG__
#define __GRAPH_CONFIG__

#define MAX_LEN 256

#define DEFAULT_BLOCK_SIZE 256

struct config_t
{
    // device number
    int device;

    // source node index , for sssp and bfs
    int source;

    // input file path
    char input_path[MAX_LEN];

    // output file path
    char output_path[MAX_LEN];

    // No. number of threads
    int blocksize;

    // whether graph directed
    bool undirected;

    // following value for pagerank
    double init_prval;

    // maximum iterations
    int maximum_iterations;

    // the damping factor, default 0.85
    double factor;

    // the tolerant error threshold
    double threshold;

    // whther print info
    bool verbose;
};

void global_config_init(config_t *);

#endif