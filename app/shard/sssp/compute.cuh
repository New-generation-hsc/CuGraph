#ifndef __GRAPH_COMPUTE__
#define __GRAPH_COMPUTE__

#include "../skelecton.cuh"
#include "../../../config.hpp"

void sssp_graph_init(graph_shard &, config_t *);
void execute(graph_shard &, config_t *);

#endif