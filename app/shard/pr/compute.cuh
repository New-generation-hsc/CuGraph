#ifndef __GRAPH_COMPUTE__
#define __GRAPH_COMPUTE__

#include "../skelecton.cuh"
#include "../../../config.hpp"

void pagerank_graph_init(graph_shard<float> &, config_t *);
void execute(graph_shard<float> &, config_t *);

#endif