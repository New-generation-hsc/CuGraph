#ifndef __GRAPH_COMPUTE__
#define __GRAPH_COMPUTE__

#include "../skelecton.cuh"
#include "../../../config.hpp"

void bfs_graph_init(graph_shard<uint> &, config_t *);
void execute(graph_shard<uint> &, config_t *);

#endif