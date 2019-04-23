#ifndef __GARPH_SKELECTON__
#define __GARPH_SKELECTON__

#include <cstdio>
#include "../../buffer.cuh"
#include "../../config.hpp"

struct graph_shard {
    // No. number of nodes
    uint n;

    // No. number of edges
    uint m;

    // No. number of shards
    uint num_shards;

    // No. shard maximum number of nodes
    uint shard_max_num_nodes;

    // No. number of threads in shard
    uint blocksize;

    // src index array
    buffer<uint> src_indexs;

    // dest index array
    buffer<uint> dest_indexs;

    // src values array
    buffer<uint> src_values;

    // edge weights
    buffer<uint> edge_values;

    // shard sizes array
    buffer<uint> shards_size;

    // windows size array
    buffer<uint> windows_size;

    // vertex values array
    buffer<uint> values;

    // vertex neigbors count array
    buffer<uint> neigbors_size;

    // node labels
    buffer<uint> labels;

    void set_up(uint nodes, uint edges, uint shards, uint shard_nodes, uint threads){
        n = nodes;
        m = edges;

        num_shards = shards;
        shard_max_num_nodes = shard_nodes;
        blocksize = threads;

        src_indexs.alloc( edges );
        dest_indexs.alloc( edges );

        src_values.alloc( edges );
        edge_values.alloc( edges );

        shards_size.alloc( shards + 1 );
        shards_size[0] = 0;
        windows_size.alloc( shards * shards + 1 );
        windows_size[0] = 0;
        
        values.alloc( num_shards * shard_max_num_nodes );
        neigbors_size.alloc( nodes );

        labels.alloc( nodes );
    }
};

void graph_construct( FILE *fp,
                      graph_shard &graph,
                      cudaDeviceProp *prop,
                      config_t       *conf
                    );

/** a general graph init function pointer */
typedef void (*graph_init)(graph_shard &, config_t*);
extern graph_init graph_shard_init;

#endif