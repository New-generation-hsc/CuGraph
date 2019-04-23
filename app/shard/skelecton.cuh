#ifndef __GARPH_SKELECTON__
#define __GARPH_SKELECTON__

#include <cstdio>
#include "../../buffer.cuh"
#include "../../config.hpp"
#include "../../defs.hpp"
#include "../../query.cuh" 

#include <iostream>
#include <cstring>

template<typename Value>
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
    buffer<Value> src_values;

    // edge weights
    buffer<uint> edge_values;

    // shard sizes array
    buffer<uint> shards_size;

    // windows size array
    buffer<uint> windows_size;

    // vertex values array
    buffer<Value> values;

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

/*template<typename Value>
void graph_construct( FILE *fp,
                      graph_shard<Value> &graph,
                      cudaDeviceProp *prop,
                      config_t       *conf
                    );
*/

template<typename Value>
struct graph_initializer {
    static void (*graph_init)(graph_shard<Value> &, config_t *);
};

template<typename Value>
void graph_construct( FILE               *fp,
                      graph_shard<Value> &graph,
                      cudaDeviceProp     *prop,
                      config_t           *conf
                    )
{
    // construct adjencent list first
    graph_t tmp_graph;
    parse_graph(fp, tmp_graph, conf->undirected, true, conf->verbose);

    blockinfo info = find_proper_blocksize(prop, sizeof(uint), tmp_graph.n, tmp_graph.m);

    //std::cout << "block blocksize : "  << info.blocksize << ", nodes " << info.shard_max_num_nodes << ".\n";

    uint shards = std::ceil((double)tmp_graph.n / info.shard_max_num_nodes);
    uint shards_vertices = shards * info.shard_max_num_nodes;

    if(conf->verbose){
        std::cout << "graph partition into " << shards << " shards, blocksize : " << info.blocksize << ", shard vertices : " << info.shard_max_num_nodes << ".\n";
    }

    graph.set_up(tmp_graph.n, tmp_graph.m, shards, info.shard_max_num_nodes, info.blocksize);

    std::vector<std::vector<edge_t>> graph_windows( shards * shards );
    for(uint nodeIdx = 0; nodeIdx < tmp_graph.n; ++nodeIdx){ // node index
        for(uint nbrsIdx = 0; nbrsIdx < tmp_graph.adj[nodeIdx].size(); ++nbrsIdx){ // neigbors index
            edge_t edge = tmp_graph.adj[nodeIdx][nbrsIdx];
            uint shard_index  = edge.dest * shards / shards_vertices;
            uint window_index = edge.src * shards / shards_vertices;

            graph_windows.at( shard_index * shards + window_index ).push_back( edge );
        }
        graph.labels[nodeIdx] = tmp_graph.labels[nodeIdx];
    }

    // initialize the graph values according to different application
    graph_initializer<Value>::graph_init(graph, conf);

    memset(graph.neigbors_size.ptr, 0, graph.neigbors_size.size());

    uint offset = 0;
    for(uint shardIdx = 0; shardIdx < shards; ++shardIdx){
        for(uint winIdx = 0; winIdx < shards; ++winIdx){
            uint index = shardIdx * shards + winIdx;
            for(uint entryIdx = 0; entryIdx < graph_windows[index].size(); ++entryIdx){
                edge_t edge = graph_windows[index][entryIdx];

                graph.src_indexs[ offset ]  = edge.src;
                graph.dest_indexs[ offset ] = edge.dest;

                graph.src_values[ offset ]  = graph.values[ edge.src ];
                graph.edge_values[ offset ] = edge.weight;

                graph.neigbors_size[ edge.src ]++;

                ++offset;
            }
            graph.windows_size[ index + 1 ] = offset;
        }

        graph.shards_size[ shardIdx + 1 ] = offset;
    }
}


#endif