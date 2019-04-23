#include "skelecton.cuh"
#include "../../defs.hpp"
#include "../../query.cuh" 

#include <iostream>
#include <cstring>

void graph_construct( FILE           *fp,
                      graph_shard    &graph,
                      cudaDeviceProp *prop,
                      config_t       *conf
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
    graph_shard_init(graph, conf);

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