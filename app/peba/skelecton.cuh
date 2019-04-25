#ifndef __GRAPH_SKELECTON__
#define __GRAPH_SKELECTON__

#include "../../buffer.cuh"
#include "../../defs.hpp"
#include "../../config.hpp"

#include <iostream>

template<typename Value>
struct graph_structure
{
    // No. number of nodes
    uint n;

    // No. number of edges
    uint m;

    // vertex values
    buffer<Value> values;

    // edge weights
    buffer<uint> weights;

    // edges src index
    buffer<uint>  src_indexs;

    // edges destination index
    buffer<uint>  dest_indexs;

    // outdegrees of each edge destination
    buffer<uint>  out_degrees;

    // the outdegrees of each node
    buffer<uint>  degrees;

    // graph labels
    buffer<uint>  labels;

    void set_up(uint nodes, uint edges){

        n = nodes;
        m = edges;

        values.alloc( nodes );

        weights.alloc( edges );
        src_indexs.alloc( edges );
        dest_indexs.alloc( edges );
        out_degrees.alloc( edges );

        degrees.alloc( nodes );
        labels.alloc( nodes );
    }

    void graph_construct(csr_graph &csr){

        set_up(csr.n, csr.m);
        
        degrees = csr.degrees;
        weights = csr.edge_weights;
        labels = csr.labels;


        // construct the basic information
        if(csr.dest_based){
            uint edge_index = 0;
            for(uint node_idx = 0; node_idx < csr.n; ++node_idx){
                for(uint nbrs = csr.row_offsets[node_idx]; nbrs < csr.row_offsets[node_idx + 1]; nbrs++){
                    out_degrees[edge_index] = csr.degrees[node_idx];
                    dest_indexs[edge_index] = node_idx;
                    src_indexs [edge_index] = csr.column_values[edge_index];

                    ++edge_index;
                }
            }
        }else{
            uint edge_index = 0;
            for(uint node_idx = 0; node_idx < csr.n; ++node_idx){
                for(uint nbrs = csr.row_offsets[node_idx]; nbrs < csr.row_offsets[node_idx + 1]; nbrs++){
                    out_degrees[edge_index] = csr.degrees[csr.column_values[edge_index]];
                    dest_indexs[edge_index] = csr.column_values[edge_index];
                    src_indexs [edge_index] = node_idx;

                    ++edge_index;
                }
            }
        }
    }

    void free(){
        values.free();
        weights.free();
        src_indexs.free();
        dest_indexs.free();
        out_degrees.free();
        degrees.free();
        labels.free();
    }

};

/** @brief define a general problem */
template<typename Value>
struct graph_problem
{
    void execute(graph_structure<Value> &, config_t *);
};

#endif