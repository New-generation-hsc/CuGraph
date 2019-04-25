#ifndef __GRAPH_DEFINITIONS__
#define __GRAPH_DEFINITIONS__

#include <vector>
#include <cstdio>
typedef unsigned int uint;

struct edge_t
{
    // edge information, source index and destination index
    uint src, dest;

    // edge weight
    uint weight;
};


// a generic adjencent list graph representation
struct graph_t
{
    // No. number of vertics
    uint n;

    // No. number of edges
    uint m;

    // adjencent list representation
    std::vector<std::vector<edge_t>> adj;

    // the node labels
    std::vector<uint> labels;
};



// a simple direct compressed sparse row format graph represention
struct csr_graph
{
    // No. number of vertices
    uint n;

    // No. number of edges
    uint m;

    // the node indices in column values array
    uint *row_offsets;    

    // the destination of each edge
    uint *column_values;

    uint *edge_weights;

    // the vertex labels
    uint *labels;

    // No. number of neighbors of each node, outdegrees
    uint *neighbors;

    // No. number of each node outdegrees
    uint *degrees;

    // whthter destination based format
    bool dest_based;

    void set_up(uint nodes, uint edges, bool based = false){
        n = nodes;
        m = edges;

        // allocate memory
        row_offsets   = new uint[ nodes + 1 ];
        row_offsets[0] = 0;
        column_values = new uint[ edges ];
        edge_weights  = new uint[ edges ];
        labels        = new uint[ nodes ];
        neighbors     = new uint[ edges ];
        degrees       = new uint[ nodes ];

        dest_based = based;
    }

    void destroy(){
        n = 0;
        m = 0;

        if(row_offsets != nullptr) delete[] row_offsets;
        if(column_values != nullptr) delete[] column_values;
        if(edge_weights != nullptr)  delete[] edge_weights;
        if(labels != nullptr) delete[] labels;
        if(neighbors != nullptr) delete[] neighbors;
        if(degrees != nullptr ) delete[] degrees;
    }
};


void parse_graph(FILE *, graph_t &, bool, bool, bool);
void parse_graph(FILE *, csr_graph &, bool, bool, bool);


#endif