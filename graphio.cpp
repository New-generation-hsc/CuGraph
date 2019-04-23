#include <fstream>
#include <cstring>

#include "global.hpp"
#include "defs.hpp"
#include "utils.hpp"

void parse_graph( FILE          *fp, 
                  graph_t       &graph,
                  const bool    undirected,
                  const bool    dest_based,
                  const bool    verbose
                )
{
    uint nodes = 0, edges = 0;
    uint mapper[MAX_NUM_VERTICES]; // whther given node labels exists
    memset(mapper, -1, sizeof(uint) * MAX_NUM_VERTICES);

    char line[LINE_MAX_LEN];
    uint src, dest, weight;

    if(verbose) {
        fprintf(stdout, "start parse graph...\n");
    }

    cpu_timer timer;
    timer.start(); // start record

    while(true){
        // read fail, or eof
        if(fscanf(fp, "%[^\n]\n", line) <= 0) break;

        // skip the comments
        if(line[0] < '0' || line[0] > '9') continue;

        int ret = sscanf(line, "%u%u%u", &src, &dest, &weight);
        if(ret == 2){
            // weight is not defined in data file
            weight = 0;
        }else if(ret < 2){
            HANDLE_ERROR("graph data file format is invalid.");
        }

        // transform the labels into index
        if(mapper[src] == -1){
            mapper[src] = nodes;
            graph.labels.push_back(src);
            nodes++;
        }

        if(mapper[dest] == -1){
            mapper[dest] = nodes;
            graph.labels.push_back(dest);
            nodes++;
        }

        edge_t edge { mapper[src], mapper[dest], weight };

        uint maxnode = std::max(src, dest);
        if(graph.adj.size()  <= maxnode){
            graph.adj.resize( maxnode + 1 );
        }

        if(dest_based){
            graph.adj[mapper[dest]].push_back(edge);
            edges++;

            if(undirected){
                edge_t back_edge { mapper[dest], mapper[src], weight };
                graph.adj[mapper[src]].push_back( back_edge );
                edges++;
            }
        }
        else {
            graph.adj[mapper[src]].push_back( edge );
            edges++;

            if(undirected){
                edge_t back_edge { mapper[dest], mapper[src], weight };
                graph.adj[mapper[dest]].push_back( back_edge );
                edges++;
            }
        }

        if(verbose){
            fprintf(stdout, "\rparsing %uth lines..., please wait for a minutes.", edges);
            fflush(stdout);
        }

        if(nodes >= MAX_NUM_VERTICES){
            fprintf(stdout, "exceed the max number of vertices.\n");
            break;
        }
    }

    graph.n = nodes;
    graph.m = edges;

    if(verbose){
        fprintf(stdout, "\rparsing finished. ( nodes, edges ) : ( %u, %u ). costs : %.3lf (ms).\n", nodes, edges, timer.stop());
        
    }
}

/** @brief parse the graph data, construct the csr format
 *  @param[fp] FILE pointer
 *  @param[graph] csr graph structure
 *  @param[undirected] whther graph is undirected
 *  @param[dest_based] the csr destination format
 *  @param[verbose] print some useful info
*/
void parse_graph( FILE       *fp,
                  csr_graph  &graph,
                  const bool undirected,
                  const bool dest_based,
                  const bool verbose
                )
{
    // parse graph into adjencent list
    graph_t tmp_graph;
    parse_graph(fp, tmp_graph, undirected, dest_based, verbose);

    graph.set_up(tmp_graph.n, tmp_graph.m);

    uint offset = 0;
    for(uint nodeIdx = 0; nodeIdx < tmp_graph.n; ++nodeIdx){  // graph node Index
        for(uint nbrIdx = 0; nbrIdx < tmp_graph.adj[nodeIdx].size(); ++nbrIdx){ // graph node neighbors index
            edge_t edge = tmp_graph.adj[nodeIdx][nbrIdx];

            if(dest_based){
                graph.column_values[offset] = edge.src;
            }else{
                graph.column_values[offset] = edge.dest;
            }
            graph.edge_weights[offset] = edge.weight;

            ++offset;
        }

        // set node neighbor index
        graph.row_offsets[nodeIdx + 1] = offset;
        graph.labels[nodeIdx] = tmp_graph.labels[nodeIdx];
    }
}