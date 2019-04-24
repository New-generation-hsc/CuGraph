#ifndef __COMPUTE_GRAPH__
#define __COMPUTE_GRAPH__

/**
 * the file constribute to implement a parallel node based algorithm
*/
#include <iostream>
#include <cstring>
#include <cstdio>

#include "../../../global.hpp"
#include "../../../buffer.cuh"
#include "../../../timer.cuh"
#include "../../../defs.hpp"
#include "../../../utils.hpp"
#include "../../../config.hpp"

/* @brief kernel function, initialize the graph values */
template<typename Value>
__global__ void kernel_init( Value  *values,
                             const uint nodes,
                             const uint source,
                             int    *flags
                           )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < nodes){
        values[tid] = INF;
        flags[tid] = 0;

        if(tid == source){
            values[tid] = 0;
            flags[tid] = 1;
        }
    }
}

template<typename Value>
__global__ void kernel_relax( const uint   nodes,
                              Value        *values,
                              const uint   *offsets,
                              const uint   *edges,
                              const Value  *weights,
                              int          *flags,
                              bool         *lock
                            )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < nodes && flags[tid] == 1){
        flags[tid] = 0;

        // updates tid node neighbors
        uint start_addr = offsets[tid];
        uint end_addr   = offsets[tid + 1];

        for(uint eid = start_addr; eid < end_addr; ++eid){
            uint dest = edges[eid];
            if(values[dest] > values[tid] + weights[eid]){
                *lock = true;
                atomicMin(values + dest, values[tid] + weights[eid]);
                atomicExch(flags + dest, 1);
            }
        }
    }
}

/* @brief for each node update its neighbors and set its flag 
 * @param[nodes] number of vertices of graph
 * @param[blocksize] number of threads in a block
 * @param[values] distance from source node to each other node
 * @param[offsets] the neighbors start offset
 * @param[edges] the neighbors destination
 * @param[weights] weight of each edge
 * @param[flags] indicate each node whther had been accessed
 * @param[verbose] whther print some useful information
*/
template<typename Value>
void cuda_sssp( const uint    nodes,
                const uint    blocksize,
                const uint    source,
                Value        *values,
                const uint   *offsets,
                const uint   *edges,
                const Value  *weights,
                int          *flags,
                bool         verbose = false
              )
{
    uint blocks = (nodes + blocksize - 1) / blocksize;
    bool lock = true;
    bool *dev_lock;
    cudaMalloc(&dev_lock, sizeof(bool));

    GpuTimer timer;
    timer.start_record();
    
    kernel_init<<< blocks, blocksize >>> ( values,
                                           nodes,
                                           source,
                                           flags
                                         );
    cudaDeviceSynchronize();
    double elapsed = timer.stop_record();

    if(verbose) {
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| Times |                     function                              |  costs(ms) +\n");
        printf("+--------------------------------------------------------------------------------+\n");
        printf("|  1th  |            kernel initialization function                 |   %0.4f   |\n", elapsed);
        printf("+--------------------------------------------------------------------------------+\n");
    }

    int iterations = 0;
    float total_time = 0.0;

    while(lock){

        timer.start_record();

        lock = false;
        cudaMemcpy(dev_lock, &lock, sizeof(bool), cudaMemcpyHostToDevice);
        kernel_relax<<< blocks, blocksize >>> ( nodes,
                                                values,
                                                offsets,
                                                edges,
                                                weights,
                                                flags,
                                                dev_lock
                                              );
        cudaDeviceSynchronize();
        cudaMemcpy(&lock, dev_lock, sizeof(bool), cudaMemcpyDeviceToHost);

        elapsed = timer.stop_record();
        total_time += elapsed;
        ++iterations;

        if(verbose){
            printf("| %3dth |                  kernel function iteration                |   %3.4f   |\n", iterations, elapsed);
        }
    }

    
    if(verbose){
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| total |                                                           |   %3.4f   |\n", total_time );
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| avg   |                                                           |   %3.4f   |\n", total_time / iterations);
        printf("+--------------------------------------------------------------------------------+\n");
    }
}

/* @brief allocate device memory and initialze the device memory
 * @parm[csr] the csr format graph
 * @param[source] the sssp source node index
 * @param[blocksize] number of threads in a block
 * @param[verbose] whther print some useful information
*/

template<typename Value>
void execute( csr_graph       &csr,
              config_t        *conf
            )
{
    // allocate device memory
    buffer<Value> dev_values(DEVICE);
    dev_values.alloc(csr.n);

    buffer<uint> dev_edges(csr.m, DEVICE);
    dev_edges = csr.column_values;

    buffer<Value> dev_weights(csr.m, DEVICE);
    dev_weights = csr.edge_weights;

    buffer<uint> dev_offsets(csr.n + 1, DEVICE);
    dev_offsets = csr.row_offsets;

    buffer<int> dev_flags(DEVICE);
    dev_flags.alloc(csr.n);

    cuda_sssp( csr.n,
               conf->blocksize,
               conf->source,
               dev_values.ptr,
               dev_offsets.ptr,
               dev_edges.ptr,
               dev_weights.ptr,
               dev_flags.ptr,
               conf->verbose
            );
    
    buffer<Value> values;
    values = dev_values;

    write_to_file(conf->output_path, values.ptr, csr.labels, csr.n);
    
    /* free the memory */
    dev_values.free();

    dev_edges.free();

    dev_weights.free();

    dev_offsets.free();

    dev_flags.free();

    values.free();

    csr.destroy();
}

#endif