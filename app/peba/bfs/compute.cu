#include "../skelecton.cuh"
#include "../../../timer.cuh"
#include "../../../utils.hpp"
#include "../../../global.hpp"

__global__ void kernel_init( const uint nodes,
                             const uint source,
                             uint       *values,
                             const uint *degrees,
                             uint       *active_degrees
                           )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < nodes){
        values[tid] = INF;
        active_degrees[tid] = 0;

        if(tid == source){
            values[tid] = 0;
            active_degrees[tid] = degrees[tid];
        }
    }
}

__global__ void kernel_relax( const uint num_edges,
                              uint       *values,
                              const uint *srcIndex,
                              const uint *destIndex,
                              const uint *outdegrees,
                              uint       *active_degrees,
                              bool       *lock
                            )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid < num_edges && active_degrees[srcIndex[tid]] > 0){
        if(values[destIndex[tid]] > values[srcIndex[tid]] + 1){
            *lock = true;
            atomicMin(values + destIndex[tid], values[srcIndex[tid]] + 1);
            atomicExch(active_degrees + destIndex[tid], outdegrees[tid]);
        }
        atomicSub(active_degrees + srcIndex[tid], 1);
    }
}

void cuda_bfs  ( const uint nodes,
                 const uint num_edges,
                 const uint source,
                 const uint blocksize,
                 const uint maximum_iterations,
                 uint       *values,
                 const uint *srcIndex,
                 const uint *destIndex,
                 const uint *outdegrees,
                 const uint *degrees,
                 uint       *active_degrees,
                 bool       verbose
               )
{
    uint init_blocks  = (nodes + blocksize - 1) / blocksize;
    uint relax_blocks = (num_edges + blocksize - 1) / blocksize;

    GpuTimer timer;
    timer.start_record();

    // initialize the whole graph values and active degree for each node
    kernel_init<<< init_blocks, blocksize >>> ( nodes,
                                                source,
                                                values,
                                                degrees,
                                                active_degrees
                                              );
    
    cudaDeviceSynchronize();
    float elapsed = timer.stop_record();

    if(verbose) {
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| Times |                     function                              |  costs(ms) +\n");
        printf("+--------------------------------------------------------------------------------+\n");
        printf("|  1th  |            kernel initialization function                 |   %0.4f   |\n", elapsed);
        printf("+--------------------------------------------------------------------------------+\n");
    }

    bool lock = true;
    bool *dev_lock;
    cudaMalloc(&dev_lock, sizeof(bool));

    int iterations = 0;
    float total_time = 0;

    while(lock && iterations < maximum_iterations){

        timer.start_record();
        
        lock = false;
        cudaMemcpy(dev_lock, &lock, sizeof(bool), cudaMemcpyHostToDevice);

        kernel_relax <<< relax_blocks, blocksize >>> ( num_edges,
                                                       values,
                                                       srcIndex,
                                                       destIndex,
                                                       outdegrees,
                                                       active_degrees,
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

template<>
void graph_problem<uint>::execute(graph_structure<uint> &graph, config_t *conf)
{
    buffer<uint, DEVICE> dev_values;
    dev_values = graph.values;
    buffer<uint, DEVICE> dev_srcIndex;
    dev_srcIndex = graph.src_indexs;
    buffer<uint, DEVICE> dev_destIndex;
    dev_destIndex = graph.dest_indexs;
    buffer<uint, DEVICE> dev_outdegrees;
    dev_outdegrees = graph.out_degrees;
    buffer<uint, DEVICE> dev_degrees;
    dev_degrees = graph.degrees;

    buffer<uint, DEVICE> dev_active_degrees ( graph.n );

    cuda_bfs  ( graph.n,
                graph.m,
                conf->source,
                conf->blocksize,
                conf->maximum_iterations,
                dev_values.ptr,
                dev_srcIndex.ptr,
                dev_destIndex.ptr,
                dev_outdegrees.ptr,
                dev_degrees.ptr,
                dev_active_degrees.ptr,
                conf->verbose
              );
    
    graph.values = dev_values;

    // write sssp result to file
    write_to_file(conf->output_path, graph.values.ptr, graph.labels.ptr, graph.n);

    // free memory
    graph.free();

    dev_values.free();
    dev_srcIndex.free();
    dev_destIndex.free();
    dev_outdegrees.free();
    dev_degrees.free();
    dev_active_degrees.free();
}