#include "../skelecton.cuh"
#include "../../../timer.cuh"
#include "../../../utils.hpp"
#include "../../../global.hpp"

__global__ void kernel_init( const uint nodes,
                             float      *values,
                             const float val
                           )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < nodes){
        values[tid] = val;
    }
}

__global__ void kernel_relax( const uint num_edges,
                              float      *values,
                              float      *result,
                              const uint *srcIndex,
                              const uint *destIndex,
                              const uint *outdegrees
                            )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid < num_edges){
        atomicAdd(result + destIndex[tid], values[srcIndex[tid]] / outdegrees[tid]);
    }
}

__global__ void kernel_update( const uint  nodes,
                               float       *values, 
                               float       *result,
                               const float factor,
                               const float threshold,
                               bool *lock
                             )
{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < nodes){
        result[tid] = result[tid] * factor + (1.0 - factor) / nodes;
        if(fabs(result[tid] - values[tid]) > threshold){
            *lock = true;
        }
        values[tid] = result[tid];
        result[tid] = 0.0;
    }
}

void cuda_pagerank( const uint nodes,
                    const uint num_edges,
                    const float init_prval,
                    const float factor,
                    const float threshold,
                    const uint blocksize,
                    const uint maximum_iterations,
                    float       *values,
                    float       *result,
                    const uint *srcIndex,
                    const uint *destIndex,
                    const uint *outdegrees,
                    bool       verbose
                  )
{
    uint init_blocks  = (nodes + blocksize - 1) / blocksize;
    uint relax_blocks = (num_edges + blocksize - 1) / blocksize;

    GpuTimer timer;
    timer.start_record();

    // initialize the whole graph values and active degree for each node
    kernel_init<<< init_blocks, blocksize >>> ( nodes,
                                                values,
                                                init_prval
                                              );
                                
    kernel_init<<< init_blocks, blocksize >>> ( nodes,
                                                result,
                                                0.0
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
                                                       result,
                                                       srcIndex,
                                                       destIndex,
                                                       outdegrees
                                                     );
        cudaDeviceSynchronize();

        // check whther divergence
        kernel_update <<< init_blocks, blocksize >>> ( nodes, 
                                                       values,
                                                       result,
                                                       factor,
                                                       threshold,
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
void graph_problem<float>::execute(graph_structure<float> &graph, config_t *conf)
{
    buffer<float, DEVICE> dev_values;
    dev_values = graph.values;
    buffer<float, DEVICE> dev_result;
    dev_result = graph.values;
    buffer<uint, DEVICE> dev_srcIndex;
    dev_srcIndex = graph.src_indexs;
    buffer<uint, DEVICE> dev_destIndex;
    dev_destIndex = graph.dest_indexs;
    buffer<uint, DEVICE> dev_outdegrees;
    dev_outdegrees = graph.src_degrees;

    cuda_pagerank ( graph.n,
                    graph.m,
                    conf->init_prval,
                    conf->factor,
                    conf->threshold,
                    conf->blocksize,
                    conf->maximum_iterations,
                    dev_values.ptr,
                    dev_result.ptr,
                    dev_srcIndex.ptr,
                    dev_destIndex.ptr,
                    dev_outdegrees.ptr,
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
    dev_result.free();

}