#include "compute.cuh"
#include "../../../global.hpp"
#include "../../../timer.cuh"
#include "../../../utils.hpp"
#include <cstdio>


void bfs_graph_init(graph_shard &graph, config_t *conf){
    for(uint nodeIdex = 0; nodeIdex < graph.n; ++nodeIdex){
        if(nodeIdex == conf->source) graph.values[nodeIdex] = 0;
        else graph.values[nodeIdex] = INF;
    }
}

// graph_shard initialize function point to a specific function
graph_init graph_shard_init = &bfs_graph_init;

void write_to_file(graph_shard &graph, config_t *conf){

    FILE *fp = open_file_access(conf->output_path, "w");
    for(uint nodeIdex = 0; nodeIdex < graph.n; ++nodeIdex){
        fprintf(fp, "%u -> %u\n", graph.labels[nodeIdex], graph.values[nodeIdex]);
    }
    fclose(fp);
}

inline __device__ void compute(uint* local, const uint *src){
    if(*src != INF){
        atomicMin(local, *src + 1);
    }
}

__global__ void kernel ( const uint nShards,
                         const uint shardMaxNumVertices,
                         const uint *srcIndex,
                         const uint *destIndex,
                         uint       *srcValues,
                         uint       *values,
                         const uint *shardSizeScan,
                         const uint *windowSizeScan,
                         bool       *lock
                       )
{
    extern __shared__ uint localValues[];

    uint shardOffSet = blockIdx.x * shardMaxNumVertices;
    uint shardStartAddr = shardSizeScan[ blockIdx.x ];
    uint shardEndAddr   = shardSizeScan[ blockIdx.x + 1 ];
    uint *blockValues = values + shardOffSet;

    for(uint tid = threadIdx.x; tid < shardMaxNumVertices; tid += blockDim.x){
        localValues[ tid ] = blockValues[ tid ];
    }
    __syncthreads();

    for(uint entryAddr = shardStartAddr + threadIdx.x;
        entryAddr < shardEndAddr;
        entryAddr += blockDim.x
       )
    {
        compute(localValues + (destIndex[entryAddr] - shardOffSet),
                srcValues   + entryAddr
               );
    }
    __syncthreads();

    bool flag = false;
    for(uint tid = threadIdx.x; tid < shardMaxNumVertices; tid += blockDim.x){
        if(localValues[ tid ] < blockValues[ tid ]){
            flag = true;
            blockValues[ tid ] = localValues[ tid ];
        }
    }

    if(__syncthreads_or(flag)){

        for(uint shardIdx = threadIdx.x / warpSize;
            shardIdx < nShards;
            shardIdx += (blockDim.x / warpSize)
           )
        {
            uint windowStartAddr = windowSizeScan[ shardIdx * nShards + blockIdx.x ];
            uint windowEndAddr   = windowSizeScan[ shardIdx * nShards + blockIdx.x + 1 ];

            for(uint entryAddr = windowStartAddr + ( threadIdx.x & (warpSize - 1));
                entryAddr < windowEndAddr;
                entryAddr += warpSize
               )
            {
                srcValues[entryAddr] = localValues [ srcIndex[entryAddr] - shardOffSet ];
            }
        }

        if(threadIdx.x == 0) *lock = true;
    }
}

void process( const uint blocksize,
              const uint shardMaxNumVertices,
              const uint nShards,
              uint      *values,
              const uint *windowSizeScan,
              const uint *shardSizeScan,
              uint      *srcValues,
              const uint *srcIndex,
              const uint *destIndex,
              const uint max_iterations,
              bool       verbose
            )
{
    bool lock;
    bool *dev_lock;
    cudaMalloc(&dev_lock, sizeof(bool));

    uint iterations = 0;
    GpuTimer timer;
    float total_time = 0.0f;

    printf("+--------------------------------------------------------------------------------+\n");
    printf("| Times |                     function                              |  costs(ms) +\n");
    printf("+--------------------------------------------------------------------------------+\n");

    do {
        lock = false;
        timer.start_record();

        cudaMemcpyAsync(dev_lock, &lock, sizeof(bool), cudaMemcpyHostToDevice);
        kernel<<< nShards, blocksize, sizeof(uint) * shardMaxNumVertices >>> 
            (
                nShards,
                shardMaxNumVertices,
                srcIndex,
                destIndex,
                srcValues,
                values,
                shardSizeScan,
                windowSizeScan,
                dev_lock
            );
        cudaPeekAtLastError();
        cudaMemcpyAsync(&lock, dev_lock, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        ++iterations;
        float elapsed = timer.stop_record();
        total_time += elapsed;
        if(verbose){
            printf("| %3dth |                  kernel function iteration                |   %3.4f   |\n", iterations, elapsed);
        }
    }while(lock && iterations < max_iterations);

    if(verbose){
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| total |                                                           |   %3.4f   |\n", total_time );
        printf("+--------------------------------------------------------------------------------+\n");
        printf("| avg   |                                                           |   %3.4f   |\n", total_time / iterations);
        printf("+--------------------------------------------------------------------------------+\n");
    }
}

void execute(graph_shard &graph, config_t *conf){

    buffer<uint>   dev_values( DEVICE );
    buffer<uint>   dev_srcValues( DEVICE );
    buffer<uint>   dev_destIndex ( DEVICE );
    buffer<uint>   dev_srcIndex( DEVICE );
    buffer<uint>   dev_shardSizeScan( DEVICE );
    buffer<uint>   dev_windowSizeScan( DEVICE );

    dev_values  = graph.values;
    dev_srcValues = graph.src_values;
    dev_srcIndex  = graph.src_indexs;
    dev_destIndex = graph.dest_indexs;

    dev_shardSizeScan = graph.shards_size;
    dev_windowSizeScan = graph.windows_size;

    cudaDeviceSynchronize();

    process ( graph.blocksize,
              graph.shard_max_num_nodes,
              graph.num_shards,
              dev_values.ptr,
              dev_windowSizeScan.ptr,
              dev_shardSizeScan.ptr,
              dev_srcValues.ptr,
              dev_srcIndex.ptr,
              dev_destIndex.ptr,
              conf->maximum_iterations,
              conf->verbose
            );

    graph.values = dev_values;
    write_to_file(graph, conf);
    
    dev_values.free();
    dev_srcValues.free();
    dev_srcIndex.free();
    dev_destIndex.free();
    dev_shardSizeScan.free();
    dev_windowSizeScan.free();

    cudaDeviceReset();
}