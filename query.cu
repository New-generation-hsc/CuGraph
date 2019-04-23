#include <iostream>
#include <cstdio>
#include "query.cuh"
#include "utils.hpp"

/**
  * query the cuda properties, include the following essential issue
  * - Shared Memory size per block
  * - Device Count
  * - Device Name
  * - Warp Size
  * - Maximum Thread per SM
  * - Maximum Thread per Block
*/


void query_device(device_list *props){
    int count = 0;

    // allocate memory for device property
    cudaGetDeviceCount(&count);
    props->count = count;
    props->devices = new cudaDeviceProp[count];

    // query each device properties
    for(size_t cnt = 0; cnt < count; ++cnt){
        cudaGetDeviceProperties(&props->devices[cnt], cnt);
    }
}

blockinfo  default_find_blocksize (cudaDeviceProp *prop, 
                                size_t const vsize, 
                                size_t const vertices, 
                                size_t const edges){
    
    /* the maximum number of resident blocks per multiprocessor
     * refer the wikipeida for detail, https://en.wikipedia.org/wiki/CUDA
     */
    int maxBlockPerSM;
    #if __CUDA_ARCH__ < 300
        maxBlockPerSM = 8;
    #endif
    #if __CUDA_ARCH__ >= 300 & __CUDA_ARCH__ < 500
        maxBlockPerSM = 16;
    #endif
    #if __CUDA_ARCH__ >= 500 & __CUDA_ARCH__ < 700
        maxBlockPerSM = 32;
    #endif
    #if __CUDA_ARCH__ > 700
        maxBlockPerSM = 16;
    #endif

    int maxVerticesPerSM = prop->sharedMemPerBlock / vsize;
    
    /* the average window size is |E||N|^2 / |V|^2 involved in the paper
     * the paper assume the average window size equals to wrap size
     */
    int avg_window_size = prop->warpSize;
    // the approximated maximum vertices in the shard
    int approximated_N = (int)std::sqrt((avg_window_size * std::pow(vertices, 2)) / edges);

    //int proper_size = prop->maxThreadsPerMultiProcessor / maxBlockPerSM;

    blockinfo info;

    for(int block = 2; block <= maxBlockPerSM; ++block){
        if(prop->maxThreadsPerMultiProcessor % block != 0) continue;

        int blocksize = prop->maxThreadsPerMultiProcessor / block;
        
        // block size can't large than maximum thread per block
        if(blocksize > prop->maxThreadsPerBlock) continue;

        int shardMaxVertices = maxVerticesPerSM / block;
        if(shardMaxVertices > approximated_N) {
            //std::cout << "block num : " << block << std::endl;
            info.blocksize = blocksize;
            info.shard_max_num_nodes = shardMaxVertices;
        }
    }

    return info;
}


blockinfo find_max_block (cudaDeviceProp *prop,
                          size_t const vsize,
                          size_t const vertices,
                          size_t const edges)
{
    int maxBlockPerSM;
    #if __CUDA_ARCH__ < 300
        maxBlockPerSM = 8;
    #endif
    #if __CUDA_ARCH__ >= 300 & __CUDA_ARCH__ < 500
        maxBlockPerSM = 16;
    #endif
    #if __CUDA_ARCH__ >= 500 & __CUDA_ARCH__ < 700
        maxBlockPerSM = 32;
    #endif
    #if __CUDA_ARCH__ > 700
        maxBlockPerSM = 16;
    #endif

    int maxVerticesPerSM = prop->sharedMemPerBlock / vsize;

    maxBlockPerSM = 2;

    blockinfo info;
    info.blocksize = prop->maxThreadsPerMultiProcessor / maxBlockPerSM;
    info.shard_max_num_nodes = maxVerticesPerSM / maxBlockPerSM;

    return info;
}


find_blocksize find_proper_blocksize = &default_find_blocksize;

/** @brief display some useful information about device
 *  @param[device_list] the array list about device
 *  @param[dev_num] the device No.
*/
void display_device_info(device_list props, int dev_num){

    if( props.count <= 0 || props.count <= dev_num ) {
        HANDLE_ERROR("please check gpu device exists, or choose a valid device.");
    }

    cudaDeviceProp *prop = &(props.devices[dev_num]);

    printf("+--------------------------------------------------------------------------------+\n");
    printf("|   Device %d             %20s                      |\n", dev_num, prop->name);
    printf("+--------------------------------------------------------------------------------+\n");
    printf("|   Total amount of Global Memory                      |     %10zuMBytes    |\n", prop->totalGlobalMem / 1024);
    printf("|   Total amount of Shared Memory Per block            |     %10zuMBytes    |\n", prop->sharedMemPerBlock / 1024);
    printf("|   Maximum Threads Per block                          |     %10d          |\n", prop->maxThreadsPerBlock);
    printf("+--------------------------------------------------------------------------------+\n");
}