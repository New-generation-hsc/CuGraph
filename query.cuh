#ifndef __QUERY_CUH__
#define __QUERY_CUH__

/**
  * query the cuda properties, include the following essential issue
  * - Shared Memory size per block
  * - Device Count
  * - Device Name
  * - Warp Size
  * - Maximum Thread per SM
  * - Maximum Thread per Block
*/
struct device_list {
    cudaDeviceProp *devices; // array of devices
    size_t count; // the number of devices

    void destroy(){
        if(devices != nullptr) delete[] devices;
    }
};


/**
 * block information include the following details
 */
struct blockinfo {
    size_t blocksize;
    size_t shard_max_num_nodes;
};

/**
 * determine a proper blocksize for better performance
 * the find block size function is a function pointer, which means that you can
 * custom yourself find block size function. the default method is the approch
 * involve in the CuSha paper
 */

blockinfo default_find_blocksize (cudaDeviceProp *prop, 
                                size_t const vsize, 
                                size_t const vertices, 
                                size_t const edges);

blockinfo find_max_block(cudaDeviceProp *prop,
                         size_t const vsize,
                         size_t const vertices,
                         size_t const edges);

typedef blockinfo (*find_blocksize)(cudaDeviceProp *prop, 
                                   size_t const vsize, 
                                   size_t const vertices, 
                                   size_t const edges);

extern find_blocksize find_proper_blocksize;

void query_device (device_list *props);

void display_device_info(device_list devices, int dev_num = 0);

#endif