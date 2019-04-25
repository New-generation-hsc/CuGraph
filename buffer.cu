#include "buffer.cuh"

void hostpinned_malloc(void **ptr, size_t const size){
    assert(size != 0);
    cudaHostAlloc(ptr, size, cudaHostAllocPortable);
}

void hostpinned_free(void *ptr){
    assert(ptr != NULL);
    cudaFree(ptr);
}

void host_malloc(void **ptr, size_t const size){
    assert(size != 0);
    *ptr = malloc(size);
}

void host_free(void *ptr){
    assert(ptr != NULL);
    free(ptr);
}

void device_malloc(void **ptr, size_t const size){
    assert(size != 0);
    cudaMalloc(ptr, size);
}

void device_free(void *ptr){
    assert(ptr != NULL);
    cudaFree(ptr);
}

/** memory copy functions */

void host_to_host(void *dest, const void* src, size_t count){
    memcpy(dest, src, count);
}

void host_to_device(void *dest, const void* src, size_t count){
    cudaMemcpy(dest, src, count, cudaMemcpyHostToDevice);
}

void hostpinned_to_device(void *dest, const void *src, size_t count){
    cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice);
}

void device_to_host(void *dest, const void* src, size_t count){
    cudaMemcpy(dest, src, count, cudaMemcpyDeviceToHost);
}

void device_to_hostpinned(void *dest, const void *src, size_t count){
    cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost);
}

void host_to_hostpinned(void *dest, const void *src, size_t count){
    host_to_host(dest, src, count);
}

// instantiate the buffer allocator to specific
template<>
void (*buffer_allocator<HOSTPINNED>::buffer_malloc)(void **, size_t const) = &hostpinned_malloc;

template<>
void (*buffer_allocator<HOSTPINNED>::buffer_free)(void *) = &hostpinned_free;

template<>
void (*buffer_allocator<HOST>::buffer_malloc)(void **, size_t const) = &host_malloc;


template<>
void (*buffer_allocator<HOST>::buffer_free)(void *) = &host_free;

template<>
void (*buffer_allocator<DEVICE>::buffer_malloc)(void **, size_t const) = &device_malloc;

template<>
void (*buffer_allocator<DEVICE>::buffer_free)(void *) = &device_free;

void (*memory_copy[3][3])(void *, const void *, size_t) = {
    {
        NULL,
        NULL,
        &hostpinned_to_device
    },
    {
        &host_to_hostpinned,
        &host_to_host,
        &host_to_device
    },
    {
        &device_to_hostpinned,
        &device_to_host,
        NULL
    }
};