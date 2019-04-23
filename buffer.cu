#include "buffer.cuh"

void host_malloc(buffer_allocator *allocator, void **ptr, size_t const size){
    assert(allocator != NULL && size != 0);
    cudaHostAlloc(ptr, size, cudaHostAllocPortable);
}

void host_free(buffer_allocator *allocator, void *block){
    assert(allocator != NULL && block != NULL);
    cudaFree(block);
}

void device_malloc(buffer_allocator *allocator, void **ptr, size_t const size){
    assert(allocator != NULL && size != 0);
    cudaMalloc(ptr, size);
}

void device_free(buffer_allocator *allocator, void *block){
    assert(allocator != NULL && block != NULL);
    cudaFree(block);
}

void default_malloc(buffer_allocator *allocator, void **ptr, size_t const size){
    assert(allocator != NULL && size != 0);
    *ptr = malloc(size);
}

void default_free(buffer_allocator *allocator, void *block){
    assert(allocator != NULL && block != NULL);
    free(block);
}

buffer_allocator host_allocator  = {
    host_malloc,
    host_free
};

buffer_allocator device_allocator = {
    device_malloc,
    device_free
};

buffer_allocator default_allocator = {
    default_malloc,
    default_free
};

buffer_allocator allocators[3] = {
    host_allocator,
    default_allocator,
    device_allocator
};

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

memory_copy copy_funcs = {
    {
        NULL,
        NULL,
        &hostpinned_to_device
    },
    {
        NULL,
        &host_to_host,
        &host_to_device
    },
    {
        &device_to_hostpinned,
        &device_to_host,
        NULL
    }
};