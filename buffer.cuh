#ifndef __GRAPH_BUFFER__
#define __GRAPH_BUFFER__

#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <iostream>


enum MemType{   // memory type
    HOSTPINNED, // data in host   pinned memory
    HOST,       // data in host   global memory
    DEVICE      // data in device global memory
};

struct buffer_allocator {
    void (*buffer_malloc)(buffer_allocator *allocator, void **ptr, size_t const size);
    void (*buffer_free)  (buffer_allocator *allocator, void *block);
};

extern buffer_allocator allocators[3];

typedef void (*memory_copy[3][3])(void *, const void*, size_t );
extern memory_copy copy_funcs;

template<typename T>
struct buffer{
    size_t nElems;
    T *ptr;         // array of elements
    buffer_allocator *allocator;
    MemType type;

    buffer(){
        nElems     = 0;
        ptr        = NULL;
        type       = HOST;
        allocator = &allocators[type];
    }

    buffer(size_t n){
        type = HOST;
        allocator = &allocators[type];
        nElems = n;
        allocator->buffer_malloc(allocator, (void**)&ptr, sizeof(T) * nElems);
    }

    buffer(MemType t){
        nElems    = 0;
        ptr       = NULL;
        type = t;
        allocator = &allocators[type];
    }

    buffer(size_t n, MemType t){
        nElems  = n;
        ptr     = NULL;
        type    = t;
        allocator = &allocators[type];
        allocator->buffer_malloc(allocator, (void**)&ptr, sizeof(T) * nElems);
    }

    void alloc(size_t n){
        nElems = n;
        allocator->buffer_malloc(allocator, (void**)&ptr, sizeof(T) * nElems);
    }

    void free(){
        nElems = 0;
        allocator->buffer_free(allocator, (void*)ptr);
        ptr = NULL;
    }

    buffer<T>& operator=(buffer<T> & buf){
        
        if(nElems == 0){
            alloc(buf.nElems); // allocate n elements for storage
        }

        assert (allocator && buf.allocator && 
            copy_funcs[buf.type][type]);
        
        size_t size = buf.size();

        size = ((size < this->size()) ? size : this->size());

        (copy_funcs[buf.type][type])((void*)ptr, (const void*)buf.ptr, size);
        return *this;
    }

    buffer<T>& operator=(const T *buf){
        if(allocator != NULL && nElems != 0){
            (copy_funcs[HOSTPINNED][type])((void*)ptr, (const void*)buf, nElems * sizeof(T));
        }
        return *this;
    }

    size_t size(){
        return sizeof(T) * nElems;
    }

    T& operator[](size_t index) {
        if(index >= nElems){
            throw std::runtime_error("buffer index out of range");
        }
        return ptr[index];
    }
};

#endif