#ifndef __GRAPH_BUFFER__
#define __GRAPH_BUFFER__

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <stdexcept>

enum MemType{
    HOSTPINNED, // data in host   pinned memory
    HOST,       // data in host   global memory
    DEVICE      // data in device global memory
};

template<MemType Type>
struct buffer_allocator {
    static void (*buffer_malloc)(void **ptr, size_t const size);
    static void (*buffer_free)(void *block);
};

// memory copy function
extern void (*memory_copy[3][3])(void *, const void *, size_t);

template< typename T, MemType type = HOSTPINNED >
struct buffer{

    // specific memory type
    static const MemType buffer_type = type;

    // No. number of elements
    size_t num_elems;

    // elements pointer
    T *ptr;

    // static allocator, all same memory type buffer share the same allocator
    static const buffer_allocator<type> allocator;

    // default construct
    buffer(){
        num_elems = 0;
        ptr       = NULL;
    }

    /* @brief construct n elements buffer */
    buffer(size_t n){
        num_elems = n;
        allocator.buffer_malloc((void**)&ptr, sizeof(T) * num_elems);
    }

    void alloc(size_t n){
        num_elems = n;
        allocator.buffer_malloc((void**)&ptr, sizeof(T) * num_elems);
    }

    void free(){
        num_elems = 0;
        allocator.buffer_free((void*)ptr);
    }

    template<MemType U>
    buffer<T, type>& operator=(buffer<T, U> &buf){

        // if this buffer has no storage, then allocate storage
        if(num_elems == 0){
            alloc(buf.num_elems);
        }

        size_t sizebytes = buf.size();
        sizebytes = ((sizebytes < this->size()) ? sizebytes : this->size());

        assert(memory_copy[U][type] != NULL);

        (memory_copy[U][type])((void*)ptr, (const void*)buf.ptr, sizebytes);

        return *this;
    }

    buffer<T, type>& operator=(T *buf){
        if(num_elems != 0){
            (memory_copy[HOST][type])((void*)ptr, (const void*)buf, num_elems * sizeof(T));
        }
        return *this;
    }

    size_t size(){
        return sizeof(T) * num_elems;
    }

    T& operator[](size_t index){
        if(index >= num_elems){
            throw std::out_of_range("buffer index out of range");
        }
        return ptr[index];
    }
};

#endif