#include <cstdio>
#include <cstdlib>

#include "../query.cuh"

int main(int argc, char* argv[]){
    if(argc < 2){
        printf("Usage: %s [device]", argv[0]);
        exit(EXIT_FAILURE);
    }

    device_list devices;
    query_device(&devices);

    display_device_info(devices, atoi(argv[1]));

    return 0;
}