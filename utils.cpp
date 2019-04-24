#include "utils.hpp"
#include "config.hpp"

#include <cstring>

void handle_error(const char* msg, const char *file, int line){
	printf("[ line %d in %s ] error : %s\n", line, file, msg);
	exit(EXIT_FAILURE);
}

// safe open file to access
FILE *open_file_access(const char *filename, const char *mode){
	FILE *fp = fopen(filename, mode);

	if(fp == NULL){
		// open failed
		fprintf(stderr, "open file : %s failed. program terminated.\n", filename);
		exit(EXIT_FAILURE);
	}
	return fp;
}

void global_config_init(config_t *conf){
	strcpy(conf->output_path, "out.txt");
	conf->device = 0;
	conf->source = 0;
	conf->undirected = false;
	conf->init_prval = 1.0;
	conf->maximum_iterations = 100;
	conf->factor = 0.85;
	conf->threshold = 1e-4;
	conf->verbose = true;
}

/* instantiate the template print helper to specific  */
template<>
const char* template_print<uint>::fmt = "%u";

template<>
const char* template_print<float>::fmt = "%.4f";