#ifndef __GRAPH_UTILS__
#define __GRAPH_UTILS__

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <ctime>

struct cpu_timer {

	std::chrono::steady_clock::time_point t1;

	void start(){
		t1 = std::chrono::steady_clock::now();
	}

	double stop(){
		auto t2 = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		return elapsed;
	}

};

void handle_error(const char*, const char*, int);

#define HANDLE_ERROR(msg) (handle_error(msg, __FILE__, __LINE__))

FILE *open_file_access(const char*, const char*);

#endif