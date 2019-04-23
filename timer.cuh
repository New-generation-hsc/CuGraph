#ifndef __GPU_TIMER__
#define __GPU_TIMER__

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer(){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer(){
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void start_record(){
		cudaEventRecord(start, 0);
	}

	float stop_record(){
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsed;
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

#endif