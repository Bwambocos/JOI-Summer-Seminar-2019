#include "bits/stdc++.h"
#include <cuda_runtime.h>
#include <sys/time.h>
#define in std::cin
#define out std::cout
#define rep(i,N) for(LL i=0;i<N;++i)
typedef long long int LL;

#define CHECK(call)	\
{	\
	const cudaError_t error = call;	\
	if (error != cudaSuccess)	\
	{	\
		out << "Error: " << __FILE__ << ":" << __LINE__ << ", ";	\
		out << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl;	\
		exit(1);	\
	}	\
}	\

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float* ip, int size)
{
	time_t t;
	srand((unsigned int) time(&t));
	rep(i, size) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArraysOnHost(float* A, float* B, float* C, const int N)
{
	rep(i, N) C[i] = A[i] + B[i];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = 1;

	rep(i, N)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = 0;
			out << "Arrays do not match!" << std::endl;
			out << "host " << hostRef[i] << " gpu " << gpuRef[i] << " at current " << i << std::endl;
			break;
		}
	}

	if (match) out << "Arrays match.\n\n";
}

int main(int argc, char **argv)
{
	out << argv[0] << " Starting...\n";

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	out << "Using Device " << dev << ": " << deviceProp.name << "\n";
	CHECK(cudaSetDevice(dev));

	int nElem = 1 << 24;
	out << "Vector size " << nElem << "\n";

	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	double iStart, iElaps;
	iStart = cpuSecond();
	initialData(h_A, nElem);
	initialData(h_B, nElem);
	iElaps = cpuSecond() - iStart;
	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	iStart = cpuSecond();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iElaps = cpuSecond() - iStart;

	float *d_A, *d_B, *d_C;
	CHECK(cudaMalloc((float**)&d_A, nBytes));
	CHECK(cudaMalloc((float**)&d_B, nBytes));
	CHECK(cudaMalloc((float**)&d_C, nBytes));

	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

	int iLen = 1024;
	dim3 block(iLen);
	dim3 grid((nElem + block.x - 1) / block.x);

	iStart = cpuSecond();
	sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C);
	CHECK(cudaDeviceSynchronize());
	iElaps = cpuSecond() - iStart;
	float gpuScore = iElaps;
	out << "sumArraysOnGPU <<<" << grid.x << ", " << block.x << ">>> Time elapsed " << gpuScore << "sec\n";

	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

	iStart = cpuSecond();
	sumArraysOnHost(h_A, h_B, hostRef, nElem);
	iElaps = cpuSecond() - iStart;
	float cpuScore = iElaps;
	out << "sumArraysOnCPU Time elapsed " << cpuScore << "sec\n";
	
	out << cpuScore / gpuScore << std::endl;

	checkResult(hostRef, gpuRef, nElem);

	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();
}
