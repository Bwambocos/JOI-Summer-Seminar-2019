#include "bits/stdc++.h"
#include <cuda_runtime.h>
#define in std::cin
#define out std::cout
#define rep(i,N) for(LL i=0;i<N;++i)
typedef long long int LL;

// #define CHECK(call)
// {
// 	const cudaError_t error = call;
// 	if (error != cudaSuccess)
// 	{
// 		out << "Error: " << __FILE__ << ":" << __LINE__ << ", ";
// 		out << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl;
// 		exit(1);
// 	}
// }

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

int main(int argc, char **argv)
{
	out << argv[0] << " Starting...\n";

	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 32;
	out << "Vector size " << nElem << "\n";

	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice);

	dim3 block(1);
	dim3 grid(nElem);

	sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C);
	out << "Execution configure <<<" << grid.x << ", " << block.x << ">>>\n";

	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

	sumArraysOnHost(h_A, h_B, hostRef, nElem);

	checkResult(hostRef, gpuRef, nElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();
}
