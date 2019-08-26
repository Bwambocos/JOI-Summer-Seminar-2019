#include "bits/stdc++.h"
#include <cuda_runtime.h>
#define in std::cin
#define out std::cout
#define rep(i,N) for(LL i=0;i<N;++i)
typedef long long int LL;

#define CHECK(call)																			\
{																							\
	const cudaError_t error = call;															\
	if (error != cudaSuccess)																\
	{																						\
		out << "Error: " << __FILE__ << ":" << __LINE__ << ", ";							\
		out << "code:" << error << ", reason: " << cudaGetErrorString(error) << std::endl;	\
		exit(1);																			\
	}																						\
}																							\

__global__ void calcSum(int n, int *a, int& s)
{
	rep(i, n) s += a[i];
}

int main()
{
	int N;
	in >> N;
	size_t nBytes = sizeof(int) * N;

	int *h_A, *d_A, *h_S, *d_S;
	h_A = (int *)malloc(nBytes);
	CHECK(cudaMalloc((int**)&d_A, nBytes));
	h_S = (int *)malloc(sizeof(int));
	CHECK(cudaMalloc(&d_S, sizeof(int)));

	rep(i, N) h_A[i] = i + 1;

	CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
	calcSum<<< 1, 1 >>>(N, d_A, *d_S);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(h_S, d_S, sizeof(int), cudaMemcpyDeviceToHost));
	out << *h_S << std::endl;
	
	free(h_A);
	CHECK(cudaFree(d_A));
	free(h_S);
	CHECK(cudaFree(d_S));
	CHECK(cudaDeviceReset());
}
