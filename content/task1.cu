#include "bits/stdc++.h"
#include <cuda_runtime.h>
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

__global__ void addOne(int& n)
{
	n++;
}

int main()
{
	size_t nBytes = sizeof(int);

	int *h_A;
	h_A = (int *)malloc(nBytes);

	int *d_A;
	CHECK(cudaMalloc(&d_A, nBytes));

	addOne<<< 100, 100 >>>(*d_A);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(h_A, d_A, nBytes, cudaMemcpyDeviceToHost));
	out << *h_A << std::endl;
	
	free(h_A);
	CHECK(cudaFree(d_A));
	CHECK(cudaDeviceReset());
}
