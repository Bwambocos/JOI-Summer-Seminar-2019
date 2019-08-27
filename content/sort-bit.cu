#include "bits/stdc++.h"
#include <cuda_runtime.h>
#include <sys/time.h>
#define in std::cin
#define out std::cout
#define rep(i,N) for(LL i=0;i<N;++i)
typedef long long int LL;

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

const LL inf = RAND_MAX;

__global__ void solveOnGPU(LL A[], const LL block, const LL step)
{
	LL idx = blockIdx.x * blockDim.x + threadIdx.x;
	LL e = (idx ^ step);
	if (e > idx)
	{
		LL v1 = A[idx];
		LL v2 = A[e];
		if (((idx & block) != 0 && v1 < v2)
			|| ((idx & block) == 0) && v1 > v2)
		{
			A[e] = v1;
			A[idx] = v2;
		}
	}
}

int main()
{
	LL N;
	in >> N;

	// ブロック・スレッド数の上限を取得
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	LL maxThreads = deviceProp.maxThreadsPerBlock;

	// メモリを確保
	LL newN = 1;
	while (newN < N) newN *= 2;
	size_t nBytes = sizeof(LL) * N, newNBytes = sizeof(LL) * newN;
	LL *input, *d_Array, *output;
	input = (LL *)malloc(nBytes);
	output = (LL *)malloc(nBytes);
	cudaMalloc((LL**)&d_Array, newNBytes);

	// 数列を初期化
	srand((unsigned int)time(0));
	rep(i, N) input[i] = rand();
	
	// GPU 処理時間を計測
	auto gpuStartTime = cpuSecond();
	
	LL *temp;
	temp = (LL *)malloc(newNBytes);
	memcpy(temp, input, nBytes);
	for (LL i = N; i < newN; ++i) temp[i] = inf;
	cudaMemcpy(d_Array, temp, newNBytes, cudaMemcpyHostToDevice);
	for (LL block = 2; block <= newN; block *= 2)
	{
		for (LL step = block / 2; step >= 1; step /= 2)
		{
			solveOnGPU<<< std::max(newN / maxThreads, 1LL), std::min(newN, maxThreads) >>>(d_Array, block, step);
		}
	}
	cudaMemcpy(temp, d_Array, newNBytes, cudaMemcpyDeviceToHost);
	memcpy(output, temp, nBytes);
	free(temp);

	auto gpuEndTime = cpuSecond();

	out << (gpuEndTime - gpuStartTime) * 1000 << "ms" << std::endl;

	// 後始末
	free(input);
	free(output);
	cudaFree(d_Array);
}
