#include "bits/stdc++.h"
#include <cuda_runtime.h>
#include <sys/time.h>
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

double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void solveOnCPU(LL N, LL *A)
{
	std::sort(A, A + N);
}

const LL inf = 1145148101919364364;
__global__ void solveOnGPU(LL N, LL *A)
{
	LL newN = 1;
	while (newN < N) newN *= 2;
	LL B[112345678];
	rep(i, newN)
	{
		if (i < N) B[i] = A[i];
		else B[i] = inf;
	}
	
	for (LL block = 2; block <= newN; block *= 2)
	{
		for (LL step = block / 2; step >= 1; step /= 2) 
		{
			LL ixj = threadIdx.x ^ step;
			if (ixj > threadIdx.x)
			{
				if ((threadIdx.x & block) == 0)
				{
					if (B[threadIdx.x] > B[ixj])
					{
						LL t = B[threadIdx.x];
						B[threadIdx.x] = B[ixj];
						B[ixj] = t;
					}
				}
				else
				{
					if (B[threadIdx.x] < B[ixj])
					{
						LL t = B[threadIdx.x];
						B[threadIdx.x] = B[ixj];
						B[ixj] = t;
					}
				}
			}
			__syncthreads();
		}
	}
	rep(i, N) A[i] = B[i];
}

int main()
{
	LL S, T;
	in >> S >> T;

	std::ofstream writing_file;
	writing_file.open("output.txt", std::ios::trunc);

	out << "N\tCPUTime\t\tGPUTime\t\tStatus\t\tRatio\t\tWinner" << std::endl;
	for (LL N = S; N <= T; ++N)
	{
		// メモリを確保
		size_t nBytes = sizeof(LL) * N;
		LL *h_A, *h_B, *d_B, *cpuRes, *gpuRes;
		h_A = (LL *)malloc(nBytes);
		h_B = (LL *)malloc(nBytes);
		cpuRes = (LL *)malloc(nBytes);
		gpuRes = (LL *)malloc(nBytes);
		CHECK(cudaMalloc((LL**)&d_B, nBytes));

		// 数列を初期化
		srand((unsigned int)time(0));
		rep(i, N) h_A[i] = rand();

		// CPU 処理時間を計測
		auto cpuStartTime = cpuSecond();

		h_B = h_A;
		solveOnCPU(N, h_B);

		auto cpuEndTime = cpuSecond();
		
		// GPU 処理時間を計測
		auto gpuStartTime = cpuSecond();

		CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice))
		solveOnGPU<<< 1, N, nBytes >>>(N, d_B);
		CHECK(cudaDeviceSynchronize());

		auto gpuEndTime = cpuSecond();

		// ジャッジ
		std::sort(h_A, h_A + N);
		cpuRes = h_B;
		CHECK(cudaMemcpy(gpuRes, d_B, nBytes, cudaMemcpyDeviceToHost));
		bool flag = true;
		rep(i, N)
		{
			if (h_A[i] != cpuRes[i] || h_A[i] != gpuRes[i])
			{
				flag = false;
				break;
			}
		}
		out << std::fixed << std::setprecision(7);
		out << N << "\t" << (cpuEndTime - cpuStartTime) * 1000 << "ms\t" << (gpuEndTime - gpuStartTime) * 1000 << "ms\t" << (flag ? "Accepted\t" : "Wrong Answer\t") << (cpuEndTime - cpuStartTime) / (gpuEndTime - gpuStartTime) << "\t" << ((cpuEndTime - cpuStartTime) / (gpuEndTime - gpuStartTime) > 1. ? "GPU" : "CPU") << std::endl;
		writing_file << N << "," << (cpuEndTime - cpuStartTime) * 1000 << "," << (gpuEndTime - gpuStartTime) * 1000 << std::endl;
		if (!flag) return 0;

		// 後始末
		free(cpuRes);
		free(gpuRes);
		CHECK(cudaFree(d_B));
		CHECK(cudaDeviceReset());
	}
}
