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

void initialData(float* ip, int size)
{
	time_t t;
	srand((unsigned int) time(&t));
	rep(i, size) ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
	float *ia = A;
	float *ib = B;
	float *ic = C;
	rep(iy, ny)
	{
		rep(ix, nx)
		{
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	if (ix < nx)
	{
		rep(iy, ny)
		{
			int idx = iy * nx + ix;
			MatC[idx] = MatA[idx] + MatB[idx];
		}
	}
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

	// デバイスのセットアップ
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	out << "Using Device " << dev << ": " << deviceProp.name << "\n";
	CHECK(cudaSetDevice(dev));

	// 行列のデータサイズを設定
	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	out << "Matrix size: nx " << nx << " ny " << ny << "\n";

	// ホストメモリを確保
	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	hostRef = (float *)malloc(nBytes);
	gpuRef = (float *)malloc(nBytes);

	// ホスト側でデータを初期化
	double iStart;
	iStart = cpuSecond();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	double iElaps = cpuSecond() - iStart;

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// 結果をチェックするためにホスト側で行列を加算
	iStart = cpuSecond();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	iElaps = cpuSecond() - iStart;
	double cpuTime = iElaps;

	// デバイスのグローバルメモリを確保
	float *d_MatA, *d_MatB, *d_MatC;
	CHECK(cudaMalloc((float**)&d_MatA, nBytes));
	CHECK(cudaMalloc((float**)&d_MatB, nBytes));
	CHECK(cudaMalloc((float**)&d_MatC, nBytes));

	// ホストからデバイスへデータを転送
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

	int dimx = 128, dimy = 1;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, 1);

	iStart = cpuSecond();
	sumMatrixOnGPU1D<<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
	CHECK(cudaDeviceSynchronize());
	iElaps = cpuSecond() - iStart;
	float gpuTime = iElaps;
	out << "sumMatrixOnGPU1D <<< (" << grid.x << "," << grid.y << "), (" << block.x << "," << block.y << ") >>> Time elapsed " << gpuTime << "sec\n";
	// カーネルエラーをチェック
	CHECK(cudaGetLastError());

	// カーネルの結果をホスト側にコピー
	CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

	// デバイスの結果をチェック
	checkResult(hostRef, gpuRef, nxy);
	
	out << "cpuTime : " << cpuTime * 1000 << "ms\n";
	out << "gpuTime : " << gpuTime * 1000 << "ms\n";
	out << cpuTime / gpuTime << std::endl;

	// ホストのメモリを解放
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	// デバイスをリセット
	CHECK(cudaDeviceReset());
}
