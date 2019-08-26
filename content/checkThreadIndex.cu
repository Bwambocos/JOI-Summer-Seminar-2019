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

void printMatrix(int *C, const int nx, const int ny)
{
	int *ic = C;
	out << "\nMatrix: (" << nx << "." << ny << ")\n";
	rep(iy, ny)
	{
		rep(ix, nx)
		{
			out << ic[ix];
		}
		ic += nx;
		out << "\n";
	}
	out << "\n";
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %d ival %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv)
{
	out << argv[0] << " Starting...\n";

	// デバイス情報を取得
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	out << "Using Device " << dev << ": " << deviceProp.name << "\n";
	CHECK(cudaSetDevice(dev));

	// 行列の次元を設定
	int nx = 8, ny = 6;
	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);

	// ホストメモリを確保
	int *h_A = (int *)malloc(nBytes);

	// ホスト行列を整数で初期化
	rep(i, nxy) h_A[i] = i;
	printMatrix(h_A, nx, ny);

	// デバイスメモリを確保
	int *d_MatA;
	CHECK(cudaMalloc((void **)&d_MatA, nBytes));

	// ホストからデバイスへデータを転送
	CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

	// 実行設定をセットアップ
	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// カーネルを呼び出す
	printThreadIndex<<< grid, block >>>(d_MatA, nx, ny);
	CHECK(cudaDeviceSynchronize());

	// ホストとデバイスのメモリを解放
	CHECK(cudaFree(d_MatA));
	free(h_A);

	// デバイスをリセット
	CHECK(cudaDeviceReset());
}
