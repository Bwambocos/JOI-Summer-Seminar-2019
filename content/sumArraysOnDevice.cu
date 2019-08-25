#include <stdlib.h>
#include <time.h>
#include <stdio.h>

__global__ void sumArraysOnDevice(float* A, float* B, float* C, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] + B[i];
	}
}

void initialData(float* ip, int size)
{
	// —”ƒV[ƒh‚ðì¬
	time_t t;
	srand((unsigned int) time(&t));

	for (int i = 0; i < size; ++i)
	{
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
	}

	return;
}

int main()
{
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	h_C = (float*)malloc(nBytes);
	printf("%s\n", cudaGetErrorString(cudaMalloc((float**)&d_A, nBytes)));
	printf("%s\n", cudaGetErrorString(cudaMalloc((float**)&d_B, nBytes)));
	printf("%s\n", cudaGetErrorString(cudaMalloc((float**)&d_C, nBytes)));

	initialData(h_A, nElem);
	initialData(h_B, nElem);
	printf("%s\n", cudaGetErrorString(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice)));
	printf("%s\n", cudaGetErrorString(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice)));

	sumArraysOnDevice<<<1, 1>>>(d_A, d_B, d_C, nElem);
	cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaDeviceReset();
	return 0;
}
