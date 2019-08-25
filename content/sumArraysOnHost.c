#include <stdlib.h>
#include <time.h>
#include <stdio.h>

void sumArraysOnHost(float* A, float* B, float* C, const int N)
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

	float* h_A, * h_B, * h_C;

	h_A = (float*)malloc(nBytes);
	h_B = (float*)malloc(nBytes);
	h_C = (float*)malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	sumArraysOnHost(h_A, h_B, h_C, nElem);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
