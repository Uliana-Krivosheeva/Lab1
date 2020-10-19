

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#define BLOCK_SIZE 16
using namespace std;

__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}
 
void cpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k) {
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			int tmp = 0;
			for (int h = 0; h < n; h++)
			{
				tmp += a[i * n + h] * b[h * k + j];
			}
			c[i * k + j] = tmp;
		}
	}
}

int main(int argc, char const* argv[])
{
	// ������ ������� ������
	int m, n, k;
	printf("please type in m n and k\n");
	scanf("%d %d %d", &m, &n, &k);


	// �������� ������ �� �����
	int* h_a, *h_b, *h_c, *h_res; // ��������� ������� ����� ��� ���������� ������������ �� �����, � �� ����� ��� � 3� ����� ������������ ��������� � �������
	cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
	cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
	cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
	cudaMallocHost((void**)&h_res, sizeof(int) * m * k);
	float time_cpu, time_gpu;

	// ������� ��� �������� ������� ������ ������������ �� �������
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//������ ������� �������
	cudaEventRecord(start, 0);

	// �������� ������ �� �������
	int* d_a, *d_b, *d_c;
	cudaMalloc((void**)&d_a, sizeof(int) * m * n);
	cudaMalloc((void**)&d_b, sizeof(int) * n * k);
	cudaMalloc((void**)&d_c, sizeof(int) * m * k);

	// �������� �������� ������� � ����� �� ������
	cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

	// ���������� ������ ����� � �����
	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// ��������� ����
	gpu_matrix_mult << < dimGrid, dimBlock >> > (d_a, d_b, d_c, m, n, k);

	// �������� ��������� � ������� �� ����
	cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	// ����� ������� �������
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// ������� ����� ������ �� �������
	cudaEventElapsedTime(&time_gpu, start, stop);


	// ������ ������� �������
	auto begin = chrono::steady_clock::now();

	// ��������� ������������ �� �����
	cpu_matrix_mult(h_a, h_b, h_res, m, n, k);

	// ����� ������� �������
	auto end = chrono::steady_clock::now();

	// ������� ����� ������ �� �����
	auto elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end - begin);
	float time_cpu = elapsed_ms.count();


	printf("CPU time: %f ms.\n\n", time_cpu);
	printf("GPU time: %f ms.\n\n", time_gpu);


	//������� ������
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFreeHost(h_res);

	return 0;
}




