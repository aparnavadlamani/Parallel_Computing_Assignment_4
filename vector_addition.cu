#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cassert>
#include<iostream>
#include<stdio.h>

using namespace std;

__global__ void vectorAdd(int *a, int* b, int* c, int n)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(tid<n)
	{
		c[tid] = a[tid] + b[tid];
	}
}

void verify_result(int* a, int* b, int* c, int n)
{
	for(int i=0;i<n;i++)
	{
		assert(c[i]==a[i]+b[i]);
	}
}

int main(int argc, char const *argv[])
{
	int n;
	srand(0);
	cout<<"Enter value for n"<<endl;
    cin>>n;
	
	int *a, *b, *c;

	a = (int*)malloc(n*sizeof(int));
	b = (int*)malloc(n*sizeof(int));
	c = (int*)malloc(n*sizeof(int));

	for(int i=0;i<n;i++)
	{
		a[i] = rand()%100;
		b[i] = rand()%100;
	}

	int* d_a, *d_b, *d_c;
	cudaMalloc(&d_a, n*sizeof(int));
	cudaMalloc(&d_b, n*sizeof(int));
	cudaMalloc(&d_c, n*sizeof(int));

	cudaMemcpy(d_a, a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n*sizeof(int), cudaMemcpyHostToDevice);

	int num_threads = 1<<10;

	int num_blocks = (n+num_threads-1)/num_threads;
	vectorAdd<<<num_blocks, num_threads>>>(d_a, d_b, d_c, n);

	cudaMemcpy(c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

	cout<<"Vector Sum: ";
	for(int i=0;i<n;i++)
	{
		cout<<c[i]<<" ";
	}
	cout<<endl;
	verify_result(a, b, c, n);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}