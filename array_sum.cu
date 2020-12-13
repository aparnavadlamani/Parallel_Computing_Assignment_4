#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<math.h>
#include<assert.h>

using namespace std;

__global__ void array_sum(int* v, int* r)
{
	__shared__ int partial_sum[1024];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	partial_sum[threadIdx.x] = v[tid];
	__syncthreads();

	for(int s = blockDim.x/2; s>0; s>>=1)
	{
		if(threadIdx.x<s)
		{
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x+s];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		r[blockIdx.x] = partial_sum[0];
	}
}

int main(int argc, char const *argv[])
{
	int n;
	srand(0);
	cout<<"Enter value for n"<<endl;
    cin>>n;
	
	
	int *v, *r;
	int *d_v, *d_r;

	v = (int*)malloc(n*sizeof(int));
	r = (int*)malloc(n*sizeof(int));

	cudaMalloc(&d_v, n*sizeof(int));
	cudaMalloc(&d_r, n*sizeof(int));

	for(int i=0;i<n;i++)
	{
		v[i] = rand()%100;
	}

	cudaMemcpy(d_v, v, n*sizeof(int), cudaMemcpyHostToDevice);

	int num_threads = 256;
	int num_blocks = n/num_threads;

	array_sum<<<num_blocks, num_threads>>> (d_v, d_r);
	array_sum<<<1, num_threads>>> (d_r, d_r);

	cudaMemcpy(r, d_r, n*sizeof(int), cudaMemcpyDeviceToHost);

	cout<<"Array Sum: "<<r[0]<<endl;

	return 0;
}