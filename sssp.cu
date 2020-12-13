#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<limits.h>

#define NUM_NODES 5

using namespace std;

__global__ void CUDA_SSSP_KERNEL1(int *Va, int *Ea, int *Wa, bool *Ma, int *Ca, int *Ua, bool *done)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id>NUM_NODES)
	{
		*done = false;
	}
	if(Ma[id]==true)
	{
		Ma[id] = false;
		__syncthreads(); 
		int start = Va[id];
		int end = Va[id+1];
		for(int i=start;i<end;i++) 
		{
			int nid = Ea[i];

			if(Ua[nid]>(Ca[nid]+Wa[nid]))
			{
				Ua[nid] = Ca[id] + Wa[nid];
			}
		}
	}
}
__global__ void CUDA_SSSP_KERNEL2(int *Va, int *Ea, int *Wa, bool *Ma, int *Ca, int *Ua, bool *done)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id>NUM_NODES)
	{
		*done = false;
	}
	if(Ca[id]>Ua[id])
	{
		Ca[id] = Ua[id];
		Ma[id] = true;
		*done = false;
	}
	Ua[id] = Ca[id];
}

int main(int argc, char** argv)
{
	int** graph = new int* [NUM_NODES];
	int edges = 0;
    for(int i=0;i<NUM_NODES;i++)
    {
        graph[i] = new int[NUM_NODES];
    }
    for(int i=0;i<NUM_NODES;i++)
    {
        for(int j=i+1;j<NUM_NODES;j++)
        {
			int x = rand()%100;
            if(x!=0)
            {
				edges+=1;
                graph[i][j] = x;
                graph[j][i] = x;
            }
        }
    }
	
	int* v = new int[NUM_NODES+1];
	int* e = new int[2*edges];
	int* w = new int[2*edges];
	int x = 0;
	for(int i=0;i<NUM_NODES;i++)
	{
		v[i] = x;
		for(int j=0;j<NUM_NODES;j++)
		{
			if(graph[i][j]!=0)
			{
				e[x] = j;
				w[x] = graph[i][j];
				x+=1;
			}
		}
	}
	v[NUM_NODES] = x;
	bool mask[NUM_NODES] = { false };
	int cost[NUM_NODES] = { INT_MAX };
	int updated[NUM_NODES] = { INT_MAX };

	int source = 0;
	mask[source] = true;
	updated[source] = 0;
	cost[source] = 0;

	int* Va;
	cudaMalloc(&Va, sizeof(int)*(NUM_NODES+1));
	cudaMemcpy(Va, v, sizeof(int)*(NUM_NODES+1), cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc(&Ea, sizeof(int)*(2*edges));
	cudaMemcpy(Ea, e, sizeof(int)*(2*edges), cudaMemcpyHostToDevice);

	int* Wa;
	cudaMalloc(&Wa, sizeof(int)*(2*edges));
	cudaMemcpy(Wa, w, sizeof(int)*(2*edges), cudaMemcpyHostToDevice);

	bool* Ma;
	cudaMalloc(&Ma, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Ma, mask, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ua;
	cudaMalloc(&Ua, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ua, updated, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ca;
	cudaMalloc(&Ca, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	int num_blks = 1;
	int threads = 5;

	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	
	do {
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_SSSP_KERNEL1<<<num_blks, threads>>>(Va, Ea, Wa, Ma, Ca, Ua, d_done);
		CUDA_SSSP_KERNEL2<<<num_blks, threads>>>(Va, Ea, Wa, Ma, Ca, Ua, d_done);
		cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);
	} while (!done);

	cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);
	
	cout<<"Cost: "<<endl;
	for(int i=0;i<NUM_NODES;i++)
	{
		cout<<cost[i]<<" ";
	}
	cout<<endl;
	return 0;
}

