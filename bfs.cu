#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#define NUM_NODES 5

using namespace std;

__global__ void CUDA_BFS_KERNEL(int *Va, int *Ea, bool *Fa, bool *Xa, int *Ca, bool *done)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id>NUM_NODES)
	{
		*done = false;
	}
	if(Fa[id]==true&&Xa[id]==false)
	{
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads(); 
		int start = Va[id];
		int end = Va[id+1];
		for(int i=start;i<end;i++) 
		{
			int nid = Ea[i];

			if(Xa[nid]==false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}
		}
	}
}
// The BFS frontier corresponds to all the nodes being processed at the current level.

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
            if(rand()%2==1)
            {
				edges+=1;
                graph[i][j] = 1;
                graph[j][i] = 1;
            }
        }
    }
	
	int* v = new int[NUM_NODES+1];
	int* e = new int[2*edges];
	int x = 0;
	for(int i=0;i<NUM_NODES;i++)
	{
		v[i] = x;
		for(int j=0;j<NUM_NODES;j++)
		{
			if(graph[i][j]!=0)
			{
				e[x] = j;
				x+=1;
			}
		}
	}
	v[NUM_NODES] = x;
	bool frontier[NUM_NODES] = { false };
	bool visited[NUM_NODES] = { false };
	int cost[NUM_NODES] = { 0 };

	int source = 0;
	frontier[source] = true;

	int* Va;
	cudaMalloc((void**)&Va, sizeof(int)*(NUM_NODES+1));
	cudaMemcpy(Va, v, sizeof(int)*(NUM_NODES+1), cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc((void**)&Ea, sizeof(int)*(2*edges));
	cudaMemcpy(Ea, e, sizeof(int)*(2*edges), cudaMemcpyHostToDevice);

	bool* Fa;
	cudaMalloc((void**)&Fa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	bool* Xa;
	cudaMalloc((void**)&Xa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);

	int* Ca;
	cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);

	int num_blks = 1;
	int threads = 5;

	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	
	do {
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_BFS_KERNEL <<<num_blks, threads >>>(Va, Ea, Fa, Xa, Ca, d_done);
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

