#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<iostream>
#define BLOCK_SIZE 16

using namespace std;

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void verify_result(int *a, int *b, int *c, int m, int n, int k) 
{
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += a[i*n+h]*b[h*k+j];
            }
            assert(c[i*k+j]==tmp);
        }
    }
}

int main(int argc, char const *argv[])
{
    int m, n, k;
    srand(0);
    cout<<"Enter values for m n and k"<<endl;
    cin>>m>>n>>k;

    int *a, *b, *c;
    a = (int*)malloc(m*n*sizeof(int));
	b = (int*)malloc(n*k*sizeof(int));
	c = (int*)malloc(m*k*sizeof(int));
    
    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = rand() % 100;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = rand() % 100;
        }
    }

    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    
    
    // Transefr results from device to host 
    cudaMemcpy(c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    verify_result(a, b, c, m, n, k);
    cout<<"Product matrix is: "<<endl;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<k;j++)
        {
            cout<<c[i*k+j]<<" ";
        }
        cout<<endl;
    }
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(a);
    free(b);
    free(c);
    return 0;
}