#define  BLOCK_SIZE 4
#include <iostream>
#include <stdio.h>
__global__
void MatMulKernel(float* A,float* B,float* C,int n){
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < n; ++e)
        Cvalue += A[row * n + e]* B[e * n + col];
  
    C[row * n + col] = Cvalue;


}
int main(){
    int N = 1<<2;
    size_t size = N * N* sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    //initialize h_A and h_B
    for(int i=0;i<N*N;i++){
        h_A[i]=i+1;
        h_B[i]=i+N*N;
    }
    
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
   
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
     // Free device memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     
     // Free host memory
     free(h_A);
     free(h_B);
     cudaDeviceSynchronize();

     for(int i=0;i<N;i++){
         for(int j=0;j<N;j++)
            std::cout<<h_C[i*N+j]<<" ";
        std::cout<<"\n";
     }
     free(h_C);
}
