/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "descriptor.h"
#include "filter.h"
#include <emmintrin.h>
#include <fstream>
using namespace std;

Descriptor::Descriptor(uint8_t* I,int32_t width,int32_t height,int32_t bpl,bool half_resolution) {
  I_desc        = (uint8_t*)_mm_malloc(16*width*height*sizeof(uint8_t),16);
  uint8_t* I_du = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  uint8_t* I_dv = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  filter::sobel3x3(I,I_du,I_dv,bpl,height);
  createDescriptor(I_du,I_dv,width,height,bpl,half_resolution);
  _mm_free(I_du);
  _mm_free(I_dv);
}

Descriptor::~Descriptor() {
  _mm_free(I_desc);
}

__global__ void createDescriptorKernel(uint8_t* I_du,uint8_t* I_dv,uint8_t* I_desc,int32_t width,int32_t height,int32_t bpl,bool half_resolution){
  int u,v;
  int idx=blockIdx.x*blockDim.x + threadIdx.x;
  v=idx/width;
  u=idx%width;

  if(v<3 || v>=height-3  || u<3 || u>=width-3)
  return;
  if(half_resolution && v%2==1)
  return;
  uint8_t *I_desc_curr;  
  uint32_t addr_v0,addr_v1,addr_v2,addr_v3,addr_v4;
  addr_v2 = v*bpl;
  addr_v0 = addr_v2-2*bpl;
  addr_v1 = addr_v2-1*bpl;
  addr_v3 = addr_v2+1*bpl;
  addr_v4 = addr_v2+2*bpl;
  I_desc_curr = I_desc+(v*width+u)*16;
  *(I_desc_curr++) = *(I_du+addr_v0+u+0);
  *(I_desc_curr++) = *(I_du+addr_v1+u-2);
  *(I_desc_curr++) = *(I_du+addr_v1+u+0);
  *(I_desc_curr++) = *(I_du+addr_v1+u+2);
  *(I_desc_curr++) = *(I_du+addr_v2+u-1);
  *(I_desc_curr++) = *(I_du+addr_v2+u+0);
  *(I_desc_curr++) = *(I_du+addr_v2+u+0);
  *(I_desc_curr++) = *(I_du+addr_v2+u+1);
  *(I_desc_curr++) = *(I_du+addr_v3+u-2);
  *(I_desc_curr++) = *(I_du+addr_v3+u+0);
  *(I_desc_curr++) = *(I_du+addr_v3+u+2);
  *(I_desc_curr++) = *(I_du+addr_v4+u+0);
  *(I_desc_curr++) = *(I_dv+addr_v1+u+0);
  *(I_desc_curr++) = *(I_dv+addr_v2+u-1);
  *(I_desc_curr++) = *(I_dv+addr_v2+u+1);
  *(I_desc_curr++) = *(I_dv+addr_v3+u+0);

}
void Descriptor::createDescriptor (uint8_t* I_du,uint8_t* I_dv,int32_t width,int32_t height,int32_t bpl,bool half_resolution) {
  
  size_t size;
  uint8_t* d_I_du;
  uint8_t* d_I_dv;
  size=bpl*height*sizeof(uint8_t);
  cudaMalloc(&d_I_du,size);
  cudaMalloc(&d_I_dv,size);
  cudaMemcpy(d_I_du,I_du,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_I_dv,I_dv,size,cudaMemcpyHostToDevice);
  uint8_t* d_I_desc;
  size=16*width*height*sizeof(uint8_t);
  cudaMalloc(&d_I_desc,size);
  int threadsPerBlock=1024;
  int numBlocks=(width*height+threadsPerBlock-1)/threadsPerBlock;
  createDescriptorKernel<<<numBlocks,threadsPerBlock>>>(d_I_du,d_I_dv,d_I_desc,width,height,bpl,half_resolution);
  cudaMemcpy(I_desc,d_I_desc,size,cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_I_du);
  cudaFree(d_I_dv);
  cudaFree(d_I_desc);

  // ofstream file;
  // file.open("desc_parallel.txt");
  // for(int i=0;i<height;i++)
  //   for(int j=0;j<width;j++)

  //   for(int k=0;k<16;k++)
  //     file<<(int)I_desc[16*(width*i+j)+k]<<"\n";
  // file.close();

}
