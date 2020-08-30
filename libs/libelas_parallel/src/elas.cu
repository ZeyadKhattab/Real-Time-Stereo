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

#include "elas.h"

#include <algorithm>
#include <math.h>
#include "descriptor.h"
#include "triangle.h"
#include "matrix.h"
#include <fstream>

using namespace std;
#define BLOCK_X 32
// #define numBlocks 32
void Elas::hello(int x){
    std::cout<<x<<"\n";
  }
  __device__ uint32_t getAddressOffsetImage_device (const int32_t& u,const int32_t& v,const int32_t& width) {
    return v*width+u;
  }

  __device__ uint32_t getAddressOffsetGrid_device (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
    return (y*width+x)*disp_num+d;
  }
  inline uint32_t getAddressOffsetImage (const int32_t& u,const int32_t& v,const int32_t& width) {
    return v*width+u;
  }

  inline uint32_t getAddressOffsetGrid (const int32_t& x,const int32_t& y,const int32_t& d,const int32_t& width,const int32_t& disp_num) {
    return (y*width+x)*disp_num+d;
  }
 void Elas::process (uint8_t* I1_,uint8_t* I2_,float* D1,float* D2,const int32_t* dims){
  
  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;
  
  // copy images to byte aligned memory
  I1 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  I2 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  memset (I1,0,bpl*height*sizeof(uint8_t));
  memset (I2,0,bpl*height*sizeof(uint8_t));
  if (bpl==dims[2]) {
    memcpy(I1,I1_,bpl*height*sizeof(uint8_t));
    memcpy(I2,I2_,bpl*height*sizeof(uint8_t));
  } else {
    for (int32_t v=0; v<height; v++) {
      memcpy(I1+v*bpl,I1_+v*dims[2],width*sizeof(uint8_t));
      memcpy(I2+v*bpl,I2_+v*dims[2],width*sizeof(uint8_t));
    }
  }

#ifdef PROFILE
  timer.start("Descriptor");  
#endif
  Descriptor desc1(I1,width,height,bpl,param.subsampling);
  Descriptor desc2(I2,width,height,bpl,param.subsampling);

#ifdef PROFILE
  timer.start("Support Matches");
#endif
  vector<support_pt> p_support = computeSupportMatches(desc1.I_desc,desc2.I_desc);
  
  // if not enough support points for triangulation
  if (p_support.size()<3) {
    cout << "ERROR: Need at least 3 support points!" << endl;
    _mm_free(I1);
    _mm_free(I2);
    return;
  }
  

#ifdef PROFILE
  timer.start("Delaunay Triangulation");
#endif
  vector<triangle> tri_1 = computeDelaunayTriangulation(p_support,0);
  vector<triangle> tri_2 = computeDelaunayTriangulation(p_support,1);

#ifdef PROFILE
  timer.start("Disparity Planes");
#endif
  computeDisparityPlanes(p_support,tri_1,0);
  computeDisparityPlanes(p_support,tri_2,1);

#ifdef PROFILE
  timer.start("Grid");
#endif

  // allocate memory for disparity grid
  int32_t grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
  int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
  int32_t grid_dims[3] = {param.disp_max+2,grid_width,grid_height};
  int32_t* disparity_grid_1 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));
  int32_t* disparity_grid_2 = (int32_t*)calloc((param.disp_max+2)*grid_height*grid_width,sizeof(int32_t));
  
  createGrid(p_support,disparity_grid_1,grid_dims,0);
  createGrid(p_support,disparity_grid_2,grid_dims,1);
#ifdef PROFILE
  timer.start("Matching");
#endif
  computeDisparity(p_support,tri_1,disparity_grid_1,grid_dims,desc1.I_desc,desc2.I_desc,0,D1);
  computeDisparity(p_support,tri_2,disparity_grid_2,grid_dims,desc1.I_desc,desc2.I_desc,1,D2);

#ifdef PROFILE
  timer.start("L/R Consistency Check");
#endif
  leftRightConsistencyCheck(D1,D2);

#ifdef PROFILE
  timer.start("Remove Small Segments");
#endif
  removeSmallSegments(D1);
  if (!param.postprocess_only_left)
    removeSmallSegments(D2);

#ifdef PROFILE
  timer.start("Gap Interpolation");
#endif
  gapInterpolation(D1);
  if (!param.postprocess_only_left)
    gapInterpolation(D2);

  if (param.filter_adaptive_mean) {
#ifdef PROFILE
    timer.start("Adaptive Mean");
#endif
    adaptiveMean(D1);
    // if (!param.postprocess_only_left)
    //   adaptiveMean(D2);
  }

  if (param.filter_median) {
#ifdef PROFILE
    timer.start("Median");
#endif
    median(D1);
    if (!param.postprocess_only_left)
      median(D2);
  }

#ifdef PROFILE
  timer.plot();
#endif

  // release memory
  free(disparity_grid_1);
  free(disparity_grid_2);
  _mm_free(I1);
  _mm_free(I2);
}

void Elas::removeInconsistentSupportPoints (int16_t* D_can,int32_t D_can_width,int32_t D_can_height) {
  
  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {
        
        // compute number of other points supporting the current point
        int32_t support = 0;
        for (int32_t u_can_2=u_can-param.incon_window_size; u_can_2<=u_can+param.incon_window_size; u_can_2++) {
          for (int32_t v_can_2=v_can-param.incon_window_size; v_can_2<=v_can+param.incon_window_size; v_can_2++) {
            if (u_can_2>=0 && v_can_2>=0 && u_can_2<D_can_width && v_can_2<D_can_height) {
              int16_t d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
              if (d_can_2>=0 && abs(d_can-d_can_2)<=param.incon_threshold)
                support++;
            }
          }
        }
        
        // invalidate support point if number of supporting points is too low
        if (support<param.incon_min_support)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::removeRedundantSupportPoints(int16_t* D_can,int32_t D_can_width,int32_t D_can_height,
                                        int32_t redun_max_dist, int32_t redun_threshold, bool vertical) {
  
  // parameters
  int32_t redun_dir_u[2] = {0,0};
  int32_t redun_dir_v[2] = {0,0};
  if (vertical) {
    redun_dir_v[0] = -1;
    redun_dir_v[1] = +1;
  } else {
    redun_dir_u[0] = -1;
    redun_dir_u[1] = +1;
  }
    
  // for all valid support points do
  for (int32_t u_can=0; u_can<D_can_width; u_can++) {
    for (int32_t v_can=0; v_can<D_can_height; v_can++) {
      int16_t d_can = *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width));
      if (d_can>=0) {
        
        // check all directions for redundancy
        bool redundant = true;
        for (int32_t i=0; i<2; i++) {
          
          // search for support
          int32_t u_can_2 = u_can;
          int32_t v_can_2 = v_can;
          int16_t d_can_2;
          bool support = false;
          for (int32_t j=0; j<redun_max_dist; j++) {
            u_can_2 += redun_dir_u[i];
            v_can_2 += redun_dir_v[i];
            if (u_can_2<0 || v_can_2<0 || u_can_2>=D_can_width || v_can_2>=D_can_height)
              break;
            d_can_2 = *(D_can+getAddressOffsetImage(u_can_2,v_can_2,D_can_width));
            if (d_can_2>=0 && abs(d_can-d_can_2)<=redun_threshold) {
              support = true;
              break;
            }
          }
          
          // if we have no support => point is not redundant
          if (!support) {
            redundant = false;
            break;
          }
        }
               
        // invalidate support point if it is redundant
        if (redundant)
          *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width)) = -1;
      }
    }
  }
}

void Elas::addCornerSupportPoints(vector<support_pt> &p_support) {
  
  // list of border points
  vector<support_pt> p_border;
  p_border.push_back(support_pt(0,0,0));
  p_border.push_back(support_pt(0,height-1,0));
  p_border.push_back(support_pt(width-1,0,0));
  p_border.push_back(support_pt(width-1,height-1,0));
  
  // find closest d
  for (int32_t i=0; i<p_border.size(); i++) {
    int32_t best_dist = 10000000;
    for (int32_t j=0; j<p_support.size(); j++) {
      int32_t du = p_border[i].u-p_support[j].u;
      int32_t dv = p_border[i].v-p_support[j].v;
      int32_t curr_dist = du*du+dv*dv;
      if (curr_dist<best_dist) {
        best_dist = curr_dist;
        p_border[i].d = p_support[j].d;
      }
    }
  }
  
  // for right image
  p_border.push_back(support_pt(p_border[2].u+p_border[2].d,p_border[2].v,p_border[2].d));
  p_border.push_back(support_pt(p_border[3].u+p_border[3].d,p_border[3].v,p_border[3].d));
  
  // add border points to support points
  for (int32_t i=0; i<p_border.size(); i++)
    p_support.push_back(p_border[i]);
}

inline int16_t Elas::computeMatchingDisparity (const int32_t &u,const int32_t &v,uint8_t* I1_desc,uint8_t* I2_desc,const bool &right_image) {
  
  const int32_t u_step      = 2;
  const int32_t v_step      = 2;
  const int32_t window_size = 3;
  
  int32_t desc_offset_1 = -16*u_step-16*width*v_step;
  int32_t desc_offset_2 = +16*u_step-16*width*v_step;
  int32_t desc_offset_3 = -16*u_step+16*width*v_step;
  int32_t desc_offset_4 = +16*u_step+16*width*v_step;
  
  __m128i xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;
  uint8_t *a1,*a2,*a3,*a4,*a5;
  a1=(uint8_t*)malloc(16*sizeof(uint8_t));
  a2=(uint8_t*)malloc(16*sizeof(uint8_t));
  a3=(uint8_t*)malloc(16*sizeof(uint8_t));
  a4=(uint8_t*)malloc(16*sizeof(uint8_t));
  a5=(uint8_t*)malloc(16*sizeof(uint8_t));

  // check if we are inside the image region
  if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
    
    // compute desc and start addresses
    int32_t  line_offset = 16*width*v;
    uint8_t *I1_line_addr,*I2_line_addr;
    if (!right_image) {
      I1_line_addr = I1_desc+line_offset;
      I2_line_addr = I2_desc+line_offset;
    } else {
      I1_line_addr = I2_desc+line_offset;
      I2_line_addr = I1_desc+line_offset;
    }

    // compute I1 block start addresses
    uint8_t* I1_block_addr = I1_line_addr+16*u;
    uint8_t* I2_block_addr;
    
    // we require at least some texture
    int32_t sum = 0;
    for (int32_t i=0; i<16; i++)
      sum += abs((int32_t)(*(I1_block_addr+i))-128);
    if (sum<param.support_texture)
      return -1;
    
    // load first blocks to xmm registers
    xmm1 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_1));
    xmm2 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_2));
    xmm3 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_3));
    xmm4 = _mm_load_si128((__m128i*)(I1_block_addr+desc_offset_4));
    for(int i=0;i<16;i++){
      a1[i] = *(I1_block_addr+desc_offset_1+i);
      a2[i] = *(I1_block_addr+desc_offset_2+i);
      a3[i] = *(I1_block_addr+desc_offset_3+i);
      a4[i] = *(I1_block_addr+desc_offset_4+i);
      // std::cout<<a1[i]<<"\n";
    }
    // declare match energy for each disparity
    int32_t u_warp;
    // best match
    int16_t min_1_E = 32767;
    int16_t min_1_d = -1;
    int16_t min_2_E = 32767;
    int16_t min_2_d = -1;

    // get valid disparity range
    int32_t disp_min_valid = max(param.disp_min,0);
    int32_t disp_max_valid = param.disp_max;
    if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);
    else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);
    
    // assume, that we can compute at least 10 disparities for this pixel
    if (disp_max_valid-disp_min_valid<10)
      return -1;

    // for all disparities do
    for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {
      int32_t cost=0;

      // warp u coordinate
      if (!right_image) u_warp = u-d;
      else              u_warp = u+d;

      // compute I2 block start addresses
      I2_block_addr = I2_line_addr+16*u_warp;

      // compute match energy at this disparity
      xmm6 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_1));
      xmm6 = _mm_sad_epu8(xmm1,xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_2));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm2,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_3));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm3,xmm5),xmm6);
      xmm5 = _mm_load_si128((__m128i*)(I2_block_addr+desc_offset_4));
      xmm6 = _mm_add_epi16(_mm_sad_epu8(xmm4,xmm5),xmm6);
      sum  = _mm_extract_epi16(xmm6,0)+_mm_extract_epi16(xmm6,4);
      for(int i=0;i<16;i++){
        a5[i] = *(I2_block_addr+desc_offset_1+i);
        cost+=abs(a1[i]-a5[i]);
      }
      for(int i=0;i<16;i++){
        a5[i] = *(I2_block_addr+desc_offset_2+i);
        cost+=abs(a2[i]-a5[i]);
      }
      for(int i=0;i<16;i++){
        a5[i] = *(I2_block_addr+desc_offset_3+i);
        cost+=abs(a3[i]-a5[i]);
      }
      for(int i=0;i<16;i++){
        a5[i] = *(I2_block_addr+desc_offset_4+i);
        cost+=abs(a4[i]-a5[i]);
      }
      // sum=cost;
      // best + second best match
      if (sum<min_1_E) {
        min_2_E = min_1_E;   
        min_2_d = min_1_d;
        min_1_E = sum;
        min_1_d = d;
      } else if (sum<min_2_E) {
        min_2_E = sum;
        min_2_d = d;
      }
    }

    // check if best and second best match are available and if matching ratio is sufficient
    if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
      return min_1_d;
    else
      return -1;
    
  } else
    return -1;
}
__global__ void computeSupportMatchesKernel(int16_t* D_can,parameters param,uint8_t* I1_desc,uint8_t* I2_desc,int32_t D_can_width,int32_t D_can_height,int32_t width,int32_t height){
    int32_t D_candidate_stepsize = param.candidate_stepsize;
    if (param.subsampling)
    D_candidate_stepsize += D_candidate_stepsize%2;
    int16_t ret1,ret2;
    
    
    int32_t u,v;
    int32_t u_can,v_can;
    int numThreads=blockDim.x*BLOCK_X*BLOCK_X;
    int bound=(D_can_width-1)*(D_can_height-1);
    for(int idx=blockIdx.x * BLOCK_X*BLOCK_X + threadIdx.x*BLOCK_X+threadIdx.y;idx<bound;idx+=numThreads){
      v_can=idx/(D_can_width-1)+1;
      u_can=idx%(D_can_width-1)+1;
      
      u = u_can*D_candidate_stepsize;
        v = v_can*D_candidate_stepsize;
        // initialize disparity candidate to invalid
        *(D_can+v_can*D_can_width+u_can) = -1;
          
        // find forwards
        // d = computeMatchingDisparity(u,v,I1_desc,I2_desc,false);
        {
    
          bool right_image=false;
          const int32_t u_step      = 2;
          const int32_t v_step      = 2;
          const int32_t window_size = 3;
          
          int32_t desc_offset_1 = -16*u_step-16*width*v_step;
          int32_t desc_offset_2 = +16*u_step-16*width*v_step;
          int32_t desc_offset_3 = -16*u_step+16*width*v_step;
          int32_t desc_offset_4 = +16*u_step+16*width*v_step;
          
        
          // check if we are inside the image region
          if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
            
            // compute desc and start addresses
            int32_t  line_offset = 16*width*v;
            uint8_t *I1_line_addr,*I2_line_addr;
            if (!right_image) {
              I1_line_addr = I1_desc+line_offset;
              I2_line_addr = I2_desc+line_offset;
            } else {
              I1_line_addr = I2_desc+line_offset;
              I2_line_addr = I1_desc+line_offset;
            }
        
            // compute I1 block start addresses
            uint8_t* I1_block_addr = I1_line_addr+16*u;
            uint8_t* I2_block_addr;
            
            // we require at least some texture
            int32_t sum = 0;
            for (int32_t i=0; i<16; i++)
              sum += abs((int32_t)(*(I1_block_addr+i))-128);
            if (sum<param.support_texture)
              ret1=-1;
            else
            {
               
              
              // declare match energy for each disparity
              int32_t u_warp;
              
              // best match
              int16_t min_1_E = 32767;
              int16_t min_1_d = -1;
              int16_t min_2_E = 32767;
              int16_t min_2_d = -1;
          
              // get valid disparity range
              int32_t disp_min_valid = max(param.disp_min,0);
              int32_t disp_max_valid = param.disp_max;
              if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);
              else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);
              
              // assume, that we can compute at least 10 disparities for this pixel
              if (disp_max_valid-disp_min_valid<10)
               ret1=-1;
              else
              {
                  // for all disparities do
              for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {
          
                // warp u coordinate
                if (!right_image) u_warp = u-d;
                else              u_warp = u+d;
          
                // compute I2 block start addresses
                I2_block_addr = I2_line_addr+16*u_warp;
                int32_t cost=0;
                // compute match energy at this disparity
                
                //////
                for(int i=0;i<16;i++){
                  uint8_t b1=*(I1_block_addr+desc_offset_1+i);
                  uint8_t b2=*(I1_block_addr+desc_offset_2+i);
                  uint8_t b3=*(I1_block_addr+desc_offset_3+i);
                  uint8_t b4=*(I1_block_addr+desc_offset_4+i);
                  uint8_t c1=*(I2_block_addr+desc_offset_1+i);
                  uint8_t c2=*(I2_block_addr+desc_offset_2+i);
                  uint8_t c3=*(I2_block_addr+desc_offset_3+i);
                  uint8_t c4=*(I2_block_addr+desc_offset_4+i);
                  cost+=abs(b1-c1);
                  cost+=abs(b2-c2);
                  cost+=abs(b3-c3);
                  cost+=abs(b4-c4);
                }
                ////
                sum=cost;
                // best + second best match
                if (sum<min_1_E) {
                  min_2_E = min_1_E;   
                  min_2_d = min_1_d;
                  min_1_E = sum;
                  min_1_d = d;
                } else if (sum<min_2_E) {
                  min_2_E = sum;
                  min_2_d = d;
                }
            }
        
            // check if best and second best match are available and if matching ratio is sufficient
            if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
              ret1= min_1_d;
            else
              ret1= -1;  
              
            }
              }
          
              
            
          
            
          } else
            ret1= -1;
    
    
    
    
        }
        int16_t d=ret1;
       
        // for all disparities do
        
        
        
        if (d>=0) {
          
          // find backwards
          // d2 = computeMatchingDisparity(u-d,v,I1_desc,I2_desc,true);
         u-=d;
         {
    
          bool right_image=true;
          const int32_t u_step      = 2;
          const int32_t v_step      = 2;
          const int32_t window_size = 3;
          
          int32_t desc_offset_1 = -16*u_step-16*width*v_step;
          int32_t desc_offset_2 = +16*u_step-16*width*v_step;
          int32_t desc_offset_3 = -16*u_step+16*width*v_step;
          int32_t desc_offset_4 = +16*u_step+16*width*v_step;
          
        
          // check if we are inside the image region
          if (u>=window_size+u_step && u<=width-window_size-1-u_step && v>=window_size+v_step && v<=height-window_size-1-v_step) {
            
            // compute desc and start addresses
            int32_t  line_offset = 16*width*v;
            uint8_t *I1_line_addr,*I2_line_addr;
            if (!right_image) {
              I1_line_addr = I1_desc+line_offset;
              I2_line_addr = I2_desc+line_offset;
            } else {
              I1_line_addr = I2_desc+line_offset;
              I2_line_addr = I1_desc+line_offset;
            }
        
            // compute I1 block start addresses
            uint8_t* I1_block_addr = I1_line_addr+16*u;
            uint8_t* I2_block_addr;
            
            // we require at least some texture
            int32_t sum = 0;
            for (int32_t i=0; i<16; i++)
              sum += abs((int32_t)(*(I1_block_addr+i))-128);
            if (sum<param.support_texture)
              ret2=-1;
            else
            {
                
              
              // declare match energy for each disparity
              int32_t u_warp;
              
              // best match
              int16_t min_1_E = 32767;
              int16_t min_1_d = -1;
              int16_t min_2_E = 32767;
              int16_t min_2_d = -1;
          
              // get valid disparity range
              int32_t disp_min_valid = max(param.disp_min,0);
              int32_t disp_max_valid = param.disp_max;
              if (!right_image) disp_max_valid = min(param.disp_max,u-window_size-u_step);
              else              disp_max_valid = min(param.disp_max,width-u-window_size-u_step);
              
              // assume, that we can compute at least 10 disparities for this pixel
              if (disp_max_valid-disp_min_valid<10)
              ret2=-1;
              else
              {
                  // for all disparities do
              for (int16_t d=disp_min_valid; d<=disp_max_valid; d++) {
          
                // warp u coordinate
                if (!right_image) u_warp = u-d;
                else              u_warp = u+d;
                int32_t cost=0;
    
                // compute I2 block start addresses
                I2_block_addr = I2_line_addr+16*u_warp;
          
                // compute match energy at this disparity
                
                //////
                for(int i=0;i<16;i++){
                  uint8_t b1=*(I1_block_addr+desc_offset_1+i);
                  uint8_t b2=*(I1_block_addr+desc_offset_2+i);
                  uint8_t b3=*(I1_block_addr+desc_offset_3+i);
                  uint8_t b4=*(I1_block_addr+desc_offset_4+i);
                  uint8_t c1=*(I2_block_addr+desc_offset_1+i);
                  uint8_t c2=*(I2_block_addr+desc_offset_2+i);
                  uint8_t c3=*(I2_block_addr+desc_offset_3+i);
                  uint8_t c4=*(I2_block_addr+desc_offset_4+i);
                  cost+=abs(b1-c1);
                  cost+=abs(b2-c2);
                  cost+=abs(b3-c3);
                  cost+=abs(b4-c4);
                }
                ////
                sum=cost;
                // best + second best match
                if (sum<min_1_E) {
                  min_2_E = min_1_E;   
                  min_2_d = min_1_d;
                  min_1_E = sum;
                  min_1_d = d;
                } else if (sum<min_2_E) {
                  min_2_E = sum;
                  min_2_d = d;
                }
            }
        
            // check if best and second best match are available and if matching ratio is sufficient
            if (min_1_d>=0 && min_2_d>=0 && (float)min_1_E<param.support_threshold*(float)min_2_E)
            ret2= min_1_d;
            else
            ret2= -1;  
              
            }
              }
          
              
            
          
            
          } else
          ret2= -1;
    
    
    
    
        }
    
    
        int16_t d2=ret2;
          u+=d;
          if (d2>=0 && abs(d-d2)<=param.lr_threshold)
            *(D_can+v_can*D_can_width+u_can) = d;
        }
    }
    // for(u_can=threadIdx.x+1;u_can<D_can_width;u_can+=BLOCK_X)
    //   for(v_can=threadIdx.y+1;v_can<D_can_height;v_can+=BLOCK_X){
        
    //   }
    
}

vector<support_pt> Elas::computeSupportMatches (uint8_t* I1_desc,uint8_t* I2_desc) {
  cudaError_t err ;
  // be sure that at half resolution we only need data
  // from every second line!
  int32_t D_candidate_stepsize = param.candidate_stepsize;
  if (param.subsampling)
    D_candidate_stepsize += D_candidate_stepsize%2;

  // create matrix for saving disparity candidates
  int32_t D_can_width  = 0;
  int32_t D_can_height = 0;
  for (int32_t u=0; u<width;  u+=D_candidate_stepsize) D_can_width++;
  for (int32_t v=0; v<height; v+=D_candidate_stepsize) D_can_height++;
  int16_t* D_can = (int16_t*)calloc(D_can_width*D_can_height,sizeof(int16_t));


  int16_t* d_D_can;

  err=cudaMalloc(&d_D_can,D_can_width*D_can_height*sizeof(int16_t));
  if(err!=cudaSuccess) cout<<"Couldnt't allocate D_can \n";
  
  dim3 dimBlock(32);
  dim3 dimGrid(BLOCK_X,BLOCK_X);
  size_t size;

  uint8_t* d_I1_desc;
  uint8_t* d_I2_desc;
  size=16*width*height*sizeof(uint8_t);
  err=cudaMalloc(&d_I1_desc,size);
  if(err!=cudaSuccess) cout<<"error";
  err=cudaMalloc(&d_I2_desc,size);
  if(err!=cudaSuccess) cout<<"error";
  err=cudaMemcpy(d_I1_desc,I1_desc,size,cudaMemcpyHostToDevice);
  err=cudaMemcpy(d_I2_desc,I2_desc,size,cudaMemcpyHostToDevice);
  if(err!=cudaSuccess) cout<<"error";

  computeSupportMatchesKernel<<<dimBlock,dimGrid>>>(d_D_can,param,d_I1_desc,d_I2_desc,D_can_width,D_can_height,width,height);
  err = cudaGetLastError();
  if(err!=cudaSuccess) cout<<"error";
  cudaMemcpy(D_can,d_D_can,D_can_width*D_can_height*sizeof(int16_t),cudaMemcpyDeviceToHost);
  
  cudaFree(d_D_can);
  cudaFree(d_I1_desc);
  cudaFree(d_I2_desc);
  cudaDeviceSynchronize();


  // ofstream file;
  // file.open ("disparity_parallel.txt");
  // for(int i=0;i<D_can_height;i++)
  // for(int j=0;j<D_can_width;j++)
  // file<<D_can[i*D_can_width+j]<<"\n";
  // file.close();
  // remove inconsistent support points
  removeInconsistentSupportPoints(D_can,D_can_width,D_can_height);
  
  // remove support points on straight lines, since they are redundant
  // this reduces the number of triangles a little bit and hence speeds up
  // the triangulation process
  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,true);
  removeRedundantSupportPoints(D_can,D_can_width,D_can_height,5,1,false);
  
  // move support points from image representation into a vector representation
  vector<support_pt> p_support;
  for (int32_t u_can=1; u_can<D_can_width; u_can++)
    for (int32_t v_can=1; v_can<D_can_height; v_can++)
      if (*(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))>=0)
        p_support.push_back(support_pt(u_can*D_candidate_stepsize,
                                       v_can*D_candidate_stepsize,
                                       *(D_can+getAddressOffsetImage(u_can,v_can,D_can_width))));
  
  // if flag is set, add support points in image corners
  // with the same disparity as the nearest neighbor support point
  if (param.add_corners)
    addCornerSupportPoints(p_support);

  // free memory
  free(D_can);
  
  // return support point vector
  return p_support; 
}

vector<triangle> Elas::computeDelaunayTriangulation (vector<support_pt> p_support,int32_t right_image) {

  // input/output structure for triangulation
  struct triangulateio in, out;
  int32_t k;

  // inputs
  in.numberofpoints = p_support.size();
  in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
  k=0;
  if (!right_image) {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u;
      in.pointlist[k++] = p_support[i].v;
    }
  } else {
    for (int32_t i=0; i<p_support.size(); i++) {
      in.pointlist[k++] = p_support[i].u-p_support[i].d;
      in.pointlist[k++] = p_support[i].v;
    }
  }
  in.numberofpointattributes = 0;
  in.pointattributelist      = NULL;
  in.pointmarkerlist         = NULL;
  in.numberofsegments        = 0;
  in.numberofholes           = 0;
  in.numberofregions         = 0;
  in.regionlist              = NULL;
  
  // outputs
  out.pointlist              = NULL;
  out.pointattributelist     = NULL;
  out.pointmarkerlist        = NULL;
  out.trianglelist           = NULL;
  out.triangleattributelist  = NULL;
  out.neighborlist           = NULL;
  out.segmentlist            = NULL;
  out.segmentmarkerlist      = NULL;
  out.edgelist               = NULL;
  out.edgemarkerlist         = NULL;

  // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
  char parameters[] = "zQB";
  triangulate(parameters, &in, &out, NULL);
  
  // put resulting triangles into vector tri
  vector<triangle> tri;
  k=0;
  for (int32_t i=0; i<out.numberoftriangles; i++) {
    tri.push_back(triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]));
    k+=3;
  }
  
  // free memory used for triangulation
  free(in.pointlist);
  free(out.pointlist);
  free(out.trianglelist);
  
  // return triangles
  return tri;
}

void Elas::computeDisparityPlanes (vector<support_pt> p_support,vector<triangle> &tri,int32_t right_image) {

  // init matrices
  Matrix A(3,3);
  Matrix b(3,1);
  
  // for all triangles do
  for (int32_t i=0; i<tri.size(); i++) {
    
    // get triangle corner indices
    int32_t c1 = tri[i].c1;
    int32_t c2 = tri[i].c2;
    int32_t c3 = tri[i].c3;
    
    // compute matrix A for linear system of left triangle
    A.val[0][0] = p_support[c1].u;
    A.val[1][0] = p_support[c2].u;
    A.val[2][0] = p_support[c3].u;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;
    
    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;
    
    // on success of gauss jordan elimination
    if (b.solve(A)) {
      
      // grab results from b
      tri[i].t1a = b.val[0][0];
      tri[i].t1b = b.val[1][0];
      tri[i].t1c = b.val[2][0];
      
    // otherwise: invalid
    } else {
      tri[i].t1a = 0;
      tri[i].t1b = 0;
      tri[i].t1c = 0;
    }

    // compute matrix A for linear system of right triangle
    A.val[0][0] = p_support[c1].u-p_support[c1].d;
    A.val[1][0] = p_support[c2].u-p_support[c2].d;
    A.val[2][0] = p_support[c3].u-p_support[c3].d;
    A.val[0][1] = p_support[c1].v; A.val[0][2] = 1;
    A.val[1][1] = p_support[c2].v; A.val[1][2] = 1;
    A.val[2][1] = p_support[c3].v; A.val[2][2] = 1;
    
    // compute vector b for linear system (containing the disparities)
    b.val[0][0] = p_support[c1].d;
    b.val[1][0] = p_support[c2].d;
    b.val[2][0] = p_support[c3].d;
    
    // on success of gauss jordan elimination
    if (b.solve(A)) {
      
      // grab results from b
      tri[i].t2a = b.val[0][0];
      tri[i].t2b = b.val[1][0];
      tri[i].t2c = b.val[2][0];
      
    // otherwise: invalid
    } else {
      tri[i].t2a = 0;
      tri[i].t2b = 0;
      tri[i].t2c = 0;
    }
  }  
}

void Elas::createGrid(vector<support_pt> p_support,int32_t* disparity_grid,int32_t* grid_dims,bool right_image) {
  
  // get grid dimensions
  int32_t grid_width  = grid_dims[1];
  int32_t grid_height = grid_dims[2];
  
  // allocate temporary memory
  int32_t* temp1 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
  int32_t* temp2 = (int32_t*)calloc((param.disp_max+1)*grid_height*grid_width,sizeof(int32_t));
  
  // for all support points do
  for (int32_t i=0; i<p_support.size(); i++) {
    
    // compute disparity range to fill for this support point
    int32_t x_curr = p_support[i].u;
    int32_t y_curr = p_support[i].v;
    int32_t d_curr = p_support[i].d;
    int32_t d_min  = max(d_curr-1,0);
    int32_t d_max  = min(d_curr+1,param.disp_max);
    
    // fill disparity grid helper
    for (int32_t d=d_min; d<=d_max; d++) {
      int32_t x;
      if (!right_image)
        x = floor((float)(x_curr/param.grid_size));
      else
        x = floor((float)(x_curr-d_curr)/(float)param.grid_size);
      int32_t y = floor((float)y_curr/(float)param.grid_size);
      
      // point may potentially lay outside (corner points)
      if (x>=0 && x<grid_width &&y>=0 && y<grid_height) {
        int32_t addr = getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1);
        *(temp1+addr) = 1;
      }
    }
  }
  
  // diffusion pointers
  const int32_t* tl = temp1 + (0*grid_width+0)*(param.disp_max+1);
  const int32_t* tc = temp1 + (0*grid_width+1)*(param.disp_max+1);
  const int32_t* tr = temp1 + (0*grid_width+2)*(param.disp_max+1);
  const int32_t* cl = temp1 + (1*grid_width+0)*(param.disp_max+1);
  const int32_t* cc = temp1 + (1*grid_width+1)*(param.disp_max+1);
  const int32_t* cr = temp1 + (1*grid_width+2)*(param.disp_max+1);
  const int32_t* bl = temp1 + (2*grid_width+0)*(param.disp_max+1);
  const int32_t* bc = temp1 + (2*grid_width+1)*(param.disp_max+1);
  const int32_t* br = temp1 + (2*grid_width+2)*(param.disp_max+1);
  
  int32_t* result    = temp2 + (1*grid_width+1)*(param.disp_max+1); 
  int32_t* end_input = temp1 + grid_width*grid_height*(param.disp_max+1);
  
  // diffuse temporary grid
  for( ; br != end_input; tl++, tc++, tr++, cl++, cc++, cr++, bl++, bc++, br++, result++ )
    *result = *tl | *tc | *tr | *cl | *cc | *cr | *bl | *bc | *br;
  
  // for all grid positions create disparity grid
  for (int32_t x=0; x<grid_width; x++) {
    for (int32_t y=0; y<grid_height; y++) {
        
      // start with second value (first is reserved for count)
      int32_t curr_ind = 1;
      
      // for all disparities do
      for (int32_t d=0; d<=param.disp_max; d++) {

        // if yes => add this disparity to current cell
        if (*(temp2+getAddressOffsetGrid(x,y,d,grid_width,param.disp_max+1))>0) {
          *(disparity_grid+getAddressOffsetGrid(x,y,curr_ind,grid_width,param.disp_max+2))=d;
          curr_ind++;
        }
      }
      
      // finally set number of indices
      *(disparity_grid+getAddressOffsetGrid(x,y,0,grid_width,param.disp_max+2))=curr_ind-1;
    }
  }
  
  // release temporary memory
  free(temp1);
  free(temp2);
}

__device__ void updatePosteriorMinimum(uint8_t* I2_block_addr,const int32_t &d,const int32_t &w,
  uint8_t* I1_block_addr,int32_t &min_val,int32_t &min_d) {
 int32_t cost=w;
  
  for(int i=0;i<16;i++){
    uint8_t xmm2=*(I2_block_addr+i);
    uint8_t xmm1=*(I1_block_addr+i);
    cost+=abs(xmm2-xmm1);
  }
  
if (cost<min_val) {
    min_val = cost;
    min_d   = d;
  }
}



 __device__ void findMatch(int32_t &u,int32_t &v,float &plane_a,float &plane_b,float &plane_c,
                            int32_t* disparity_grid,int32_t *grid_dims,uint8_t* I1_desc,uint8_t* I2_desc,
                            int32_t *P,int32_t &plane_radius,bool &valid,bool &right_image,float* D,
                            int32_t width,int32_t height,parameters param){
  // get image width and height
  const int32_t disp_num    = grid_dims[0]-1;
  const int32_t window_size = 2;

  // address of disparity we want to compute
  uint32_t d_addr;
  if (param.subsampling) d_addr = getAddressOffsetImage_device(u/2,v/2,width/2);
  else                   d_addr = getAddressOffsetImage_device(u,v,width);

  // check if u is ok
  if (u<window_size || u>=width-window_size)
    return;

  // compute line start address
  int32_t  line_offset = 16*width*max(min(v,height-3),2);
  uint8_t *I1_line_addr,*I2_line_addr;
  if (!right_image) {
    I1_line_addr = I1_desc+line_offset;
    I2_line_addr = I2_desc+line_offset;
  } else {
    I1_line_addr = I2_desc+line_offset;
    I2_line_addr = I1_desc+line_offset;
  }

  // compute I1 block start address
  uint8_t* I1_block_addr = I1_line_addr+16*u;
    // does this patch have enough texture?
  int32_t sum = 0;
  for (int32_t i=0; i<16; i++)
    sum += abs((int32_t)(*(I1_block_addr+i))-128);
  if (sum<param.match_texture)
    return;

  // compute disparity, min disparity and max disparity of plane prior
  int32_t d_plane     = (int32_t)(plane_a*(float)u+plane_b*(float)v+plane_c);
  int32_t d_plane_min = max(d_plane-plane_radius,0);
  int32_t d_plane_max = min(d_plane+plane_radius,disp_num-1);

  // get grid pointer
  int32_t  grid_x    = (int32_t)floor((float)u/(float)param.grid_size);
  int32_t  grid_y    = (int32_t)floor((float)v/(float)param.grid_size);
  uint32_t grid_addr = getAddressOffsetGrid_device(grid_x,grid_y,0,grid_dims[1],grid_dims[0]);  
  int32_t  num_grid  = *(disparity_grid+grid_addr);
  int32_t* d_grid    = disparity_grid+grid_addr+1;
  
  // loop variables
  int32_t d_curr, u_warp;
  int32_t min_val = 10000;

  int32_t min_d   = -1;

  // left image
  if (!right_image) { 
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u-d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((uint8_t*)(I2_line_addr+16*u_warp),d_curr,0,I1_block_addr,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u-d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((uint8_t*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,I1_block_addr,min_val,min_d);
    }
    
  // right image
  } else {
    for (int32_t i=0; i<num_grid; i++) {
      d_curr = d_grid[i];
      if (d_curr<d_plane_min || d_curr>d_plane_max) {
        u_warp = u+d_curr;
        if (u_warp<window_size || u_warp>=width-window_size)
          continue;
        updatePosteriorMinimum((uint8_t*)(I2_line_addr+16*u_warp),d_curr,0,I1_block_addr,min_val,min_d);
      }
    }
    for (d_curr=d_plane_min; d_curr<=d_plane_max; d_curr++) {
      u_warp = u+d_curr;
      if (u_warp<window_size || u_warp>=width-window_size)
        continue;
      updatePosteriorMinimum((uint8_t*)(I2_line_addr+16*u_warp),d_curr,valid?*(P+abs(d_curr-d_plane)):0,I1_block_addr,min_val,min_d);
    }
  }

  // set disparity value
  if (min_d>=0) *(D+d_addr) = min_d; // MAP value (min neg-Log probability)
  else          *(D+d_addr) = -1;    // invalid disparity
}

__global__ void computeDisparityKernel(bool right_image,bool subsampling,support_pt * p_support,triangle * tri,int32_t* disparity_grid,  int32_t *grid_dims,
  uint8_t* I1_desc,uint8_t* I2_desc,int32_t* P,int32_t plane_radius,float* D,int32_t width,int32_t height,parameters param,int tri_size){
  uint32_t i =blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=tri_size) 
    return;
  int32_t c1, c2, c3;
  float plane_a,plane_b,plane_c,plane_d;
  // get plane parameters
  uint32_t p_i = i*3;
  if (!right_image) {
    plane_a = tri[i].t1a;
    plane_b = tri[i].t1b;
    plane_c = tri[i].t1c;
    plane_d = tri[i].t2a;
  } else {
    plane_a = tri[i].t2a;
    plane_b = tri[i].t2b;
    plane_c = tri[i].t2c;
    plane_d = tri[i].t1a;
  }
  
  
  // triangle corners
  c1 = tri[i].c1;
  c2 = tri[i].c2;
  c3 = tri[i].c3;

  // sort triangle corners wrt. u (ascending)    
  float tri_u[3];
  if (!right_image) {
    tri_u[0] = p_support[c1].u;
    tri_u[1] = p_support[c2].u;
    tri_u[2] = p_support[c3].u;
  } else {
    tri_u[0] = p_support[c1].u-p_support[c1].d;
    tri_u[1] = p_support[c2].u-p_support[c2].d;
    tri_u[2] = p_support[c3].u-p_support[c3].d;
  }
  float tri_v[3] = {p_support[c1].v,p_support[c2].v,p_support[c3].v};
  
  for (uint32_t j=0; j<3; j++) {
    for (uint32_t k=0; k<j; k++) {
      if (tri_u[k]>tri_u[j]) {
        float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
        float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
      }
    }
  }
  
  // rename corners
  float A_u = tri_u[0]; float A_v = tri_v[0];
  float B_u = tri_u[1]; float B_v = tri_v[1];
  float C_u = tri_u[2]; float C_v = tri_v[2];

  // compute straight lines connecting triangle corners
  float AB_a = 0; float AC_a = 0; float BC_a = 0;
  if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
  if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
  if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
  float AB_b = A_v-AB_a*A_u;
  float AC_b = A_v-AC_a*A_u;
  float BC_b = B_v-BC_a*B_u;
  
  // a plane is only valid if itself and its projection
  // into the other image is not too much slanted
  bool valid = fabs(plane_a)<0.7 && fabs(plane_d)<0.7;
      
  // first part (triangle corner A->B)
  if ((int32_t)(A_u)!=(int32_t)(B_u)) {
    for (int32_t u=max((int32_t)A_u,0); u<min((int32_t)B_u,width); u++){
      if (!subsampling || u%2==0) {
        int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
        int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
        for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
                 if (!subsampling || v%2==0) {
                   findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
                             I1_desc,I2_desc,P,plane_radius,valid,right_image,D,width,height,param);
                 }
        }
      }
    }
  

  // second part (triangle corner B->C)
  if ((int32_t)(B_u)!=(int32_t)(C_u)) {
    for (int32_t u=max((int32_t)B_u,0); u<min((int32_t)C_u,width); u++){
      if (!subsampling || u%2==0) {
        int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
        int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
        for (int32_t v=min(v_1,v_2); v<max(v_1,v_2); v++)
          if (!subsampling || v%2==0) {
            findMatch(u,v,plane_a,plane_b,plane_c,disparity_grid,grid_dims,
                      I1_desc,I2_desc,P,plane_radius,valid,right_image,D,width,height,param);
          }
      }
    }
  }
}
// TODO: %2 => more elegantly
void Elas::computeDisparity(vector<support_pt> p_support,vector<triangle> tri,int32_t* disparity_grid,int32_t *grid_dims,
                            uint8_t* I1_desc,uint8_t* I2_desc,bool right_image,float* D) {

 
  // number of disparities
  const int32_t disp_num  = grid_dims[0]-1;
  
  // descriptor window_size
  int32_t window_size = 2;
  
  // init disparity image to -10
  if (param.subsampling) {
    for (int32_t i=0; i<(width/2)*(height/2); i++)
      *(D+i) = -10;
  } else {
    for (int32_t i=0; i<width*height; i++)
      *(D+i) = -10;
  }
  
  // pre-compute prior 
  float two_sigma_squared = 2*param.sigma*param.sigma;
  int32_t* P = new int32_t[disp_num];
  for (int32_t delta_d=0; delta_d<disp_num; delta_d++)
    P[delta_d] = (int32_t)((-log(param.gamma+exp(-delta_d*delta_d/two_sigma_squared))+log(param.gamma))/param.beta);
  int32_t plane_radius = (int32_t)max((float)ceil(param.sigma*param.sradius),(float)2.0);
  
  size_t size;
  size=p_support.size()*sizeof(p_support);
  support_pt* d_p_support,*p_support_tmp;
  p_support_tmp = &p_support[0];
  cudaMalloc(&d_p_support,size);
  cudaMemcpy(d_p_support,p_support_tmp,size,cudaMemcpyHostToDevice);
  
  size=tri.size()*sizeof(triangle);
  triangle* d_tri,*tri_tmp;
  tri_tmp = &tri[0];
  cudaMalloc(&d_tri,size);
  cudaMemcpy(d_tri,tri_tmp,size,cudaMemcpyHostToDevice);
  
  int32_t grid_width   = (int32_t)ceil((float)width/(float)param.grid_size);
  int32_t grid_height  = (int32_t)ceil((float)height/(float)param.grid_size);
  size=(param.disp_max+2)*grid_height*grid_width*sizeof(int32_t);
  int32_t* d_disparity_grid;
  cudaMalloc(&d_disparity_grid,size);
  cudaMemcpy(d_disparity_grid,disparity_grid,size,cudaMemcpyHostToDevice);
  
  size=3*sizeof(int32_t);
  int32_t *d_grid_dims;
  cudaMalloc(&d_grid_dims,size);
  cudaMemcpy(d_grid_dims,grid_dims,size,cudaMemcpyHostToDevice);
  
  size=16*width*height*sizeof(uint8_t);
  uint8_t *d_I1_desc,* d_I2_desc;
  cudaMalloc(&d_I1_desc,size);
  cudaMalloc(&d_I2_desc,size);
  cudaMemcpy(d_I1_desc,I1_desc,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_I2_desc,I2_desc,size,cudaMemcpyHostToDevice);
  
  size=disp_num*sizeof(int32_t);
  int32_t* d_P;
  cudaMalloc(&d_P,size);
  cudaMemcpy(d_P,P,size,cudaMemcpyHostToDevice);
  
  size=width*height*sizeof(float);
  float* d_D;
  cudaMalloc(&d_D,size);
  cudaMemcpy(d_D,D,size,cudaMemcpyHostToDevice);

  // for all triangles do
  int numBlocks=(tri.size()+31)/32;
  computeDisparityKernel<<<numBlocks,32>>>(right_image,param.subsampling,d_p_support,d_tri,d_disparity_grid,d_grid_dims,d_I1_desc,d_I2_desc,d_P,plane_radius,d_D,width,height,param,tri.size());
  cudaError_t err = cudaGetLastError();
  if(err!=cudaSuccess) cout<<"error";

  size=width*height*sizeof(float);
  cudaMemcpy(D,d_D,size,cudaMemcpyDeviceToHost);
  
  cudaFree(d_p_support);
  cudaFree(d_tri);
  cudaFree(d_disparity_grid);
  cudaFree(d_grid_dims);
  cudaFree(d_I1_desc);
  cudaFree(d_I2_desc);
  cudaFree(d_P);
  cudaFree(d_D);
  cudaDeviceSynchronize();

  delete[] P;
  // ofstream file;
  // file.open ("computedisparity_parallel.txt");
  // for(int i=0;i<height;i++)
  // for(int j=0;j<width;j++)
  // file<<D[i*width+j]<<"\n";

  // file.close();
  
}

void Elas::leftRightConsistencyCheck(float* D1,float* D2) {
  
  // get disparity image dimensions
  int32_t D_width  = width;
  int32_t D_height = height;
  if (param.subsampling) {
    D_width  = width/2;
    D_height = height/2;
  }
  
  // make a copy of both images
  float* D1_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D2_copy = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D1_copy,D1,D_width*D_height*sizeof(float));
  memcpy(D2_copy,D2,D_width*D_height*sizeof(float));

  // loop variables
  uint32_t addr,addr_warp;
  float    u_warp_1,u_warp_2,d1,d2;
  
  // for all image points do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {
      
      // compute address (u,v) and disparity value
      addr     = getAddressOffsetImage(u,v,D_width);
      d1       = *(D1_copy+addr);
      d2       = *(D2_copy+addr);
      if (param.subsampling) {
        u_warp_1 = (float)u-d1/2;
        u_warp_2 = (float)u+d2/2;
      } else {
        u_warp_1 = (float)u-d1;
        u_warp_2 = (float)u+d2;
      }
      
      
      // check if left disparity is valid
      if (d1>=0 && u_warp_1>=0 && u_warp_1<D_width) {       
                  
        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_1,v,D_width);

        // if check failed
        if (fabs(*(D2_copy+addr_warp)-d1)>param.lr_threshold)
          *(D1+addr) = -10;
        
      // set invalid
      } else
        *(D1+addr) = -10;
      
      // check if right disparity is valid
      if (d2>=0 && u_warp_2>=0 && u_warp_2<D_width) {       

        // compute warped image address
        addr_warp = getAddressOffsetImage((int32_t)u_warp_2,v,D_width);

        // if check failed
        if (fabs(*(D1_copy+addr_warp)-d2)>param.lr_threshold)
          *(D2+addr) = -10;
        
      // set invalid
      } else
        *(D2+addr) = -10;
    }
  }
  
  // release memory
  free(D1_copy);
  free(D2_copy);
}

void Elas::removeSmallSegments (float* D) {
  
  // get disparity image dimensions
  int32_t D_width        = width;
  int32_t D_height       = height;
  int32_t D_speckle_size = param.speckle_size;
  if (param.subsampling) {
    D_width        = width/2;
    D_height       = height/2;
    D_speckle_size = sqrt((float)param.speckle_size)*2;
  }
  
  // allocate memory on heap for dynamic programming arrays
  int32_t *D_done     = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_u = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t *seg_list_v = (int32_t*)calloc(D_width*D_height,sizeof(int32_t));
  int32_t seg_list_count;
  int32_t seg_list_curr;
  int32_t u_neighbor[4];
  int32_t v_neighbor[4];
  int32_t u_seg_curr;
  int32_t v_seg_curr;
  
  // declare loop variables
  int32_t addr_start, addr_curr, addr_neighbor;
  
  // for all pixels do
  for (int32_t u=0; u<D_width; u++) {
    for (int32_t v=0; v<D_height; v++) {
      
      // get address of first pixel in this segment
      addr_start = getAddressOffsetImage(u,v,D_width);
                  
      // if this pixel has not already been processed
      if (*(D_done+addr_start)==0) {
                
        // init segment list (add first element
        // and set it to be the next element to check)
        *(seg_list_u+0) = u;
        *(seg_list_v+0) = v;
        seg_list_count  = 1;
        seg_list_curr   = 0;
        
        // add neighboring segments as long as there
        // are none-processed pixels in the seg_list;
        // none-processed means: seg_list_curr<seg_list_count
        while (seg_list_curr<seg_list_count) {
        
          // get current position from seg_list
          u_seg_curr = *(seg_list_u+seg_list_curr);
          v_seg_curr = *(seg_list_v+seg_list_curr);
          
          // get address of current pixel in this segment
          addr_curr = getAddressOffsetImage(u_seg_curr,v_seg_curr,D_width);
          
          // fill list with neighbor positions
          u_neighbor[0] = u_seg_curr-1; v_neighbor[0] = v_seg_curr;
          u_neighbor[1] = u_seg_curr+1; v_neighbor[1] = v_seg_curr;
          u_neighbor[2] = u_seg_curr;   v_neighbor[2] = v_seg_curr-1;
          u_neighbor[3] = u_seg_curr;   v_neighbor[3] = v_seg_curr+1;
          
          // for all neighbors do
          for (int32_t i=0; i<4; i++) {
            
            // check if neighbor is inside image
            if (u_neighbor[i]>=0 && v_neighbor[i]>=0 && u_neighbor[i]<D_width && v_neighbor[i]<D_height) {
              
              // get neighbor pixel address
              addr_neighbor = getAddressOffsetImage(u_neighbor[i],v_neighbor[i],D_width);
              
              // check if neighbor has not been added yet and if it is valid
              if (*(D_done+addr_neighbor)==0 && *(D+addr_neighbor)>=0) {

                // is the neighbor similar to the current pixel
                // (=belonging to the current segment)
                if (fabs(*(D+addr_curr)-*(D+addr_neighbor))<=param.speckle_sim_threshold) {
                  
                  // add neighbor coordinates to segment list
                  *(seg_list_u+seg_list_count) = u_neighbor[i];
                  *(seg_list_v+seg_list_count) = v_neighbor[i];
                  seg_list_count++;            
                  
                  // set neighbor pixel in I_done to "done"
                  // (otherwise a pixel may be added 2 times to the list, as
                  //  neighbor of one pixel and as neighbor of another pixel)
                  *(D_done+addr_neighbor) = 1;
                }
              }
              
            } 
          }
          
          // set current pixel in seg_list to "done"
          seg_list_curr++;
          
          // set current pixel in I_done to "done"
          *(D_done+addr_curr) = 1;

        } // end: while (seg_list_curr<seg_list_count)
        
        // if segment NOT large enough => invalidate pixels
        if (seg_list_count<D_speckle_size) {
          
          // for all pixels in current segment invalidate pixels
          for (int32_t i=0; i<seg_list_count; i++) {
            addr_curr = getAddressOffsetImage(*(seg_list_u+i),*(seg_list_v+i),D_width);
            *(D+addr_curr) = -10;
          }
        }
      } // end: if (*(I_done+addr_start)==0)
      
    }
  }
  
  // free memory
  free(D_done);
  free(seg_list_u);
  free(seg_list_v);
}

void Elas::gapInterpolation(float* D) {
  
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  int32_t D_ipol_gap_width = param.ipol_gap_width;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
    D_ipol_gap_width = param.ipol_gap_width/2+1;
  }
  
  // discontinuity threshold
  float discon_threshold = 3.0;
  
  // declare loop variables
  int32_t count,addr,v_first,v_last,u_first,u_last;
  float   d1,d2,d_ipol;
  
  // 1. Row-wise:
  // for each row do
  for (int32_t v=0; v<D_height; v++) {
    
    // init counter
    count = 0;
    
    // for each element of the row do
    for (int32_t u=0; u<D_width; u++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);
      
      // if disparity valid
      if (*(D+addr)>=0) {
        
        // check if speckle is small enough
        if (count>=1 && count<=D_ipol_gap_width) {
          
          // first and last value for interpolation
          u_first = u-count;
          u_last  = u-1;
          
          // if value in range
          if (u_first>0 && u_last<D_width-1) {
            
            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u_first-1,v,D_width));
            d2 = *(D+getAddressOffsetImage(u_last+1,v,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);
            
            // set all values to d_ipol
            for (int32_t u_curr=u_first; u_curr<=u_last; u_curr++)
              *(D+getAddressOffsetImage(u_curr,v,D_width)) = d_ipol;
          }
          
        }
        
        // reset counter
        count = 0;
      
      // otherwise increment counter
      } else {
        count++;
      }
    }
    
    // if full size disp map requested
    if (param.add_corners) {

      // extrapolate to the left
      for (int32_t u=0; u<D_width; u++) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=max(u-D_ipol_gap_width,0); u2<u; u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }

      // extrapolate to the right
      for (int32_t u=D_width-1; u>=0; u--) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t u2=u; u2<=min(u+D_ipol_gap_width,D_width-1); u2++)
            *(D+getAddressOffsetImage(u2,v,D_width)) = *(D+addr);
          break;
        }
      }
    }
  }

  // 2. Column-wise:
  // for each column do
  for (int32_t u=0; u<D_width; u++) {
    
    // init counter
    count = 0;
    
    // for each element of the column do
    for (int32_t v=0; v<D_height; v++) {
      
      // get address of this location
      addr = getAddressOffsetImage(u,v,D_width);
      
      // if disparity valid
      if (*(D+addr)>=0) {
        
        // check if gap is small enough
        if (count>=1 && count<=D_ipol_gap_width) {
          
          // first and last value for interpolation
          v_first = v-count;
          v_last  = v-1;
          
          // if value in range
          if (v_first>0 && v_last<D_height-1) {
            
            // compute mean disparity
            d1 = *(D+getAddressOffsetImage(u,v_first-1,D_width));
            d2 = *(D+getAddressOffsetImage(u,v_last+1,D_width));
            if (fabs(d1-d2)<discon_threshold) d_ipol = (d1+d2)/2;
            else                              d_ipol = min(d1,d2);
            
            // set all values to d_ipol
            for (int32_t v_curr=v_first; v_curr<=v_last; v_curr++)
              *(D+getAddressOffsetImage(u,v_curr,D_width)) = d_ipol;
          }
          
        }
        
        // reset counter
        count = 0;
      
      // otherwise increment counter
      } else {
        count++;
      }
    }

    // added extrapolation to top and bottom since bottom rows sometimes stay unlabeled...
    // DS 5/12/2014

    // if full size disp map requested
    if (param.add_corners) {

      // extrapolate towards top
      for (int32_t v=0; v<D_height; v++) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t v2=max(v-D_ipol_gap_width,0); v2<v; v2++)
            *(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
          break;
        }
      }

      // extrapolate towards the bottom
      for (int32_t v=D_height-1; v>=0; v--) {

        // get address of this location
        addr = getAddressOffsetImage(u,v,D_width);

        // if disparity valid
        if (*(D+addr)>=0) {
          for (int32_t v2=v; v2<=min(v+D_ipol_gap_width,D_height-1); v2++)
            *(D+getAddressOffsetImage(u,v2,D_width)) = *(D+addr);
          break;
        }
      }
    }
  }
}
__device__ void bitwise_and(float *a ){
  float b=(((uint32_t)1)<<31)-1;
  
  for(int i=0;i<4;i++)
      for (size_t j = 0; j < sizeof(float); j++)
          reinterpret_cast<char*>(&a[i])[j]=
              reinterpret_cast<char*>(&a[i])[j]&
              reinterpret_cast<char*>(&b)[j];

}
__global__ void adaptiveMeanKernelHorizontal(float *D_copy,int32_t D_width,float *D_tmp,int32_t D_height ){
  int32_t v,u,u2;
  int idx=blockIdx.x*blockDim.x + threadIdx.x;
  v=idx/(D_width-7)+3;
  u=idx%(D_width-7)+7;
  if(v>=D_height-3 || u>=D_width)
  return ;

  float val [8],aweight1[4],aweight2[4],afactor1[4],afactor2[4],aval[4];
 
  

  u2=u%8;
  for(int i=0;i<8;i++){
    val[u2] = *(D_copy+v*D_width+u-i);
    u2--;
    if(u2<0)
    u2=7;

  }
  float val_curr = *(D_copy+v*D_width+(u-3));
  for(int i=0;i<4;i++)
    aweight1[i]=val[i]-val_curr;
        
  bitwise_and(aweight1);


  for(int i=0;i<4;i++){
      aweight1[i]=4-aweight1[i];
      if(aweight1[i]<0)
          aweight1[i]=0;
  }
      
  for(int i=0;i<4;i++)
    afactor1[i]=val[i]*aweight1[i];
    
  for(int i=0;i<4;i++)
    aval[i]=val[i+4]; 

  for(int i=0;i<4;i++){
    aweight2[i]=aval[i]-val_curr;
  }
  bitwise_and(aweight2);

    
  for(int i=0;i<4;i++){
    aweight2[i]=4-aweight2[i];
    if(aweight2[i]<0)
      aweight2[i]=0;
  }

  for(int i=0;i<4;i++)
    afactor2[i]=aval[i]*aweight2[i];
  for(int i=0;i<4;i++)
    aweight1[i]=aweight1[i]+aweight2[i];
  for(int i=0;i<4;i++)
    afactor1[i]=afactor1[i]+afactor2[i];
    
  float weight_sum = aweight1[0]+aweight1[1]+aweight1[2]+aweight1[3];
  float factor_sum = afactor1[0]+afactor1[1]+afactor1[2]+afactor1[3];
  if (weight_sum>0) {
    float d = factor_sum/weight_sum;
    if (d>=0) {
    *(D_tmp+v*D_width+(u-3)) = d;
      // file<<v<<" "<<u<<" "<<d<<"\n";
      // printf("%d %d %f\n",v,u,d);
    }
  }
      
}

__global__ void adaptiveMeanKernelVertical(float *D_tmp,int32_t D_width,float *D,int32_t D_height ){
  int32_t v,u,v2;
  int idx=blockIdx.x*blockDim.x + threadIdx.x;
  u=idx/(D_height-7)+3;
  v=idx%(D_height-7)+7;
  if(v>=D_height || u>=D_width-3)
  return ;

  float val [8],aweight1[4],aweight2[4],afactor1[4],afactor2[4],aval[4];
 
  

  v2=v%8;
  for(int i=0;i<8;i++){
    val[v2] = *(D_tmp+(v-i)*D_width+u);
    v2--;
    if(v2<0)
    v2=7;

  }
  float val_curr = *(D_tmp+(v-3)*D_width+u);
  for(int i=0;i<4;i++)
    aweight1[i]=val[i]-val_curr;
        
  bitwise_and(aweight1);


  for(int i=0;i<4;i++){
      aweight1[i]=4-aweight1[i];
      if(aweight1[i]<0)
          aweight1[i]=0;
  }
      
  for(int i=0;i<4;i++)
    afactor1[i]=val[i]*aweight1[i];
    
  for(int i=0;i<4;i++)
    aval[i]=val[i+4]; 

  for(int i=0;i<4;i++){
    aweight2[i]=aval[i]-val_curr;
  }
  bitwise_and(aweight2);

    
  for(int i=0;i<4;i++){
    aweight2[i]=4-aweight2[i];
    if(aweight2[i]<0)
      aweight2[i]=0;
  }

  for(int i=0;i<4;i++)
    afactor2[i]=aval[i]*aweight2[i];
  for(int i=0;i<4;i++)
    aweight1[i]=aweight1[i]+aweight2[i];
  for(int i=0;i<4;i++)
    afactor1[i]=afactor1[i]+afactor2[i];
    
  float weight_sum = aweight1[0]+aweight1[1]+aweight1[2]+aweight1[3];
  float factor_sum = afactor1[0]+afactor1[1]+afactor1[2]+afactor1[3];
  if (weight_sum>0) {
    float d = factor_sum/weight_sum;
    if (d>=0) {
      *(D+(v-3)*D_width+u) = d;
      // file<<v<<" "<<u<<" "<<d<<"\n";
      // printf("%d %d %f\n",v,u,d);
    }
  }
      
}
// implements approximation to bilateral filtering
void Elas::adaptiveMean (float* D) {
  ofstream file;
  file.open("mean_parallel.txt");
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }
  
  // allocate temporary memory
  float* D_copy = (float*)malloc(D_width*D_height*sizeof(float));
  float* D_tmp  = (float*)malloc(D_width*D_height*sizeof(float));
  memcpy(D_copy,D,D_width*D_height*sizeof(float));
  
  // zero input disparity maps to -10 (this makes the bilateral
  // weights of all valid disparities to 0 in this region)
  for (int32_t i=0; i<D_width*D_height; i++) {
    if (*(D+i)<0) {
      *(D_copy+i) = -10;
      *(D_tmp+i)  = -10;
    }
  }
  
  float *d_D_copy,*d_D_tmp,*d_D;
  size_t size;
  cudaMalloc(&d_D_copy,size);
  cudaMalloc(&d_D_tmp,size);
  cudaMalloc(&d_D,size);


  // when doing subsampling: 4 pixel bilateral filter width
  if (param.subsampling) {
  
    
  // full resolution: 8 pixel bilateral filter width
  } else {
    size=D_width*D_height*sizeof(float);

    cudaMemcpy(d_D_copy,D_copy,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_D_tmp,D_tmp,size,cudaMemcpyHostToDevice);
    // horizontal filter
    int numThreads=(D_height-6) * (D_width-7);
    int threadsPerBlock=32;
    int numBlocks=(numThreads+threadsPerBlock-1)/threadsPerBlock;
    adaptiveMeanKernelHorizontal<<<numBlocks,threadsPerBlock>>>(d_D_copy,D_width,d_D_tmp,D_height);
    cudaDeviceSynchronize();
    // cudaMemcpy(D_tmp,d_D_tmp,size,cudaMemcpyDeviceToHost);
    // vertical filter


    cudaMemcpy(d_D,D,size,cudaMemcpyHostToDevice);

    numThreads=(D_width-6) * (D_height-7);
    numBlocks=(numThreads+threadsPerBlock-1)/threadsPerBlock;
    adaptiveMeanKernelVertical<<<numBlocks,threadsPerBlock>>>(d_D_tmp,D_width,d_D,D_height);
    cudaDeviceSynchronize();
    cudaMemcpy(D,d_D,size,cudaMemcpyDeviceToHost);

  }
  
  // free memory
  for(int i=0;i<D_height*D_width;i++)
    file<<D[i]<<"\n";
  file.close();
  free(D_copy);
  free(D_tmp);
  cudaFree(d_D_tmp);
  cudaFree(d_D_copy);
  cudaFree(d_D);
}

void Elas::median (float* D) {
  
  // get disparity image dimensions
  int32_t D_width          = width;
  int32_t D_height         = height;
  if (param.subsampling) {
    D_width          = width/2;
    D_height         = height/2;
  }

  // temporary memory
  float *D_temp = (float*)calloc(D_width*D_height,sizeof(float));
  
  int32_t window_size = 3;
  
  float *vals = new float[window_size*2+1];
  int32_t i,j;
  float temp;
  
  // first step: horizontal median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) {
    for (int32_t v=window_size; v<D_height-window_size; v++) {
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {    
        j = 0;
        for (int32_t u2=u-window_size; u2<=u+window_size; u2++) {
          temp = *(D+getAddressOffsetImage(u2,v,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } else {
        *(D_temp+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }
        
    }
  }
  
  // second step: vertical median filter
  for (int32_t u=window_size; u<D_width-window_size; u++) {
    for (int32_t v=window_size; v<D_height-window_size; v++) {
      if (*(D+getAddressOffsetImage(u,v,D_width))>=0) {
        j = 0;
        for (int32_t v2=v-window_size; v2<=v+window_size; v2++) {
          temp = *(D_temp+getAddressOffsetImage(u,v2,D_width));
          i = j-1;
          while (i>=0 && *(vals+i)>temp) {
            *(vals+i+1) = *(vals+i);
            i--;
          }
          *(vals+i+1) = temp;
          j++;
        }
        *(D+getAddressOffsetImage(u,v,D_width)) = *(vals+window_size);
      } else {
        *(D+getAddressOffsetImage(u,v,D_width)) = *(D+getAddressOffsetImage(u,v,D_width));
      }
    }
  }
  
  free(D_temp);
  free(vals);
}