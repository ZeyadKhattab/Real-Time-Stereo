#include "elas.h"
#include <iostream>
// #include <stdint.h>

// #include <png++/png.hpp>

// using Image = png::image<png::gray_pixel>;
int main(){
    std::cout<<"Hello\n";
    Elas::parameters param;
    Elas elas(param);
     elas.hello(10);

    // Image right ("right.png");
	// Image left ("left.png");
    
    // size_t size=right.get_width()*right.get_height()*sizeof(unsigned char);
    // size_t size=10*10*sizeof(uint8_t);

    // unsigned char* I1 = (unsigned char*)malloc(size);
    // unsigned char* I2 = (unsigned char*)malloc(size);
    // size=right.get_width()*right.get_height()*sizeof(float);
    // // size=10*10*sizeof(float);
    // float* D1=(float*)malloc(size);;
    // float* D2=(float*)malloc(size);;
    // const int32_t dims[3] = {right.get_width(),right.get_height(),right.get_width()};
    // dims[0]=right.get_width();
    // dims[1]=right.get_height();
    // dims[2]=dims[0];
    // dims[0]=10;
    // dims[1]=10;
    // dims[2]=10;
    // elas.process(I1,I2,D1,D2,dims);
    // int n=10;
    // size_t size=n*n*sizeof(uint8_t);
    // uint8_t* I1 = (uint8_t*)malloc(size);
    // uint8_t* I2 = (uint8_t*)malloc(size);
    // size=n*n*sizeof(float);
    // float* D1=(float*)malloc(size);
    // float* D2=(float*)malloc(size);
    // const int32_t dims[3] = {n,n,n};
    // process(I1,I2,D1,D2,dims);
}

