#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <png++/png.hpp>
#define ABS_THRESH 7.0
#define REL_THRESH 0.85
void writeDisparityFile (const std::string file_name) {
	std::ofstream myfile;
  	myfile.open ("disp_truth.txt");

    png::image<png::gray_pixel_16> image(file_name);
  	int32_t   width_  = image.get_width();
   	int32_t  height_ = image.get_height();
   	float*  data_   = (float*)malloc(width_*height_*sizeof(float));
    for (int32_t v=0; v<height_; v++) {
      for (int32_t u=0; u<width_; u++) {
        uint16_t val = image.get_pixel(u,v);
        if (val==0) myfile<<-1<<" ";
        else        myfile<<((float)val)/256.0<<" ";
      }
      myfile<<"\n";
    }
     myfile.close();

  }

std::vector<std::vector<float>> getDisparity(const std::string file_name) {
	std::vector<std::vector<float>> ans;
    png::image<png::gray_pixel_16> image(file_name);
  	int32_t   width_  = image.get_width();
   	int32_t  height_ = image.get_height();
   	float*  data_   = (float*)malloc(width_*height_*sizeof(float));
    for (int32_t v=0; v<height_; v++) {
    	std::vector<float> curr;
      for (int32_t u=0; u<width_; u++) {
        uint16_t val = image.get_pixel(u,v);
        if (val==0) curr.push_back(-1);
        else       curr.push_back(((float)val)/256.0);
      }
      ans.push_back(curr);
    }
    return ans;


  }
  std::vector<float> getErrors(std::vector<std::vector<float>>truth, std::vector<std::vector<float>> estimation){
  	assert(truth.size()==estimation.size());
  	assert(truth[0].size()==estimation[0].size());
  	float pixels=0,errors=0;
	float minDiff = 1e9;
  	for(int i=0;i<truth.size();i++)
  		for(int j=0;j<truth[0].size();j++){
  			if(truth[i][j]==-1) continue;
  			float truth_value=truth[i][j],estimation_value=estimation[i][j];
			minDiff=std::min(minDiff,fabs(truth_value-estimation_value));
  			bool  err = fabs(truth_value-estimation_value)>ABS_THRESH && 
  				fabs(truth_value-estimation_value)/fabs(truth_value-estimation_value)>REL_THRESH;
  			if(err)
  				errors++;
  			pixels++;
  		}
		std::cout<<minDiff<<"\n";
  	std::vector<float>ans;
  	ans.push_back(errors);
  	ans.push_back(pixels);
  	return ans;
  }



int main(){
	std::vector<std::vector<float>> truth=getDisparity("dataset/training/disp_occ_0/000000_10.png");
	std::vector<std::vector<float>> estimation=getDisparity("CPU/sad/000000_10.png");
	std::vector<float> errors= getErrors(truth,estimation);
	std::cout<<errors[0]<<" "<<errors[1]<<"\n";


}