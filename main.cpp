#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <png++/png.hpp>
#include <stdlib.h>
#include <string>

#define ABS_THRESH 7.0
#define REL_THRESH 0.85
std::vector<std::vector<float>> getDisparity(const std::string file_name,bool groundTruth) {
	std::vector<std::vector<float>> ans;
	png::image<png::gray_pixel_16> image(file_name);
	int32_t   width_  = image.get_width();
   	int32_t  height_ = image.get_height();
   	float*  data_   = (float*)malloc(width_*height_*sizeof(float));
    for (int32_t v=0; v<height_; v++) {
    	std::vector<float> curr;
      	for (int32_t u=0; u<width_; u++) {
			uint16_t val = image.get_pixel(u,v);
			curr.push_back(groundTruth && val==0?-1:((float)val)/256.0);
		}
    	ans.push_back(curr);
    }
    return ans;


  }
void writeDisparityFile (const std::string file_name,bool groundTruth) {
	std::ofstream myfile;
  	myfile.open ("results/disp.txt");
	std::vector<std::vector<float>>disparity=getDisparity(file_name,groundTruth);
  	int width  = disparity[0].size();
   	int height= disparity.size();
    for (int v=0; v<height; v++) {
    	for (int u=0; u<width; u++) 
		{
			float val = disparity[v][u];
			myfile<<val<<" ";
      	}
    	myfile<<"\n";
    }
    myfile.close();
}
	/* errors, pixels, MSE */
  std::vector<float> getErrors(std::vector<std::vector<float>>truth, std::vector<std::vector<float>> estimation){
  	assert(truth.size()==estimation.size());
  	//assert(truth[0].size()==estimation[0].size()); //applying dopost increases width? 
  	float pixels=0,errors=0;
	float minDiff = 1e9;
	float SSD=0;
  	for(int i=0;i<truth.size();i++)
  		for(int j=0;j<truth[0].size();j++){
  			if(truth[i][j]==-1) continue;
  			float truth_value=truth[i][j],estimation_value=estimation[i][j];
			minDiff=std::min(minDiff,fabs(truth_value-estimation_value));
  			bool err = fabs(truth_value-estimation_value)>ABS_THRESH && 
  				fabs(truth_value-estimation_value)/fabs(truth_value)>REL_THRESH;
  			if(err)
  				errors++;
  			pixels++;
			SSD+=(truth_value-estimation_value)*(truth_value-estimation_value);
  		}
	//std::cout<<minDiff<<"\n";
  	std::vector<float>ans;
  	ans.push_back(errors);
  	ans.push_back(pixels);
	ans.push_back(SSD/pixels);
  	return ans;
  }
std::vector<float> getErrors(const std::string truth_file_name,const std::string estimation_file_name){
	std::vector<std::vector<float>>truth = getDisparity(truth_file_name,true);
	std::vector<std::vector<float>> estimation= getDisparity(estimation_file_name,false);
	return getErrors(truth,estimation);

}

std::string padNum(int x){
	std::string ans=std::to_string(x);
	while(ans.length()<6)
	ans="0"+ans;
	return ans;
}
int main(int argc, char** argv){
	std::cout<<getErrors(argv[0],argv[1])[2];
}