#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <png++/png.hpp>
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
	// std::vector<std::vector<float>> truth=getDisparity("dataset/training/disp_occ_0/000000_10.png");
	// std::vector<std::vector<float>> estimation=getDisparity("CPU/sad/000000_10.png");
	// std::vector<float> errors= getErrors(truth,estimation);
	// std::cout<<errors[0]<<" "<<errors[1]<<"\n";
	writeDisparityFile("results/000004_10.png",false);
	// writeDisparityFile("dataset/training/disp_occ_0/000001_10.png",false);

}