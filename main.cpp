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


std::string padNum(int x){
	std::string ans=std::to_string(x);
	while(ans.length()<6)
	ans="0"+ans;
	return ans;
}
int main(){
	for(int start=0;start<200;start+=20){
		
		std::ofstream results;
		std::string file_name="results/minError"+std::to_string(start)+".txt";
		results.open (file_name);
		for(int image=start;image<start+20;image++)
		{
			std::string image_name=padNum(image)+"_10.png";
			std::string left_image_name="dataset/training/image_2/"+image_name;
			std::string right_image_name="dataset/training/image_3/"+image_name;
			std::string methods[3]={"census","ncc","sad"};
			std::vector<std::vector<float>> truth=getDisparity("dataset/training/disp_occ_0/"+image_name,true);
			float minError=1e9;
			std::string bestCmd="";
			for(int wsize=1;wsize<=21;wsize++)
				for(int dopost=0;dopost<=1;dopost++)
					for(int postconf=0;postconf<=1;postconf++){
						for(std::string method:methods){
							std::string cmd="./CPU/"+method+"/"+method;
							cmd+=" -l "+left_image_name+" -r "+right_image_name+" -ndisp 256"+" -wsize "+ std::to_string(wsize) + " -out results "+"-out_type png ";
							if(dopost)
							cmd+="-dopost ";
							if(postconf)
							cmd+="-postconf CPU/post.conf";
							// std::cout<<cmd<<"\n";
							system(cmd.c_str()); 
							std::vector<float> errors=getErrors(truth,getDisparity("results/"+image_name,false));
							if(errors[2]<minError){
								minError=errors[2];
								bestCmd=cmd;
							}
						}
					}

			results<<minError<<"\n"<<bestCmd<<"\n";

		}
		results.close();
	}
}