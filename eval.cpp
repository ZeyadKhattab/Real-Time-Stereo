#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <png++/png.hpp>
#include <stdlib.h>
#include <string>

#define ABS_THRESH 3.0
#define REL_THRESH 0.5

std::vector<std::vector<double>> getDisparity(const std::string file_name,bool groundTruth) {
	std::vector<std::vector<double>> ans;
	png::image<png::gray_pixel_16> image(file_name);
	int32_t   width_  = image.get_width();
   	int32_t  height_ = image.get_height();
	for (int32_t v=0; v<height_; v++) {
    	std::vector<double> curr;
      	for (int32_t u=0; u<width_; u++) {
			uint16_t val = image.get_pixel(u,v);
			curr.push_back(groundTruth && val==0?-1:((double)val)/256.0);
		}
    	ans.push_back(curr);
    }
    return ans;
}

	/* errors, pixels, MSE */
std::vector<double> getErrors(std::vector<std::vector<double>>truth, std::vector<std::vector<double>> estimation){
	assert(truth.size()==estimation.size());
	//assert(truth[0].size()==estimation[0].size()); //applying dopost increases width? 
	int pixels=0,errors=0;
	long long SSD=0;
	for(int i=0;i<truth.size();i++)
		for(int j=0;j<truth[0].size();j++){
			if(truth[i][j]==-1) continue;
			double truth_value=truth[i][j],estimation_value=estimation[i][j];
			bool err = fabs(truth_value-estimation_value)>ABS_THRESH && 
				fabs(truth_value-estimation_value)/fabs(truth_value)>REL_THRESH;
			if(err)
				errors++;
			pixels++;
			SSD+=(truth_value-estimation_value)*(truth_value-estimation_value);
		}
	std::vector<double>ans;
	ans.push_back(errors);
	ans.push_back(pixels);
	ans.push_back(SSD);
	return ans;
}
  
std::vector<double> getErrors(const std::string truth_file_name,const std::string estimation_file_name){
	std::vector<std::vector<double>>truth = getDisparity(truth_file_name,true);
	std::vector<std::vector<double>> estimation= getDisparity(estimation_file_name,false);
	return getErrors(truth,estimation);

}

std::string padNum(int x){
	std::string ans=std::to_string(x);
	while(ans.length()<6)
	ans="0"+ans;
	return ans;
}
int main(){
	
		
        int pixels=0;
        long long SSD=0;
        int errors=0;
		for(int image=0;image<200;image++)
		{
			std::string image_name=padNum(image)+"_10.png";
			std::string left_image_name="dataset/training/image_2/"+image_name;
			std::string right_image_name="dataset/training/image_3/"+image_name;
			std::vector<double>curr=getErrors("dataset/training/disp_occ_0/"+image_name,"results/"+image_name);
            errors+=curr[0];
            pixels+=curr[1];
            SSD+=curr[2];
					


		}
        std::cout<<errors<<" "<<" "<<pixels<<" "<<errors*1.0/pixels<<" "<<SSD<<" "<<SSD*1.0/pixels<<"\n";
		// results.close();
	
}