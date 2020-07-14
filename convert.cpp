#include <png++/png.hpp>
#include<iostream>
#include <fstream>


using Image = png::image<png::gray_pixel>;
void convert(std::string image_name,std::string file_name){
    Image image (image_name);
    int n=image.get_height();
    int m=image.get_width();
    std::ofstream file;
    file.open (file_name);
    file<<"P5\n";
    file<<m<<" "<<n<<"\n255\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
           file<<(unsigned char)image.get_pixel(j,i);
    }
    file.close();
}
std::string padNum(int x){
	std::string ans=std::to_string(x);
	while(ans.length()<6)
	ans="0"+ans;
	return ans;
} 
int main(int argc, char** argv){
    int l,r;
    if(argc==3){
        l=std::stoi(argv[0]);
        r=std::stoi(argv[1]);
    }
    else{
        l=0;
        r=199;
    }

    for(int img=l;img<=r;img++){
    convert("../../dataset/training/image_2/"+padNum(img)+"_10.png","pgm_dataset/left/"+std::to_string(img)+".pgm");
    convert("../../dataset/training/image_3/"+padNum(img)+"_10.png","pgm_dataset/right/"+std::to_string(img)+".pgm");

    }
   
}
