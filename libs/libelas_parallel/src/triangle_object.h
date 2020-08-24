 struct triangle {
    int32_t c1,c2,c3;
    float   t1a,t1b,t1c;
    float   t2a,t2b,t2c;
    triangle(){
      
    }
    triangle(int32_t c1,int32_t c2,int32_t c3):c1(c1),c2(c2),c3(c3){}
  };