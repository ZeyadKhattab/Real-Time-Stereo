 #include <stdint.h>

 struct parameters {
    int32_t disp_min;               // min disparity
    int32_t disp_max;               // max disparity
    float   support_threshold;      // max. uniqueness ratio (best vs. second best support match)
    int32_t support_texture;        // min texture for support points
    int32_t candidate_stepsize;     // step size of regular grid on which support points are matched
    int32_t incon_window_size;      // window size of inconsistent support point check
    int32_t incon_threshold;        // disparity similarity threshold for support point to be considered consistent
    int32_t incon_min_support;      // minimum number of consistent support points
    bool    add_corners;            // add support points at image corners with nearest neighbor disparities
    int32_t grid_size;              // size of neighborhood for additional support point extrapolation
    float   beta;                   // image likelihood parameter
    float   gamma;                  // prior constant
    float   sigma;                  // prior sigma
    float   sradius;                // prior sigma radius
    int32_t match_texture;          // min texture for dense matching
    int32_t lr_threshold;           // disparity threshold for left/right consistency check
    float   speckle_sim_threshold;  // similarity threshold for speckle segmentation
    int32_t speckle_size;           // maximal size of a speckle (small speckles get removed)
    int32_t ipol_gap_width;         // interpolate small gaps (left<->right, top<->bottom)
    bool    filter_median;          // optional median filter (approximated)
    bool    filter_adaptive_mean;   // optional adaptive mean filter (approximated)
    bool    postprocess_only_left;  // saves time by not postprocessing the right image
    bool    subsampling;            // saves time by only computing disparities for each 2nd pixel
                                    // note: for this option D1 and D2 must be passed with size
                                    //       width/2 x height/2 (rounded towards zero)
    
    // constructor
    parameters (/* setting s=ROBOTICS */) {
      
      // default settings in a robotics environment
      // (do not produce results in half-occluded areas
      //  and are a bit more robust towards lighting etc.)
      
        disp_min              = 0;
        disp_max              = 255;
        support_threshold     = 0.85;
        support_texture       = 10;
        candidate_stepsize    = 5;
        incon_window_size     = 5;
        incon_threshold       = 5;
        incon_min_support     = 5;
        add_corners           = 0;
        grid_size             = 20;
        beta                  = 0.02;
        gamma                 = 3;
        sigma                 = 1;
        sradius               = 2;
        match_texture         = 1;
        lr_threshold          = 2;
        speckle_sim_threshold = 1;
        speckle_size          = 200;
        ipol_gap_width        = 3;
        filter_median         = 0;
        filter_adaptive_mean  = 1;
        postprocess_only_left = 1;
        subsampling           = 0;
        
      // default settings for middlebury benchmark
      // (interpolate all missing disparities)
      
    }
  };