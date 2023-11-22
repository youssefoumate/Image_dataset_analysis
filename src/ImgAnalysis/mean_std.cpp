// Copyright 2021 ViewMagine Co.,Ltd
#include "img_analysis.hpp"


/// This method caluclates the average intensity of an image or a sub image
std::vector<cv::Scalar> CImgAnalysis::MeanStd(std::vector<CSample> data){
  std::vector<cv::Scalar> out;
  cv::Scalar t_mean(0,0,0,0), t_stddev(0,0,0,0);
  cv::Scalar mean(0,0,0,0), stddev(0,0,0,0);
  float nb_samples = 0.0;
  int data_size = data.size();
  cv::Mat img;
  for (int i=0; i<data.size(); i++) {
    img = cv::imread(data[i].GetImgPath());
    cv::meanStdDev(img, mean, stddev);
    t_mean += mean;
    t_stddev += stddev;
  }
  out.push_back(mean/data_size);
  out.push_back(t_stddev/data_size);
  return out;
}
