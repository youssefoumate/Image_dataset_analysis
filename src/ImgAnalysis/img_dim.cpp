// Copyright 2021 ViewMagine Co.,Ltd
#include "img_analysis.hpp"


/// This method Arranges the dimension of images in a list
std::vector<CDims> CImgAnalysis::ImgDim(std::vector<CSample> data){
  std::vector<CDims> dim;
  CDims d(0,0);
  cv::Mat img;
  for (int i =0; i<data.size(); i++) {
    img = cv::imread(data[i].GetImgPath());
    d.SetWidth(img.cols);
    d.SetHeight(img.rows);
    dim.push_back(d);
  }
  return dim;
}
