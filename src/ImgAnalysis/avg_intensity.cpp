// Copyright 2021 ViewMagine Co.,Ltd
#include "img_analysis.hpp"


/// This method caluclates the average intensity of an image or a sub image.
cv::Scalar CImgAnalysis::AvgIntensity(cv::Mat& img) {

  cv::Scalar avgPixelIntensity;
  if (roi_.area()==0) {
    avgPixelIntensity = cv::mean(img);
  } else {
    cv::Mat image_roi = img(roi_);
    avgPixelIntensity = cv::mean(image_roi);
  }
  return avgPixelIntensity;
}
