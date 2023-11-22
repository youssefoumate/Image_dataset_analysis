// Copyright 2021 ViewMagine Co.,Ltd
#include "img_analysis.hpp"
#include "utils.hpp"

/// This method clusters the features extracted from images using a CNN architecture
void CImgAnalysis::FeatCluster(std::vector<CSample> data, std::vector<CClusterSample>* output, std::string model_path, std::string config_file, int layer,
                      cv::Scalar mean, cv::Scalar std, int num_classes, int in_width, int in_height) {
  std::string image_path;
  std::vector<cv::Mat> features_list;
  cv::Mat labels, centers;
  int K  = 2;
  cv::Rect init_roi(0,0,0,0);
  CImgAnalysis img_analysis(init_roi);
  cv::Mat feat;
  cv::Mat result;
  for (int i = 0; i < data.size(); i++) {
    image_path = data[i].GetImgPath();
    result = img_analysis.FeatExtract(image_path, model_path, config_file, layer, mean, std, in_width, in_height);
    features_list.push_back(result);
  }
  cv::Mat features = cv::Mat(data.size(),features_list.size(),CV_32FC2);
  for (size_t j = 0; j < features.size[0]; j++) {
    for (size_t k = 0; k < features.size[1]; k++) {
      features.at<float>(j,k) = features_list[j].at<float>(k);
    }
  }
  CClusterSample sample;
  for (size_t i = 0; i < features_list.size(); i++) {
    sample.SetPoint(cv::Point2f(float(i),float(i+1)));
    sample.SetCluster(-1);
    sample.SetMinDist(__DBL_MAX__);
    sample.SetFeatures(features_list[i]);
    output->push_back(sample);
  }

  kMeansClustering(output,0,K);

  return;
}