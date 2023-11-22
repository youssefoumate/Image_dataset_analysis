// Copyright 2021 ViewMagine Co.,Ltd
#include <opencv2/dnn.hpp>
#include "img_analysis.hpp"


/// This method extracts features from images using a CNN architecture
cv::Mat CImgAnalysis::FeatExtract(std::string image_path, std::string model_path, std::string config_file, int layer,
                      cv::Scalar mean, cv::Scalar std, int in_width=244, int in_height=244) {
  cv::Mat image = cv::imread(image_path);
  cv::Mat blob;
  int rsz_width = in_width;
  int rsz_height = in_height;
  int scale = 1;
  cv::String model = cv::samples::findFile(model_path);
  CV_Assert(!model.empty());
  cv::dnn::Net net = cv::dnn::readNet(model, config_file);
  // Create a window
  if (rsz_width != 0 && rsz_height != 0) {
      cv::resize(image, image, cv::Size(rsz_width, rsz_height));
  }
  cv::dnn::blobFromImage(image, blob, scale, cv::Size(in_width, in_height), mean);
  if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0) {
      // Divide blob by std.
      cv::divide(blob, std, blob);
  }
  net.setInput(blob);
  std::vector<std::string> layer_names = net.getLayerNames();
  //Layer names visualization
  /*for (size_t i = 0; i < layer_names.size(); i++) {
    std::cout << "layer{"<< i+1 << "}: " << layer_names[i] << '\n';
    if (i==5){
      std::cout << "..." << '\n';
      break;
    }
  }*/
  cv::Mat feat = net.forward(layer_names[layer]);
  return feat;
}