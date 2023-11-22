// Copyright 2021 ViewMagine Co.,Ltd
#include "img_analysis.hpp"
#include "label_analysis.hpp"
#include "utils.hpp"


/// This method arranges the bbox dimensions
std::vector<CDims> CLabelAnalysis::BboxDim(std::vector<CSample> data, std::string format) {
  std::vector<std::vector<float>> boxes;
  std::vector<float> box;
  std::vector<CDims> dim;
  CDims d(0,0);
  cv::Mat img;
  int cls_idx;
  int topx_idx;
  int topy_idx;
  int width_idx;
  int height_idx;
  // YOLO format (Other formats will be supported in the future)
  if (format == "YOLO") {
    cls_idx = Y_class_;
    topx_idx = Y_topx_;
    topy_idx = Y_topy_;
    width_idx = Y_width_;
    height_idx = Y_height_;
  }
  for(int i =0; i<data.size(); i++) {
    boxes = BoxToFloat(data[i].GetBbox());
    img = cv::imread(data[i].GetImgPath());
    for (int j = 0; j<boxes.size(); j++) {
      box = boxes[j];
      d.SetWidth(box[width_idx]*img.cols);
      d.SetHeight(box[height_idx]*img.rows);
      dim.push_back(d);

    }
  }
  return dim;
}