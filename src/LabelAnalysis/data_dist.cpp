// Copyright 2021 ViewMagine Co.,Ltd
#include "label_analysis.hpp"
#include "utils.hpp"


/// This method calculates the number of bboxes in each class
std::map<float,int> CLabelAnalysis::DataDist(std::vector<CSample> data, std::string format){
  std::vector<std::vector<float>> boxes;
  std::vector<float> classes;
  std::vector<float> classes_u;
  std::map<float,int> dist;
  int cls_idx;
  if (format == "YOLO") {
    cls_idx = Y_class_;
  }
  for (int i =0; i<data.size(); i++) {
    boxes = BoxToFloat(data[i].GetBbox());
    for (int j = 0; j<boxes.size(); j++) {
      classes.push_back(boxes[j][cls_idx]);
    }
  }
  std::sort(classes.begin(),classes.end());
  std::vector<float> s_classes = classes;
  classes.erase(std::unique(classes.begin(),classes.end()),classes.end());
  for (int i=0; i<s_classes.size(); i++) {
    //std::cout << classes[i] << std::endl;
    dist.insert(std::pair<float,int>(s_classes[i],count(s_classes.begin(), s_classes.end(), s_classes[i])));
  }
  return dist;
}