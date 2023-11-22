// Copyright 2021 ViewMagine Co.,Ltd
#ifndef VM_DATAANALYSIS_LABELANALYSIS_H_
#define VM_DATAANALYSIS_LABELANALYSIS_H_
#include "img_analysis.hpp"


// YOLO bounding box format
enum YOLOBox {
  Y_class_,
  Y_topx_,
  Y_topy_,
  Y_width_,
  Y_height_
};

// This class implements the data label analysis methods
class CLabelAnalysis {
private:
  std::vector<CSample> data_;
public:
  /// This method sets the data samples for CLabelAnalysis class
  /// @param[in] data The dataset samples
  void SetData(std::vector<CSample> data){
    data_ = data;
  }
  /// This method returns the dataset samples for CLabelAnalysis class
  /// @param[in] void
  /// @return the data samples
  std::vector<CSample> GetData(void){
    return data_;
  }
  /// This method arranges the bbox dimensions
  ///
  /// @param[in] data The dataset images and their labels
  /// @param[in] format The labels' format
  /// @return List of bbox dimensions
  std::vector<CDims> BboxDim(std::vector<CSample> data, std::string format);
  /// This method arranges the bbox dimensions of a specific class
  ///
  /// @param[in] data The dataset images and their labels
  /// @param[in] cls The class of the bounding box
  /// @return List of bbox dimensions
  std::vector<CDims> ClsBboxDim(std::vector<CSample> data, float cls, std::string format);
  /// This method calculates the number of bboxes in each class
  ///
  /// @param[in] data The dataset images and their labels
  /// @param[in] format The labels format
  /// @return Map of classes and their instances count
  std::map<float,int> DataDist(std::vector<CSample> data, std::string format);
};

#endif // VM_DATAANALYSIS_LABELANALYSIS_H_