// Copyright 2021 ViewMagine Co.,Ltd
#ifndef VM_DATAANALYSIS_IMGANALYSIS_H_
#define VM_DATAANALYSIS_IMGANALYSIS_H_
#define IMAGENETMEAN 104.0, 117.0, 123.0
#include <opencv2/opencv.hpp>
#include <iostream>


// RGB Channels
enum Channels {
  R_,
  G_,
  B_
};
// Dimensions of the image or the bounding box
class CDims {
private:
  int width_;
  int height_;
public:
  /// This constructor initialize the attributes of CDims Class
  ///
  /// @param[in] width The width of the image or bbox
  /// @param[in] height The height of the image or bbox
  CDims(int width, int height){
    width_ = width;
    height_ = height;
  }
  /// This method returns the width of a CDims instance
  ///
  /// @param[in] void
  /// @return the width of a CDims instance
  int GetWidth(void){
    return width_;
  }
  /// This method returns the height of a CDims instance
  ///
  /// @param[in] void
  /// @return the height of a CDims instance
  int GetHeight(void){
    return height_;
  }
  /// This method sets the width of a CDims instance
  ///
  /// @param[in] width The width of a CDims instance
  void SetWidth(int width){
    width_ = width;
  }
  /// This method sets the height of a CDims instance
  ///
  /// @param[in] height The height of image or bbox
  void SetHeight(int height){
    height_ = height;
  }
};
// A sample from the data
class CSample {
private:
  std::string image_path_;
  std::string label_path_;
  int index_;
  std::vector<std::string> bbox_;
public:
  /// This constructor initialize the attributes of CSample Class
  ///
  /// @param[in] img_path The image path
  /// @param[in] label_path The label path
  /// @param[in] Index index of the sample
  CSample(std::string img_path, std::string label_path, int index){
    image_path_ = img_path;
    label_path_ = label_path;
    index_ = index;
  }
  /// This method returns the image path of a CSample instance
  ///
  /// @param[in] void
  /// @return the image path
  std::string GetImgPath(void) {
    return image_path_;
  }
  /// This method returns the label path of a CSample instance
  ///
  /// @param[in] void
  /// @return the label path
  std::string GetLabelPath(void) {
    return label_path_;
  }
  /// This method returns the index of a CSample instance
  ///
  /// @param[in] void
  /// @return the index of a CSample instance
  int GetIndex(void) {
    return index_;
  }
  /// This method returns the list of bounding boxes of a CSample instance
  ///
  /// @param[in] void
  /// @return The list of bounding boxes of a CSample instance
  std::vector<std::string> GetBbox(void) {
    return bbox_;
  }
  /// This method sets the image path of a CSample instance
  ///
  /// @param[in] image_path The image path
  /// @return void
  void SetImgPath(std::string image_path) {
    image_path_ = image_path;
  }
  /// This method sets the label path of a CSample instance
  ///
  /// @param[in] label_path The label path
  void SetLabelPath(std::string label_path) {
    label_path_ = label_path;
  }
  /// This method sets the index of a CSample instance
  ///
  /// @param[in] index The index of a CSample instance
  void SetIndex(int index) {
    index_ = index;
  }
  /// This method sets the list of bboxes of a CSample instance
  ///
  /// @param[in] bbox The list of bboxes of an image
  void SetBbox(std::vector<std::string> bbox) {
    bbox_ = bbox;
  }
};

//This class represents the samples to be clustered using Kmeans
class CClusterSample {
private:
  cv::Point2f XY_; // coordinates to represent the samples in 2D space
  cv::Mat features_;
  int cluster_;
  double minDist_;
public:
  /// This method returns The 2D point (x,y) of a CClusterSample sample
  ///
  /// @param[in] void
  /// @return The 2D point (x,y)
  cv::Point2f GetPoint(void){
    return XY_;
  }
  /// This method returns The CNN extracted features of a sample
  ///
  /// @param[in] void
  /// @return The Matrix of CNN extracted features
  cv::Mat GetFeatures(void){
    return features_;
  }
  /// This method returns The cluster id assigned to a sample
  ///
  /// @param[in] void
  /// @return The cluster id
  int GetCluster(void){
    return cluster_;
  }
  /// This method returns The minimum distance between a centroid and a sample
  ///
  /// @param[in] void
  /// @return The minimum distance
  double GetMinDist(void){
    return minDist_;
  }
  /// This method sets the 2D point of a CClusterSample instance
  ///
  /// @param[in] XY The 2D point
  void SetPoint(cv::Point2f XY){
    XY_ = XY;
  }
  /// This method sets the features Matrix of a CClusterSample instance
  ///
  /// @param[in] features The input features Matrix
  void SetFeatures(cv::Mat features){
    features_ = features;
  }
  /// This method sets the cluster id of a CClusterSample instance
  ///
  /// @param[in] cluster The inpur cluster id
  void SetCluster(int cluster){
    cluster_ = cluster;
  }
  /// This method sets the minimum distance of a CClusterSample instance
  ///
  /// @param[in] minDist The minimum distance
  void SetMinDist(double minDist){
    minDist_ = minDist;
  }
  /// This method caluctes the multidimensional eucledean distance between two samples
  ///
  /// @param[in] features The sample features extracted from a CNN model
  double distance(CClusterSample features) {
    return cv::norm(features_,features.GetFeatures(),cv::NORM_L2SQR);
  }
};

// This class implements the data image analysis methods
class CImgAnalysis {
private:
  std::vector<CSample> data_;
  cv::Rect roi_;
public:
  /// This constructor initialize the attributes of CImgAnalysis Class
  ///
  /// @param[in] roi The region of interest in the image
  CImgAnalysis(cv::Rect roi){
    roi_ = roi;
  }
  /// This method sets the data samples for CImgAnalysis class
  ///
  /// @param[in] data The dataset samples
  void SetData(std::vector<CSample> data){
    data_ = data;
  }
  /// This method sets the data samples for CImgAnalysis class
  ///
  /// @param[in] roi The region of interest
  void SetROI(cv::Rect roi){
    roi_ = roi;
  }
  /// This method returns the roi of CImgAnalysis class
  ///
  /// @param[in] void
  /// @return the roi
  cv::Rect GetROI(void){
    return roi_;
  }
  /// This method returns the dataset samples for CImgAnalysis class
  ///
  /// @param[in] void
  /// @return the data samples
  std::vector<CSample> GetData(void){
    return data_;
  }
  /// This method caluclates the average intensity of an image or a sub image.
  ///
  /// @param[in] img The input image
  /// @param[in] roi The ROI where to calculate the average intensity
  /// @return Average intensity
  cv::Scalar AvgIntensity(cv::Mat& img);
  /// This method arranges the images dimensions of a specific class
  ///
  /// @param[in] data The dataset images and their labels
  /// @param[in] cls The class of a bounding box
  /// @param[in] format The labels' format
  /// @return List of dimensions
  std::vector<CDims> ClsImgDim(std::vector<CSample> data, float cls, std::string format);
  /// This method Arranges the dimension of images in a list
  ///
  /// @param[in] data The dataset images and their labels
  /// @return List of image dimensions
  std::vector<CDims> ImgDim(std::vector<CSample> data);
  /// This method caluclates the average intensity of an image or a sub image
  ///
  /// @params[in] data The image dataset and its labels
  /// @return Mean and std of the dataset
  std::vector<cv::Scalar> MeanStd(std::vector<CSample> data);
  /// This method extracts the features from images using a CNN architecture
  ///
  /// @param[in] image_path The path to an image
  /// @param[in] model_path The path to a model weigths
  /// @param[in] config_file The config file of the model
  /// @param[in] layer The output layer index
  /// @param[in] mean The mean of the image dataset
  /// @param[in] std The standard deviation of the image dataset
  /// @param[in] in_width The input width of the model
  /// @param[in] in_height The input height of the model
  /// @return Matrix of extracted features
  cv::Mat FeatExtract(std::string image_path, std::string model_path, std::string config_file, int layer,
                      cv::Scalar mean, cv::Scalar std, int in_width, int in_height);
  /// This method clusters the dataset images based on their CNN features
  ///
  /// @param[in] data The images and labels of the dataset
  /// @param[out] output The clustered samples
  /// @param[in] model_path The path to a model weigths
  /// @param[in] config_file The config file of the model
  /// @param[in] layer The output layer index
  /// @param[in] mean The mean of the image dataset
  /// @param[in] std The standard deviation of the image dataset
  /// @param[in] num_classes The number of clusters
  /// @param[in] in_width The input width of the model
  /// @param[in] in_height The input height of the model
  void FeatCluster(std::vector<CSample> data, std::vector<CClusterSample>* output, std::string model_path, std::string config_file, int layer,
                      cv::Scalar mean, cv::Scalar std, int num_classes, int in_width, int in_height);
};

#endif // VM_DATAANALYSIS_IMGANALYSIS_H_