// Copyright 2021 ViewMagine Co.,Ltd
#include <filesystem>
#include <fstream>
#include <stdlib.h>
#include "img_analysis.hpp"
#include "label_analysis.hpp"
#include "matplotlibcpp.h"
#include "cxxopts.hpp"


/// This function load data from input path.
///
/// @param[in] dataroot The dataset input path
/// @return List of dataset consists of images and labels
std::vector<CSample> DataLoader(std::string dataroot){
  std::vector<CSample> img_list;
  std::vector<std::string> bbox;
  CSample s("","",0);
  int i = 0;
  int j = 0;
  std::string str;
  std::string sub_str;
  std::string file_path;
  for (const auto & file : std::filesystem::recursive_directory_iterator(dataroot+"images")) {
    file_path = file.path().string();
    if ((file_path.find(".jpeg") != std::string::npos) || (file_path.find(".jpg") != std::string::npos) || (file_path.find(".png") != std::string::npos)) {
      i++;
      s.SetImgPath(file_path);
      s.SetIndex(i);
      img_list.push_back(s);
    }
  }
  for (const auto & file : std::filesystem::recursive_directory_iterator(dataroot+"labels")){
    file_path = file.path().string();
    if (file_path.find(".txt") != std::string::npos){
      s.SetLabelPath(file_path);
      std::ifstream lp(s.GetLabelPath());
      while (getline(lp, str)){
        bbox.push_back(str);
      }
      img_list[j].SetBbox(bbox);
      bbox.clear();
      j++;
    }
  }
  return img_list;
}
/// This is a plotting functions
///
/// @param[in] img_width The width of image or bbox
/// @param[in] img_height The height of image or bbox
/// @param[in] label The graph title
static void Plotting(std::vector<int> width, std::vector<int> height,std::string label) {
  matplotlibcpp::plot(width, height, "bo");
  matplotlibcpp::xlabel("width");
  matplotlibcpp::ylabel("height");
  matplotlibcpp::title(label);
  matplotlibcpp::show();
}
// Main example
int main(int argc, const char* argv[]) {
  std::vector <std::string> sources;
  std::string size;
  cxxopts::Options parser(argv[0], "Image Analysis Toolkit V1.0");
  parser.allow_unrecognised_options().add_options()
          ("data_path", "dataset path", cxxopts::value<std::string>()->default_value("../example/vin_data/"))
          ("format", "labels format", cxxopts::value<std::string>()->default_value("YOLO"))
          ("size", "CNN input size", cxxopts::value<int>()->default_value("224"))
          ("plot", "Graph visualization", cxxopts::value<bool>()->default_value("false"))
          ("view_console", "display results on the console", cxxopts::value<bool>()->default_value("false"))
          ("h,help", "Print usage");
  auto opt = parser.parse(argc, argv);
  if (opt.count("help")) {
      std::cout << parser.help() << std::endl;
      exit(0);
  }
  // Example
  std::vector<CSample> data = DataLoader(opt["data_path"].as<std::string>());
  cv::Mat img;
  cv::Scalar avg_intens;
  std::vector<CDims> img_dimensions;
  std::vector<cv::Scalar> mean_stddev;
  std::vector<int> img_width;
  std::vector<int> img_height;
  std::vector<int> cls_img_width;
  std::vector<int> cls_img_height;
  std::vector<CDims> bbox_dimensions;
  std::vector<CDims> cls_bbox_dimensions;
  std::vector<int> bbox_width;
  std::vector<int> bbox_height;
  std::vector<int> cls_bbox_width;
  std::vector<int> cls_bbox_height;
  std::map<float,int> dist;
  std::map<float,int>::iterator dist_it;
  cv::Rect init_roi(0,0,0,0);
  CLabelAnalysis label_analysis;
  CImgAnalysis img_analysis(init_roi);
  bool plot = opt["plot"].as<bool>();
  bool view_console = opt["view_console"].as<bool>();
  // Average intensity
  img_analysis.SetData(data);
  std::vector<CSample> img_data = img_analysis.GetData();
  std::cout << "--------Average Intensity----------" << std::endl;
  cv::Rect roi( 0, 10, 10, 10 );
  img_analysis.SetROI(roi);
  for (int i=0;i<data.size();i++) {
    img = cv::imread(data[i].GetImgPath());
    avg_intens = img_analysis.AvgIntensity(img);
    if (view_console){
      std::cout <<"image_"<<std::to_string(i)<<": "<< avg_intens << std::endl;
    }
  }
  // Image_dims
  std::cout << "--------Image dimensions----------" << std::endl;
  img_dimensions = img_analysis.ImgDim(data);
  for (int i=0;i<img_dimensions.size();i++) {
    if (view_console){
      std::cout << "width: " << img_dimensions[i].GetWidth() << std::endl;
      std::cout << "height: "<< img_dimensions[i].GetHeight() << std::endl;
    }
    img_width.push_back(img_dimensions[i].GetWidth());
    img_height.push_back(img_dimensions[i].GetHeight());

  }
  std::string label = "image dimensions";
  // Visualization
  if (plot){
    Plotting(img_width,img_height,label);
  }
  // Class-specific image dimensions
  std::cout << "--------class image dimensions----------" << std::endl;
  std::string format = opt["format"].as<std::string>();
  img_dimensions = img_analysis.ClsImgDim(data,0,format);
  for (int i=0;i<img_dimensions.size();i++) {
    if (view_console){
      std::cout << "width: "<< img_dimensions[i].GetWidth() << std::endl;
      std::cout << "height: "<< img_dimensions[i].GetHeight() << std::endl;
    }
    cls_img_width.push_back(img_dimensions[i].GetWidth());
    cls_img_height.push_back(img_dimensions[i].GetHeight());
  }
  // visualization
  label = "cls img dimensions";
  if (plot){
    Plotting(cls_img_width,cls_img_height,label);
  }
  // Mean and standard deviation
  std::cout << "--------mean stddev ----------" << std::endl;
  mean_stddev = img_analysis.MeanStd(data);
  if (view_console){
    std::cout << "mean: " << mean_stddev[0] << std::endl;
    std::cout << "standard deviation: " << mean_stddev[1] << std::endl;
  }
  //Feature extraction
  std::cout << "--------Feature Extraction----------" << std::endl;
  std::string model_path = "../example/models/bvlc_googlenet.caffemodel";
  std::string config_file = "../example/models/bvlc_googlenet.prototxt";
  //cv::Scalar mean = cv::Scalar(123.675, 116.28, 103.53);
  //cv::Scalar std = cv::Scalar(58.395, 57.12, 57.375);
  static const std::vector<float> imagenetmean{IMAGENETMEAN};
  cv::Scalar mean = cv::Scalar(imagenetmean[R_], imagenetmean[G_], imagenetmean[B_]);
  cv::Scalar std = cv::Scalar(0.0, 0.0, 0.0);
  int in_width = opt["size"].as<int>();
  int in_height = opt["size"].as<int>();
  int layer = 5;
  cv::Mat feat = img_analysis.FeatExtract(data[0].GetImgPath(), model_path, config_file, layer, mean, std, in_width, in_height);
  if (view_console){
    std::cout << "features size: (" << feat.size[0]<<","<< feat.size[1] <<","<<feat.size[2]<<"," <<feat.size[3]<<")"<<'\n';
  }
  //Image clustering
  std::cout << "--------Image clustering----------" << std::endl;
  layer = 140;
  int num_classes = 5;
  std::vector<std::vector<float>> x;
  std::vector<std::vector<float>> y;
  char cluster_color[2];
  std::vector<CClusterSample>* output = new std::vector<CClusterSample>;
  img_analysis.FeatCluster(data, output, model_path, config_file, layer, mean, std, num_classes, in_width, in_height);
  for (std::vector<CClusterSample>::iterator it = output->begin(); it != output->end(); ++it) {
      std::vector<float> temp_x;
      std::vector<float> temp_y;
      int cluster_id = it->GetCluster();
      if ( cluster_id == 0) {
        cluster_color[0] = 'b';
        cluster_color[1] = 'o';
      }else if(cluster_id == 1){
        cluster_color[0] = 'r';
        cluster_color[1] = 'o';
      }else if(cluster_id == 2){
        cluster_color[0] = 'y';
        cluster_color[1] = 'o';
      }else{
        cluster_color[0] = 'g';
        cluster_color[1] = 'o';
      }
      temp_x.push_back(it->GetPoint().x);
      temp_y.push_back(it->GetPoint().y);
      if (view_console){
        std::cout << "sample: " << it->GetPoint().x << " / cluster_ID: "<< cluster_id << '\n';
      }
      matplotlibcpp::plot(temp_x, temp_y, cluster_color);
  }
  label = "Clusters";
  if (plot){
    matplotlibcpp::title(label);
    matplotlibcpp::show();
  }
  //free memory
  std::vector<CClusterSample>().swap(*output);
  //Bbox dimensions
  std::cout << "--------bbox dimensions----------" << std::endl;
  label_analysis.SetData(data);
  std::vector<CSample> label_data = label_analysis.GetData();
  bbox_dimensions = label_analysis.BboxDim(label_data,format);
  for (int i=0; i<bbox_dimensions.size(); i++) {
    if (view_console){
      std::cout << "bbox width: "<< bbox_dimensions[i].GetWidth() << std::endl;
      std::cout << "bbox height: "<< bbox_dimensions[i].GetHeight() << std::endl;
    }
    bbox_width.push_back(bbox_dimensions[i].GetWidth());
    bbox_height.push_back(bbox_dimensions[i].GetHeight());
  }
  // visualization
  label = "bbox dimensions";
  if (plot){
    Plotting(bbox_width,bbox_height,label);
  }
  //Class Bbox dimensions
  std::cout << "--------Class bbox dimensions----------" << std::endl;
  cls_bbox_dimensions = label_analysis.ClsBboxDim(data,0,format);
  for (int i=0; i<cls_bbox_dimensions.size(); i++) {
    if (view_console){
      std::cout << "bbox width: "<< cls_bbox_dimensions[i].GetWidth() << std::endl;
      std::cout << "bbox height: "<< cls_bbox_dimensions[i].GetHeight() << std::endl;
    }
    cls_bbox_width.push_back(cls_bbox_dimensions[i].GetWidth());
    cls_bbox_height.push_back(cls_bbox_dimensions[i].GetHeight());
  }
  // visualization
  label = "cls bbox dimensions";
  if (plot){
    Plotting(cls_bbox_width,cls_bbox_height,label);
  }
  std::cout << "--------Data Distribution----------" << std::endl;
  dist = label_analysis.DataDist(data,format);
  if (view_console){
    std::cout << "| class | instances |" << std::endl;
  }
  for (dist_it = dist.begin(); dist_it != dist.end(); dist_it++) {
    if (view_console){
      std::cout << "---------------------" << std::endl;
      std::cout << "|   "<< dist_it->first << "   |     " << dist_it->second << "     |" << std::endl;
    }
  }
  return 1;
}
