// Copyright 2021 ViewMagine Co.,Ltd
#include <iostream>
#include <vector>
#include <string.h>
#include "img_analysis.hpp"


/// This function implements a N-dimensional Kmeans algorithms to cluster CNN features
void kMeansClustering(std::vector<CClusterSample>* output, int epochs, const int k){
  //Randomly select centdroids
  std::vector<CClusterSample> centroids;
  for (int i = 0; i < k; ++i) {
    centroids.push_back(output->at(i));
  }
  for (std::vector<CClusterSample>::iterator c = begin(centroids); c != end(centroids); ++c) {
    int cluster_id = c - begin(centroids);
    for (std::vector<CClusterSample>::iterator it = output->begin(); it != output->end(); ++it) {
        CClusterSample p = *it;
        double dist = c->distance(p);
        if (dist < p.GetMinDist()) {
            p.SetMinDist(dist);
            p.SetCluster(cluster_id);
        }
        *it = p;
    }
  }
  std::vector<int> n_points;
  std::vector<double> sum_x, sum_y;
  // Initialise with zeroes
  for (int j = 0; j < k; ++j) {
      n_points.push_back(0);
      sum_x.push_back(0.0);
      sum_y.push_back(0.0);
  }
  // Iterate over points to append data to centroids
  for (std::vector<CClusterSample>::iterator it = output->begin(); it != output->end(); ++it) {
      int cluster_id = it->GetCluster();
      n_points[cluster_id] += 1;
      sum_x[cluster_id] += it->GetPoint().x;
      sum_y[cluster_id] += it->GetPoint().y;

      it->SetMinDist(__DBL_MAX__);  // reset distance
  }
  // Compute the new centroids
  for (std::vector<CClusterSample>::iterator c = begin(centroids); c != end(centroids); ++c) {
      int cluster_id = c - begin(centroids);
      c->SetPoint(cv::Point2f(sum_x[cluster_id] / n_points[cluster_id], sum_y[cluster_id] / n_points[cluster_id]));
  }
}