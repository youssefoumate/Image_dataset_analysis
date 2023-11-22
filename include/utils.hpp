// Copyright 2021 ViewMagine Co.,Ltd
#ifndef VM_DATAANALYSIS_UTILS_H_
#define VM_DATAANALYSIS_UTILS_H_
#include <iostream>


/// This function parse the coordinates of bbox from string to a List of floats
///
/// @params[in] box The bbox coordinates
/// @return List of the bbox coordinates
std::vector<std::vector<float>> BoxToFloat(std::vector<std::string>);
/// This function split a string intto substrings wrt. a delimiter
///
/// @params[in] str The string to split
/// @params[in] dl The Delimiter
/// @return List of substrings
std::vector<std::string> SplitStrings(std::string, char);
/// This function implements a N-dimensional Kmeans algorithms to cluster CNN features
///
/// @param[in] output The list of samples to be clustered
/// @param[in] epochs The number of clustering iterations
/// @param[in] k The number of clusters
void kMeansClustering(std::vector<CClusterSample>*, int, int);

#endif // VM_DATAANALYSIS_UTILS_H_