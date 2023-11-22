// Copyright 2021 ViewMagine Co.,Ltd
#include <iostream>
#include <vector>
#include <string.h>


/// This function split a string intto substrings wrt. a delimiter
std::vector<std::string> SplitStrings(std::string str, char dl){
    std::string word = "";
    int num = 0;
    str = str + dl;
    int l = str.size();
    std::vector<std::string> substr_list;
    for (int i = 0; i < l; i++) {
        if (str[i] != dl) {
          word = word + str[i];
        } else {
            if ((int)word.size() != 0) {
              substr_list.push_back(word);
            }
            word = "";
        }
    }
    return substr_list;
}
/// This function parse the coordinates of bbox from string to a List of floats
std::vector<std::vector<float>> BoxToFloat(std::vector<std::string> box){
  std::string string_box;
  std::vector<std::string> string_coordinates;
  std::vector<float> float_coordinates;
  std::vector<std::vector<float>> fbox;
  char dl = ' ';
  for (int i=0; i<box.size(); i++) {
    string_box = box[i];
    string_coordinates = SplitStrings(string_box,dl);
    for (int j=0; j<string_coordinates.size(); j++) {
      float_coordinates.push_back(stof(string_coordinates[j]));
    }
    fbox.push_back(float_coordinates);
  }
  return fbox;
}
