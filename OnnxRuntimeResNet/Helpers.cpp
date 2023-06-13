

#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>

#include "Helpers.h"

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>

//static std::unique_ptr<OrtTensorRTProviderOptionsV2> get_default_trt_provider_options() {
//   auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
//   tensorrt_options->device_id = 0;
//   tensorrt_options->has_user_compute_stream = 0;
//   tensorrt_options->user_compute_stream = nullptr;
//   tensorrt_options->trt_max_partition_iterations = 1000;
//   tensorrt_options->trt_min_subgraph_size = 1;
//   tensorrt_options->trt_max_workspace_size = 1 << 30;
//   tensorrt_options->trt_fp16_enable = false;
//   tensorrt_options->trt_int8_enable = false;
//   tensorrt_options->trt_int8_calibration_table_name = "";
//   tensorrt_options->trt_int8_use_native_calibration_table = false;
//   tensorrt_options->trt_dla_enable = false;
//   tensorrt_options->trt_dla_core = 0;
//   tensorrt_options->trt_dump_subgraphs = false;
//   tensorrt_options->trt_engine_cache_enable = false;
//   tensorrt_options->trt_engine_cache_path = "";
//   tensorrt_options->trt_engine_decryption_enable = false;
//   tensorrt_options->trt_engine_decryption_lib_path = "";
//   tensorrt_options->trt_force_sequential_engine_build = false;
//
//   return tensorrt_options;
//}

static std::vector<float> loadImage(const std::string& filename, int sizeX = 224, int sizeY = 224)
{
   cv::Mat image = cv::imread(filename);
   if (image.empty()) {
      std::cout << "No image found.";
   }

   // convert from BGR to RGB
   cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

   // resize
   cv::resize(image, image, cv::Size(sizeX, sizeY));

   // reshape to 1D
   image = image.reshape(1, 1);

   // uint_8, [0, 255] -> float, [0, 1]
   // Normalize number to between 0 and 1
   // Convert to vector<float> from cv::Mat.
   std::vector<float> vec;
   image.convertTo(vec, CV_32FC1, 1. / 255);

   // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
   std::vector<float> output;
   for (size_t ch = 0; ch < 3; ++ch) {
      for (size_t i = ch; i < vec.size(); i += 3) {
         output.emplace_back(vec[i]);
      }
   }
   return output;
}

static std::vector<std::string> loadLabels(const std::string& filename)
{
   std::vector<std::string> output;

   std::ifstream file(filename);
   if (file) {
      std::string s;
      while (getline(file, s)) {
         output.emplace_back(s);
      }
      file.close();
   }

   return output;
}
