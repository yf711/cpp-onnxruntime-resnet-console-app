// OnnxRuntimeResNet.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <ctime>
#include "Helpers.cpp"

int main()
{
   Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
   const auto& api = Ort::GetApi();
   Ort::RunOptions runOptions;
   Ort::Session session(nullptr);

   constexpr int64_t numChannels = 3;
   constexpr int64_t width = 224;
   constexpr int64_t height = 224;
   constexpr int64_t numClasses = 1000;
   constexpr int64_t numInputElements = numChannels * height * width;


   const std::string imageFile = "C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\dog.png";
   const std::string labelFile = "C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\imagenet_classes.txt";
   auto modelPath = L"C:\\code\\cpp-onnxruntime-resnet-console-app\\OnnxRuntimeResNet\\assets\\resnet50v2.onnx";


   //load labels
   std::vector<std::string> labels = loadLabels(labelFile);
   if (labels.empty()) {
      std::cout << "Failed to load labels: " << labelFile << std::endl;
      return 1;
   }

   // load image
   const std::vector<float> imageVec = loadImage(imageFile);
   if (imageVec.empty()) {
      std::cout << "Failed to load image: " << imageFile << std::endl;
      return 1;
   }

   if (imageVec.size() != numInputElements) {
      std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
      return 1;
   }

   //// One way adding trt options
   //Ort::SessionOptions sf;
   //int device_id = 0;
   //OrtCUDAProviderOptions cuda_options;
   //cuda_options.device_id = device_id;
   //cuda_options.do_copy_in_default_stream = true;
   //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
   //sf.AppendExecutionProvider_CUDA(cuda_options);

   // 2nd way adding trt options
   //**************************************************************************************************************************
   // It's suggested to use CreateTensorRTProviderOptions() to get provider options
   // since ORT takes care of valid options for you
   //**************************************************************************************************************************
   Ort::SessionOptions session_options;
   OrtTensorRTProviderOptionsV2* tensorrt_options;
   Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
   tensorrt_options->device_id = 0;
   tensorrt_options->trt_fp16_enable = true;
   tensorrt_options->trt_dla_enable = true;
   tensorrt_options->trt_dla_core = 0;
   tensorrt_options->trt_engine_cache_enable = true;
   tensorrt_options->trt_engine_cache_path = "C:\\tmp";
   std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
   Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options),
      rel_trt_options.get()));

   // 3rd way adding trt options
   //*****************************************************************************************
   // It's not suggested to directly new OrtTensorRTProviderOptionsV2 to get provider options
   //*****************************************************************************************
   //
   // auto tensorrt_options = get_default_trt_provider_options();
   // session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options.get());

   time_t start, s_created, pre_infer, finish;
   // stores time in current_time
   time(&start);
   std::cout << "Running ORT TRT EP with default provider options" << std::endl;

   // create session
   session = Ort::Session(env, modelPath, session_options);
   time(&s_created);

   // define shape
   const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
   const std::array<int64_t, 2> outputShape = { 1, numClasses };

   // define array
   std::array<float, numInputElements> input;
   std::array<float, numClasses> results;

   // define Tensor
   auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
   auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
   auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

   // copy image data to input array
   std::copy(imageVec.begin(), imageVec.end(), input.begin());



   // define names
   Ort::AllocatorWithDefaultOptions ort_alloc;
   Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
   Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
   const std::array<const char*, 1> inputNames = { inputName.get() };
   const std::array<const char*, 1> outputNames = { outputName.get() };
   inputName.release();
   outputName.release();


   // run inference
   try {
      time(&pre_infer);
      session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
      time(&finish);
   }
   catch (Ort::Exception& e) {
      std::cout << e.what() << std::endl;
      return 1;
   }

   // sort results
   std::vector<std::pair<size_t, float>> indexValuePairs;
   for (size_t i = 0; i < results.size(); ++i) {
      indexValuePairs.emplace_back(i, results[i]);
   }
   std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

   // show Top5
   for (size_t i = 0; i < 5; ++i) {
      const auto& result = indexValuePairs[i];
      std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
   }
   std::cout << "Session creation latency = " << difftime(s_created, start) << " seconds" << std::endl;
   std::cout << "Inference latency = " << difftime(finish, pre_infer) << " seconds" << std::endl;
}