#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>

// Function to calculate expected values
double calculate_expected(const std::string& function_name, double x) {
    if (function_name == "square") return x * x;
    if (function_name == "cube") return x * x * x;
    if (function_name == "sin") return std::sin(x);
    return 0.0;
}

// Function to save results to CSV
void save_to_csv(const std::string& function_name,
                 const std::vector<float>& x_values,
                 const std::vector<float>& predicted_values,
                 const std::vector<float>& expected_values) {
    // Create filename based on function name
    std::string filename = function_name + "_result.csv";
    
    // Open file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Set precision for floating point numbers
    file << std::fixed << std::setprecision(6);

    // Write header
    file << "X,Predicted,Expected\n";

    // Write data
    for (size_t i = 0; i < x_values.size(); ++i) {
        file << x_values[i] << ","
             << predicted_values[i] << ","
             << expected_values[i] << "\n";
    }

    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

// Function to test a model
void test_model(const std::string& model_path, const std::string& function_name,
                const std::vector<float>& x_values) {
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Create session
        Ort::Session session(env, model_path.c_str(), session_options);

        // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, 1};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<float> predicted_values;
        std::vector<float> expected_values;

        // Print header
        std::cout << "Function: " << function_name << "\nPredictions\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        // Process each input value
        for (float x : x_values) {
            std::vector<float> input_tensor = {x};
            
            // Create input tensor
            Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());

            // Prepare input/output names
            const char* input_names[] = {input_name.get()};
            const char* output_names[] = {output_name.get()};

            // Run inference
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor_ort, 1,
                output_names, 1);

            float predicted = output_tensors.front().GetTensorData<float>()[0];
            float expected = calculate_expected(function_name, x);

            predicted_values.push_back(predicted);
            expected_values.push_back(expected);

            // Print results
            std::cout << std::fixed << std::setprecision(6)
                      << "x = " << std::setw(5) << x 
                      << "  predicted = " << std::setw(10) << predicted
                      << "  expected = " << std::setw(10) << expected << "\n";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "\nTesting Time: " << duration.count() / 1000000.0 << " seconds\n\n";

        // Save results to CSV
        save_to_csv(function_name, x_values, predicted_values, expected_values);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    // Define input values
    std::vector<float> x_values = {-1.0f, -0.8f, -0.6f, -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};

    // Define model paths
    const std::string base_path = "/home/vboxuser/PycharmProjects/ONNX_ApproxFunc/";
    std::vector<std::pair<std::string, std::string>> models = {
        {base_path + "Square_model.onnx", "square"},
        {base_path + "Cube_model.onnx", "cube"},
        {base_path + "Sin_model.onnx", "sin"}
    };

    // Test each model
    for (const auto& [model_path, function_name] : models) {
        test_model(model_path, function_name, x_values);
    }

    return 0;
}
