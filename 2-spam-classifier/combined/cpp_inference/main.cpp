#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

// Structure to hold our test data
struct TestData {
    std::string message;
    int label;  // 1 for spam, 0 for not spam
};

// Function to tokenize text using vocabulary
std::vector<float> tokenize_text(const std::string& text, const std::string& vocab_file, int max_length = 50) {
    std::vector<float> result(max_length, 0.0f);
    std::unordered_map<std::string, int> word_index;
    
    // Read vocabulary
    std::ifstream vocab_stream(vocab_file);
    std::string line;
    // Skip header lines
    for (int i = 0; i < 3; i++) {
        std::getline(vocab_stream, line);
    }
    while (std::getline(vocab_stream, line)) {
        std::istringstream iss(line);
        std::string word;
        int index;
        iss >> word >> index;
        word_index[word] = index;
    }
    
    // Tokenize input text
    std::istringstream iss(text);
    std::string word;
    int pos = 0;
    while (iss >> word && pos < max_length) {
        // Convert word to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        result[pos++] = word_index.count(word) ? word_index[word] : 0;
    }
    
    return result;
}

// Function to read test data from CSV
std::vector<TestData> read_test_data(const std::string& file_path) {
    std::vector<TestData> data;
    std::ifstream file(file_path);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string message;
        std::string label_str;
        
        if (std::getline(iss, message, ',') && std::getline(iss, label_str)) {
            TestData entry;
            entry.message = message;
            entry.label = std::stoi(label_str);
            data.push_back(entry);
        }
    }
    
    return data;
}

// Function to calculate metrics
void calculate_metrics(const std::vector<int>& true_labels, 
                      const std::vector<int>& predicted_labels,
                      double& accuracy, double& precision, 
                      double& recall, double& f1_score) {
    int tp = 0, fp = 0, fn = 0, tn = 0;
    
    for (size_t i = 0; i < true_labels.size(); i++) {
        if (true_labels[i] == 1 && predicted_labels[i] == 1) tp++;
        else if (true_labels[i] == 0 && predicted_labels[i] == 1) fp++;
        else if (true_labels[i] == 1 && predicted_labels[i] == 0) fn++;
        else if (true_labels[i] == 0 && predicted_labels[i] == 0) tn++;
    }
    
    accuracy = static_cast<double>(tp + tn) / true_labels.size();
    precision = tp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    recall = tp > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    f1_score = (precision + recall) > 0 ? 
        2 * (precision * recall) / (precision + recall) : 0.0;
}

int main() {
    const std::string test_file = "test_data.csv";
    const std::string model_file = "spam_classifier.tflite";
    const std::string vocab_file = "vocab.txt";
    const int max_length = 50;
    
    // Print file information
    std::cout << "Checking files:\n";
    std::cout << "Test file path: " << std::filesystem::absolute(test_file).string() << std::endl;
    std::cout << "Test file exists: " << std::filesystem::exists(test_file) << std::endl;
    
    // Load test data
    auto test_data = read_test_data(test_file);
    std::cout << "Successfully loaded " << test_data.size() 
              << " records from " << std::filesystem::absolute(test_file).string() << std::endl << std::endl;
    
    // Load TFLite model
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        std::cerr << "Failed to create interpreter\n";
        return 1;
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return 1;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Testing time: " << duration.count() / 1000000.0 << " seconds\n";
    
    // Get model file size
    std::ifstream model_ifstream(model_file, std::ios::binary | std::ios::ate);
    double model_size = model_ifstream.tellg() / 1024.0;
    std::cout << "Model Size: " << std::fixed << std::setprecision(2) << model_size << " KB\n\n";
    
    // Process test data and collect metrics
    std::vector<int> true_labels;
    std::vector<int> predicted_labels;
    
    for (const auto& test_item : test_data) {
        auto tokens = tokenize_text(test_item.message, vocab_file, max_length);
        
        // Copy input data
        float* input = interpreter->typed_input_tensor<float>(0);
        for (size_t i = 0; i < tokens.size(); i++) {
            input[i] = tokens[i];
        }
        
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke interpreter\n";
            continue;
        }
        
        // Get output
        float* output = interpreter->typed_output_tensor<float>(0);
        int prediction = output[0] > 0.5 ? 1 : 0;
        
        true_labels.push_back(test_item.label);
        predicted_labels.push_back(prediction);
    }
    
    // Calculate and display metrics
    double accuracy, precision, recall, f1_score;
    calculate_metrics(true_labels, predicted_labels, accuracy, precision, recall, f1_score);
    
    std::cout << "Classification Metrics:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy:  " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall:    " << recall << "\n";
    std::cout << "F1 Score:  " << f1_score << "\n\n";
    
    // Test a single example
    std::string test_message = "Click now to get free money";
    std::cout << "Test Message: \"" << test_message << "\"\n";
    
    auto start_pred_time = std::chrono::high_resolution_clock::now();
    
    auto tokens = tokenize_text(test_message, vocab_file, max_length);
    float* input = interpreter->typed_input_tensor<float>(0);
    for (size_t i = 0; i < tokens.size(); i++) {
        input[i] = tokens[i];
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter\n";
        return 1;
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    bool is_spam = output[0] > 0.5;
    
    auto end_pred_time = std::chrono::high_resolution_clock::now();
    auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_pred_time - start_pred_time);
    
    std::cout << "Prediction: " << (is_spam ? "SPAM" : "NOT SPAM") << "\n";
    std::cout << "Prediction time: " << std::fixed << std::setprecision(2) 
              << pred_duration.count() / 1000000.0 << " seconds\n";
    
    return 0;
}
