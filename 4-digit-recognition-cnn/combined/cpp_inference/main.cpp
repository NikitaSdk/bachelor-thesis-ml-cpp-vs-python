#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <chrono>

// Structure to hold a single MNIST example
struct MNISTExample {
    std::vector<float> pixels;
    int label;
};

// Structure to hold test results
struct TestResult {
    int epoch;
    float accuracy;
    float loss;
};

// Function to save results to CSV
void saveResultsToCSV(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file");
    }

    // Write header
    file << "Epoch,Test Accuracy,Test Loss\n";

    // Write data
    for (const auto& result : results) {
        file << result.epoch << ","
             << std::fixed << std::setprecision(4) << result.accuracy << ","
             << result.loss << "\n";
    }
}

// Function to read CSV file
std::vector<MNISTExample> readMNISTData(const std::string& imageFile, const std::string& labelFile) {
    std::vector<MNISTExample> data;
    std::ifstream imageStream(imageFile);
    std::ifstream labelStream(labelFile);
    
    if (!imageStream.is_open() || !labelStream.is_open()) {
        throw std::runtime_error("Failed to open data files");
    }

    std::string imageLine, labelLine;
    // Skip headers if they exist
    if (std::getline(imageStream, imageLine)) {
        std::getline(labelStream, labelLine);
    }

    while (std::getline(imageStream, imageLine) && std::getline(labelStream, labelLine)) {
        MNISTExample example;
        
        // Parse label
        example.label = std::stoi(labelLine);

        // Parse image pixels
        std::stringstream ss(imageLine);
        std::string pixel;
        while (std::getline(ss, pixel, ',')) {
            example.pixels.push_back(std::stof(pixel) / 255.0f); // Normalize to [0,1]
        }

        data.push_back(example);
    }

    return data;
}

// Function to calculate categorical cross-entropy loss
float calculateLoss(const cv::Mat& predictions, int trueLabel) {
    float epsilon = 1e-15f;
    float pred = std::max(epsilon, std::min(predictions.at<float>(trueLabel), 1.0f - epsilon));
    return -std::log(pred);
}

int main() {
    // Path to models directory
    std::string modelsDir = "/home/vboxuser/PycharmProjects/CNN_Models_Training_Python/models/";
    
    // Path to test data
    std::string projectDir = "/home/vboxuser/CLionProjects/CNN_TESTMODELS_C++/";
    std::string testDataPath = projectDir + "mnist_test.csv";
    std::string testLabelsPath = projectDir + "mnist_test_labels.csv";
    std::string resultsPath = projectDir + "test_results.csv";
    
    // Vector to store all results
    std::vector<TestResult> results;
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Load test data
    std::cout << "Loading test data..." << std::endl;
    auto testData = readMNISTData(testDataPath, testLabelsPath);
    std::cout << "Loaded " << testData.size() << " test examples" << std::endl << std::endl;

    // Print header with formatting
    std::cout << std::setw(6) << "Epoch" 
              << std::setw(20) << "Test Accuracy (%)"
              << std::setw(15) << "Test Loss" << std::endl;
    std::cout << std::string(41, '-') << std::endl;

    // Test each model
    for (int epoch = 1; epoch <= 15; ++epoch) {
        std::string modelPath = modelsDir + "lenet_epoch" + std::to_string(epoch) + ".onnx";
        
        try {
            // Load the model
            cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
            
            int correctPredictions = 0;
            float totalLoss = 0.0f;
            
            // Process each test example
            for (const auto& example : testData) {
                // Reshape the input data to 28x28 image
                cv::Mat image(28, 28, CV_32F);
                for (int i = 0; i < 28; ++i) {
                    for (int j = 0; j < 28; ++j) {
                        image.at<float>(i, j) = example.pixels[i * 28 + j];
                    }
                }
                
                // Convert to blob (NCHW format)
                cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(), cv::Scalar(), true, false);
                
                // Set input and forward pass
                net.setInput(blob);
                cv::Mat output = net.forward();
                
                // Get prediction
                cv::Point maxLoc;
                cv::minMaxLoc(output, nullptr, nullptr, nullptr, &maxLoc);
                int prediction = maxLoc.x;
                
                // Update metrics
                if (prediction == example.label) {
                    correctPredictions++;
                }
                totalLoss += calculateLoss(output, example.label);
            }
            
            // Calculate metrics
            float accuracy = static_cast<float>(correctPredictions) / testData.size() * 100.0f;
            float avgLoss = totalLoss / testData.size();
            
            // Store results
            results.push_back({epoch, accuracy, avgLoss});
            
            // Print results with formatting
            std::cout << std::setw(6) << epoch 
                      << std::fixed << std::setprecision(4)
                      << std::setw(20) << accuracy
                      << std::setw(15) << avgLoss << std::endl;
            
        } catch (const cv::Exception& e) {
            std::cerr << "Error processing model for epoch " << epoch << ": " << e.what() << std::endl;
            continue;
        }
    }

    // Calculate and print total time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << std::endl << "Total testing time: " << duration.count() << " seconds" << std::endl;

    // Save results to CSV
    try {
        saveResultsToCSV(results, resultsPath);
        std::cout << "Results saved to: " << resultsPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }

    return 0;
}
