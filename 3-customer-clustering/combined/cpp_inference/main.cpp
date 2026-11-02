#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/common.h>
#include <cmath>
#include <chrono>
#include <iomanip>

// Structure to hold input data
struct InputData {
    float operator_id;
    float call_minutes;
    float sms_count;
    float internet_mb;
    float last_activity_days;
    float customer_age;
    float prepaid;
    
    // Original values for display
    float original_operator_id;
    float original_call_minutes;
    float original_sms_count;
    float original_internet_mb;
    float original_last_activity_days;
    float original_customer_age;
    float original_prepaid;
};

// Function to calculate L2 norm for a vector of values
float calculateL2Norm(const std::vector<float>& values) {
    float sum_squares = 0.0f;
    for (float value : values) {
        sum_squares += value * value;
    }
    return std::sqrt(sum_squares);
}

// Function to normalize data using L2 normalization
void normalizeData(std::vector<InputData>& data) {
    // Calculate L2 norms for each feature
    std::vector<float> operator_norms, call_norms, sms_norms, internet_norms, 
                      activity_norms, age_norms, prepaid_norms;
    
    for (const auto& row : data) {
        operator_norms.push_back(row.operator_id);
        call_norms.push_back(row.call_minutes);
        sms_norms.push_back(row.sms_count);
        internet_norms.push_back(row.internet_mb);
        activity_norms.push_back(row.last_activity_days);
        age_norms.push_back(row.customer_age);
        prepaid_norms.push_back(row.prepaid);
    }
    
    float operator_norm = calculateL2Norm(operator_norms);
    float call_norm = calculateL2Norm(call_norms);
    float sms_norm = calculateL2Norm(sms_norms);
    float internet_norm = calculateL2Norm(internet_norms);
    float activity_norm = calculateL2Norm(activity_norms);
    float age_norm = calculateL2Norm(age_norms);
    float prepaid_norm = calculateL2Norm(prepaid_norms);
    
    // Handle zero norms
    if (operator_norm == 0.0f) operator_norm = 1.0f;
    if (call_norm == 0.0f) call_norm = 1.0f;
    if (sms_norm == 0.0f) sms_norm = 1.0f;
    if (internet_norm == 0.0f) internet_norm = 1.0f;
    if (activity_norm == 0.0f) activity_norm = 1.0f;
    if (age_norm == 0.0f) age_norm = 1.0f;
    if (prepaid_norm == 0.0f) prepaid_norm = 1.0f;
    
    // Store original values and normalize
    for (auto& row : data) {
        // Store original values
        row.original_operator_id = row.operator_id;
        row.original_call_minutes = row.call_minutes;
        row.original_sms_count = row.sms_count;
        row.original_internet_mb = row.internet_mb;
        row.original_last_activity_days = row.last_activity_days;
        row.original_customer_age = row.customer_age;
        row.original_prepaid = row.prepaid;
        
        // Normalize
        row.operator_id /= operator_norm;
        row.call_minutes /= call_norm;
        row.sms_count /= sms_norm;
        row.internet_mb /= internet_norm;
        row.last_activity_days /= activity_norm;
        row.customer_age /= age_norm;
        row.prepaid /= prepaid_norm;
    }
}

// Function to read CSV file
std::vector<InputData> readCSV(const std::string& filename) {
    std::vector<InputData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    
    // Skip header
    std::getline(file, line);
    std::cout << "Header from file: " << line << std::endl;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        InputData row;
        
        std::getline(ss, value, ',');
        row.operator_id = std::stof(value);
        
        std::getline(ss, value, ',');
        row.call_minutes = std::stof(value);
        
        std::getline(ss, value, ',');
        row.sms_count = std::stof(value);
        
        std::getline(ss, value, ',');
        row.internet_mb = std::stof(value);
        
        std::getline(ss, value, ',');
        row.last_activity_days = std::stof(value);
        
        std::getline(ss, value, ',');
        row.customer_age = std::stof(value);
        
        std::getline(ss, value, ',');
        row.prepaid = std::stof(value);
        
        data.push_back(row);
    }
    
    // Normalize data using L2 normalization
    normalizeData(data);
    
    std::cout << "Read " << data.size() << " rows from file" << std::endl;
    return data;
}

// Function to calculate silhouette score
float calculateSilhouetteScore(const std::vector<InputData>& data, const std::vector<int>& clusters) {
    float total_score = 0.0f;
    int num_clusters = 3;
    int valid_points = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        float a = 0.0f;
        int same_cluster_count = 0;
        
        // Calculate a (average distance to points in same cluster)
        for (size_t j = 0; j < data.size(); ++j) {
            if (clusters[i] == clusters[j] && i != j) {
                float dist = std::sqrt(
                    std::pow(data[i].operator_id - data[j].operator_id, 2) +
                    std::pow(data[i].call_minutes - data[j].call_minutes, 2) +
                    std::pow(data[i].sms_count - data[j].sms_count, 2) +
                    std::pow(data[i].internet_mb - data[j].internet_mb, 2) +
                    std::pow(data[i].last_activity_days - data[j].last_activity_days, 2) +
                    std::pow(data[i].customer_age - data[j].customer_age, 2) +
                    std::pow(data[i].prepaid - data[j].prepaid, 2)
                );
                a += dist;
                same_cluster_count++;
            }
        }
        if (same_cluster_count > 0) {
            a /= same_cluster_count;
        }
        
        float b = std::numeric_limits<float>::max();
        for (int c = 0; c < num_clusters; ++c) {
            if (c != clusters[i]) {
                float avg_dist = 0.0f;
                int count = 0;
                for (size_t j = 0; j < data.size(); ++j) {
                    if (clusters[j] == c) {
                        float dist = std::sqrt(
                            std::pow(data[i].operator_id - data[j].operator_id, 2) +
                            std::pow(data[i].call_minutes - data[j].call_minutes, 2) +
                            std::pow(data[i].sms_count - data[j].sms_count, 2) +
                            std::pow(data[i].internet_mb - data[j].internet_mb, 2) +
                            std::pow(data[i].last_activity_days - data[j].last_activity_days, 2) +
                            std::pow(data[i].customer_age - data[j].customer_age, 2) +
                            std::pow(data[i].prepaid - data[j].prepaid, 2)
                        );
                        avg_dist += dist;
                        count++;
                    }
                }
                if (count > 0) {
                    avg_dist /= count;
                    b = std::min(b, avg_dist);
                }
            }
        }
        
        if (b > a) {
            float s = (b - a) / std::max(a, b);
            total_score += s;
            valid_points++;
        }
    }
    
    return valid_points > 0 ? total_score / valid_points : 0.0f;
}

// Function to calculate intra-cluster distance
float calculateIntraClusterDistance(const std::vector<InputData>& data, const std::vector<int>& clusters) {
    float total_distance = 0.0f;
    int total_points = 0;
    
    for (int cluster = 0; cluster < 3; ++cluster) {
        std::vector<InputData> cluster_points;
        for (size_t i = 0; i < data.size(); ++i) {
            if (clusters[i] == cluster) {
                cluster_points.push_back(data[i]);
            }
        }
        
        // Calculate pairwise distances within cluster
        for (size_t i = 0; i < cluster_points.size(); ++i) {
            for (size_t j = i + 1; j < cluster_points.size(); ++j) {
                float dist = std::sqrt(
                    std::pow(cluster_points[i].operator_id - cluster_points[j].operator_id, 2) +
                    std::pow(cluster_points[i].call_minutes - cluster_points[j].call_minutes, 2) +
                    std::pow(cluster_points[i].sms_count - cluster_points[j].sms_count, 2) +
                    std::pow(cluster_points[i].internet_mb - cluster_points[j].internet_mb, 2) +
                    std::pow(cluster_points[i].last_activity_days - cluster_points[j].last_activity_days, 2) +
                    std::pow(cluster_points[i].customer_age - cluster_points[j].customer_age, 2) +
                    std::pow(cluster_points[i].prepaid - cluster_points[j].prepaid, 2)
                );
                total_distance += dist;
                total_points++;
            }
        }
    }
    
    return total_points > 0 ? total_distance / total_points : 0.0f;
}

// Function to analyze cluster characteristics
void analyzeCluster(const std::vector<InputData>& data, const std::vector<int>& clusters, int cluster_id) {
    std::vector<InputData> cluster_data;
    for (size_t i = 0; i < data.size(); ++i) {
        if (clusters[i] == cluster_id) {
            cluster_data.push_back(data[i]);
        }
    }
    
    if (cluster_data.empty()) return;
    
    // Calculate averages using original values
    float avg_calls = 0.0f;
    float avg_sms = 0.0f;
    float avg_internet = 0.0f;
    float avg_age = 0.0f;
    
    for (const auto& point : cluster_data) {
        avg_calls += point.original_call_minutes;
        avg_sms += point.original_sms_count;
        avg_internet += point.original_internet_mb;
        avg_age += point.original_customer_age;
    }
    
    avg_calls /= cluster_data.size();
    avg_sms /= cluster_data.size();
    avg_internet /= cluster_data.size();
    avg_age /= cluster_data.size();
    
    // Determine cluster profile using ratios like in pure C++ version
    std::string profile;
    double internet_ratio = avg_internet / 5000.0;
    double calls_ratio = avg_calls / 250.0;
    double sms_ratio = avg_sms / 50.0;
    
    if (internet_ratio > calls_ratio && internet_ratio > sms_ratio && internet_ratio > 1.0) {
        profile = "Heavy Internet Users";
    } else if (calls_ratio > internet_ratio && calls_ratio > sms_ratio && calls_ratio > 1.0) {
        profile = "Heavy Call Users";
    } else if (sms_ratio > internet_ratio && sms_ratio > calls_ratio && sms_ratio > 1.0) {
        profile = "Heavy SMS Users";
    } else {
        profile = "Moderate Users";
    }
    
    // Print cluster analysis
    std::cout << "\nCluster " << cluster_id << " Summary:\n";
    std::cout << "Size: " << cluster_data.size() << " customers (" 
              << std::fixed << std::setprecision(1)
              << (cluster_data.size() * 100.0f / data.size()) << "%)\n\n";
    
    std::cout << "Average Usage Patterns:\n";
    std::cout << "- Calls: " << std::fixed << std::setprecision(1) << avg_calls << " minutes/month\n";
    std::cout << "- SMS: " << std::fixed << std::setprecision(1) << avg_sms << " messages/month\n";
    std::cout << "- Internet: " << std::fixed << std::setprecision(1) << avg_internet << " MB/month\n";
    std::cout << "- Average Age: " << std::fixed << std::setprecision(1) << avg_age << " years\n\n";
    
    std::cout << "Cluster Profile: " << profile << "\n\n";
}

// Function to calculate distance to centroids
int findNearestCentroid(const InputData& point, const std::vector<std::vector<float>>& centroids) {
    float min_dist = std::numeric_limits<float>::max();
    int nearest_centroid = -1;
    
    for (size_t i = 0; i < centroids.size(); ++i) {
        float dist = 0.0f;
        dist += std::pow(point.operator_id - centroids[i][0], 2);
        dist += std::pow(point.call_minutes - centroids[i][1], 2);
        dist += std::pow(point.sms_count - centroids[i][2], 2);
        dist += std::pow(point.internet_mb - centroids[i][3], 2);
        dist += std::pow(point.last_activity_days - centroids[i][4], 2);
        dist += std::pow(point.customer_age - centroids[i][5], 2);
        dist += std::pow(point.prepaid - centroids[i][6], 2);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_centroid = static_cast<int>(i);
        }
    }
    
    return nearest_centroid;
}

// Function to save results to CSV
void saveResultsToCSV(const std::vector<InputData>& data, 
                     const std::vector<int>& clusters,
                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }
    
    // Write header
    file << "Operator,CallMinutes,SMSCount,InternetMB,LastActivityDays,CustomerAge,Prepaid,Cluster\n";
    
    // Write data
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i].original_operator_id << ","
             << data[i].original_call_minutes << ","
             << data[i].original_sms_count << ","
             << data[i].original_internet_mb << ","
             << data[i].original_last_activity_days << ","
             << data[i].original_customer_age << ","
             << data[i].original_prepaid << ","
             << clusters[i] << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << "\n";
}

int main() {
    const std::string PROJECT_DIR = "/home/vboxuser/CLionProjects/TFLITE_C++_TEST_CLUSTERING_TRY2";
    
    std::cout << "Phase 2: Testing\n";
    
    // Load TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile("/home/vboxuser/PycharmProjects/TFLITE_CLUSTERING_PYTHON_TRAINING_MODEL/kmeans_centroid_model.tflite");
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    
    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (tflite::InterpreterBuilder(*model, resolver)(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter\n";
        return 1;
    }
    
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter\n";
        return 1;
    }
    
    // Configure interpreter
    interpreter->SetNumThreads(1);
    
    // Get tensor indices
    if (interpreter->inputs().size() != 1 || interpreter->outputs().size() != 1) {
        std::cerr << "Model should have exactly 1 input and 1 output\n";
        return 1;
    }
    
    int input_tensor_idx = interpreter->inputs()[0];
    int output_tensor_idx = interpreter->outputs()[0];
    
    // Resize input tensor to shape (7) - no batch dimension
    std::vector<int> input_shape = {7};
    if (interpreter->ResizeInputTensor(input_tensor_idx, input_shape) != kTfLiteOk) {
        std::cerr << "Failed to resize input tensor\n";
        return 1;
    }
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return 1;
    }
    
    // Read input data
    std::string input_file = PROJECT_DIR + "/clients_mobile_test_data.csv";
    std::vector<InputData> input_data = readCSV(input_file);
    
    if (input_data.empty()) {
        std::cerr << "No data was read from the input file\n";
        return 1;
    }
    
    std::cout << "Read " << input_data.size() << " customers from test file.\n\n";
    
    // Process each row and collect cluster assignments
    float* input = interpreter->typed_input_tensor<float>(0);
    if (!input) {
        std::cerr << "Failed to get input tensor data\n";
        return 1;
    }
    
    std::vector<int> cluster_assignments;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& row : input_data) {
        // Fill input tensor with normalized data
        input[0] = row.operator_id;
        input[1] = row.call_minutes;
        input[2] = row.sms_count;
        input[3] = row.internet_mb;
        input[4] = row.last_activity_days;
        input[5] = row.customer_age;
        input[6] = row.prepaid;
        
        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke interpreter\n";
            continue;
        }
        
        // Get output based on tensor type
        int cluster_id = -1;
        TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_idx);
        
        switch (output_tensor->type) {
            case kTfLiteFloat32: {
                float* output = interpreter->typed_output_tensor<float>(0);
                if (output) {
                    cluster_id = static_cast<int>(std::round(output[0]));
                }
                break;
            }
            case kTfLiteInt32: {
                int32_t* output = interpreter->typed_output_tensor<int32_t>(0);
                if (output) {
                    cluster_id = output[0];
                }
                break;
            }
            case kTfLiteInt64: {
                int64_t* output = interpreter->typed_output_tensor<int64_t>(0);
                if (output) {
                    cluster_id = static_cast<int>(output[0]);
                }
                break;
            }
        }
        
        if (cluster_id >= 0) {
            cluster_assignments.push_back(cluster_id);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate metrics
    float silhouette_score = calculateSilhouetteScore(input_data, cluster_assignments);
    float intra_cluster_distance = calculateIntraClusterDistance(input_data, cluster_assignments);
    
    // Print test metrics
    std::cout << "Test Metrics:\n";
    std::cout << "Assignment time: " << duration.count() / 1000000.0 << " seconds\n";
    std::cout << "Silhouette Score: " << silhouette_score << "\n";
    std::cout << "Intra-cluster Distance: " << intra_cluster_distance << "\n";
    std::cout << "Model size: " << model->allocation()->bytes() / 1024.0 << " KB\n\n";
    
    // Print cluster analysis
    std::cout << "Test Data Cluster Analysis:\n";
    for (int i = 0; i < 3; ++i) {
        analyzeCluster(input_data, cluster_assignments, i);
    }
    
    // Save results to CSV
    std::string output_file = PROJECT_DIR + "/result.csv";
    saveResultsToCSV(input_data, cluster_assignments, output_file);
    
    return 0;
}
