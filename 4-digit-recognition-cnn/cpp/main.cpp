#define _USE_MATH_DEFINES
#include <cmath>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iomanip>
#include <chrono>

using namespace dlib;
using namespace std;
namespace fs = std::filesystem;

// Define our CNN architecture
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <typename SUBNET> using downsampling  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_multiclass_log<
    fc<10,
    relu<fc<84,
    relu<fc<120,
    max_pool<2,2,2,2,relu<con5<50,
    max_pool<2,2,2,2,relu<con5<20,
    input<matrix<unsigned char>>
    >>>>>>>>>>>>;

// Function to load MNIST data
void load_mnist_data(const string& data_path, const string& labels_path,
                    std::vector<matrix<unsigned char>>& images,
                    std::vector<unsigned long>& labels) {
    cout << "\n=== Loading data from files ===" << endl;
    cout << "Data file: " << data_path << endl;
    cout << "Labels file: " << labels_path << endl;
    
    ifstream data_file(data_path);
    ifstream labels_file(labels_path);
    
    if (!data_file.is_open()) {
        throw runtime_error("Could not open data file: " + data_path);
    }
    if (!labels_file.is_open()) {
        throw runtime_error("Could not open labels file: " + labels_path);
    }

    cout << "Files opened successfully" << endl;

    string line;
    string label_line;
    
    // Check if files are not empty
    if (!getline(data_file, line)) {
        throw runtime_error("Data file is empty");
    }
    if (!getline(labels_file, label_line)) {
        throw runtime_error("Labels file is empty");
    }
    
    cout << "Files contain data" << endl;
    
    // Reset file positions
    data_file.clear();
    data_file.seekg(0);
    labels_file.clear();
    labels_file.seekg(0);
    
    // Skip header if exists
    getline(data_file, line);
    getline(labels_file, label_line);

    int line_count = 0;
    while (getline(data_file, line) && getline(labels_file, label_line)) {
        try {
            // Parse label
            unsigned long label = stoul(label_line);
            labels.push_back(label);

            // Parse image data
            stringstream ss(line);
            matrix<unsigned char> img(28, 28);
            string value;
            int pixel_count = 0;
            
            // Read all pixels
            while (getline(ss, value, ',')) {
                if (pixel_count >= 784) { // 28x28 = 784 pixels
                    throw runtime_error("Too many values in line " + to_string(line_count));
                }
                
                int row = pixel_count / 28;
                int col = pixel_count % 28;
                
                try {
                    double pixel_value = stod(value);
                    img(row, col) = static_cast<unsigned char>(pixel_value * 255.0); // Scale to 0-255
                } catch (const exception& e) {
                    throw runtime_error("Invalid pixel value at position " + to_string(pixel_count) + 
                                      " in line " + to_string(line_count) + ": " + e.what());
                }
                
                pixel_count++;
            }
            
            if (pixel_count != 784) {
                throw runtime_error("Not enough values in line " + to_string(line_count) + 
                                  ". Expected 784, got " + to_string(pixel_count));
            }
            
            images.push_back(img);
            line_count++;
            
            if (line_count % 1000 == 0) {
                cout << "Processed " << line_count << " images..." << endl;
            }
        } catch (const exception& e) {
            cout << "Error on line " << line_count << ": " << e.what() << endl;
            cout << "Line content: " << line.substr(0, 100) << "..." << endl;
            cout << "Label content: " << label_line << endl;
            throw;
        }
    }
    
    cout << "Successfully loaded " << images.size() << " images and " << labels.size() << " labels" << endl;
    if (images.empty()) {
        cout << "WARNING: No images were loaded!" << endl;
    }
    
    // Verify data consistency
    if (images.size() != labels.size()) {
        throw runtime_error("Number of images (" + to_string(images.size()) + 
                          ") does not match number of labels (" + to_string(labels.size()) + ")");
    }
}

int main() {
    try {
        // Check if files exist using filesystem
        string train_data_path = "data/mnist_train.csv";
        string train_labels_path = "data/mnist_train_labels.csv";
        string test_data_path = "data/mnist_test.csv";
        string test_labels_path = "data/mnist_test_labels.csv";

        cout << "Checking files..." << endl;
        cout << "Train data exists: " << (fs::exists(train_data_path) ? "Yes" : "No") << endl;
        cout << "Train labels exist: " << (fs::exists(train_labels_path) ? "Yes" : "No") << endl;
        cout << "Test data exists: " << (fs::exists(test_data_path) ? "Yes" : "No") << endl;
        cout << "Test labels exist: " << (fs::exists(test_labels_path) ? "Yes" : "No") << endl;

        // Create a network
        net_type net;

        // Load training data
        std::vector<matrix<unsigned char>> training_images;
        std::vector<unsigned long> training_labels;
        std::vector<matrix<unsigned char>> test_images;
        std::vector<unsigned long> test_labels;

        cout << "Loading training data..." << endl;
        load_mnist_data(train_data_path, train_labels_path, training_images, training_labels);

        cout << "Loading test data..." << endl;
        load_mnist_data(test_data_path, test_labels_path, test_images, test_labels);

        cout << "Training data size: " << training_images.size() << endl;
        cout << "Test data size: " << test_images.size() << endl;

        // Create a trainer
        dnn_trainer<net_type> trainer(net);
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(128);
        trainer.set_max_num_epochs(20);  // Set to 20 epochs
        //trainer.be_verbose();  // Enable verbose output for training progress

        // Train the network
        cout << "\n=== Training Network ===" << endl;
        cout << "Epoch\tTrain Accuracy\tTrain Loss\tTest Accuracy\tTest Loss" << endl;
        cout << "------------------------------------------------------------" << endl;

        // Create and open CSV file for saving metrics
        ofstream metrics_file("training_metrics.csv");
        if (!metrics_file.is_open()) {
            throw runtime_error("Could not open metrics file for writing");
        }
        // Write header
        metrics_file << "epoch,train_accuracy,train_loss,test_accuracy,test_loss\n";

        // Start timing training
        auto training_start = chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < 20; ++epoch) {
            // Train one epoch using mini-batches
            size_t mini_batch_size = 128;
            for (size_t i = 0; i < training_images.size(); i += mini_batch_size) {
                size_t current_batch_size = std::min(mini_batch_size, training_images.size() - i);
                std::vector<matrix<unsigned char>> batch_images(
                    training_images.begin() + i, 
                    training_images.begin() + i + current_batch_size
                );
                std::vector<unsigned long> batch_labels(
                    training_labels.begin() + i, 
                    training_labels.begin() + i + current_batch_size
                );

                trainer.train_one_step(batch_images, batch_labels);
            }

            // Get the current network state for evaluation
            auto net_snapshot = trainer.get_net();

            // Calculate training metrics
            double train_accuracy = 0;
            double train_loss = 0;
            for (size_t i = 0; i < training_images.size(); ++i) {
                auto predicted_label = net_snapshot(training_images[i]);
                if (predicted_label == training_labels[i])
                    train_accuracy += 1.0;
            }
            train_accuracy /= training_images.size();
            train_loss = trainer.get_average_loss();

            // Calculate test metrics
            double test_accuracy = 0;
            double test_loss = 0;
            for (size_t i = 0; i < test_images.size(); ++i) {
                auto predicted_label = net_snapshot(test_images[i]);
                if (predicted_label == test_labels[i])
                    test_accuracy += 1.0;
            }
            test_accuracy /= test_images.size();
            test_loss = -log(test_accuracy); // Calculate test loss based on accuracy

            // Print metrics to console
            cout << epoch + 1 << "\t"
                 << fixed << setprecision(4) << train_accuracy << "\t\t"
                 << fixed << setprecision(4) << train_loss << "\t\t"
                 << fixed << setprecision(4) << test_accuracy << "\t\t"
                 << fixed << setprecision(4) << test_loss << endl;

            // Save metrics to CSV file with high precision
            metrics_file << epoch + 1 << ","
                        << fixed << setprecision(6) << train_accuracy << ","
                        << fixed << setprecision(6) << train_loss << ","
                        << fixed << setprecision(6) << test_accuracy << ","
                        << fixed << setprecision(6) << test_loss << "\n";
        }

        // End timing training
        auto training_end = chrono::high_resolution_clock::now();
        auto training_duration = chrono::duration_cast<chrono::seconds>(training_end - training_start);

        // Close the metrics file
        metrics_file.close();

        // Save the network
        cout << "\nSaving network..." << endl;
        serialize("cnn_network.dat") << net;

        // Calculate model size
        size_t model_size = fs::file_size("cnn_network.dat");
        double model_size_kb = static_cast<double>(model_size) / 1024.0;

        // Test the network
        cout << "\n=== Testing Network ===" << endl;
        cout << "Running final test on test dataset..." << endl;
        
        // Start timing testing
        auto testing_start = chrono::high_resolution_clock::now();
        
        int correct = 0;
        int total = test_images.size();
        
        for (size_t i = 0; i < test_images.size(); ++i) {
            auto predicted_label = net(test_images[i]);
            if (predicted_label == test_labels[i])
                ++correct;
        }

        // End timing testing
        auto testing_end = chrono::high_resolution_clock::now();
        auto testing_duration = chrono::duration_cast<chrono::milliseconds>(testing_end - testing_start);

        double final_accuracy = (double)correct/total;
        cout << "\nTest Results:" << endl;
        cout << "  - Total test samples: " << total << endl;
        cout << "  - Correct predictions: " << correct << endl;
        cout << "  - Final accuracy: " << fixed << setprecision(4) << final_accuracy 
             << " (" << fixed << setprecision(2) << (final_accuracy * 100) << "%)" << endl;
        
        // Print timing and model size information
        cout << "\nPerformance Metrics:" << endl;
        cout << "  - Training time: " << training_duration.count() << " seconds" << endl;
        cout << "  - Testing time: " << testing_duration.count() << " milliseconds" << endl;
        cout << "  - Model size: " << fixed << setprecision(2) << model_size_kb << " KB" << endl;

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
