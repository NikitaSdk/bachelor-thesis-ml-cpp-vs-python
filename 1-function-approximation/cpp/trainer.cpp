#include "trainer.h"
#include <fann.h>
#include <fann_cpp.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <chrono>   // Додано для вимірювання часу
#include <algorithm>  // Додано для std::numeric_limits
#include <limits>
#include <random>  // Додано для std::mt19937

using namespace std;  // Додаємо цей рядок на початку файлу
namespace fs = std::filesystem;

std::function<float(float)> get_function(FunctionType type) {
    switch (type) {
        case FunctionType::SQUARE: return [](float x) { return x * x; };
        case FunctionType::CUBIC_PLUS_LINEAR: return [](float x) { return x * x * x + 2 * x; };
        case FunctionType::SINUSOID: return [](float x) { return std::sin(x); };
        default: return [](float) { return 0.0f; };
    }
}

std::string get_function_name(FunctionType type) {
    switch (type) {
        case FunctionType::SQUARE: return "square";
        case FunctionType::CUBIC_PLUS_LINEAR: return "cubic";
        case FunctionType::SINUSOID: return "sin";
        default: return "unknown";
    }
}

std::string get_activation_name(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "sigmoid";
        case ActivationType::SIGMOID_SYMMETRIC: return "symmetric_sigmoid";
        case ActivationType::SIN_SYMMETRIC: return "sin_symmetric";
        default: return "unknown";
    }
}

void train_and_test(const std::string& train_file,
                    const std::string& model_file,
                    std::function<float(float)> target_function,
                    ActivationType activation) {
    // Початок заміру часу тренування
    auto train_start = chrono::high_resolution_clock::now();

    // Set random seed for reproducible results
    srand(42);

    FANN::neural_net ann;
    // Network architecture configuration
    int hidden_neurons = 20;  // Default number of neurons
    float learning_rate = 0.1f;
    
    // Get the function type from the training file name
    std::string func_name = get_function_name_from_file(train_file);
    
    // Configure network architecture based on function type
    if (func_name == "square") {
        hidden_neurons = 30;  // 30 neurons for square function
        learning_rate = 0.1f;
        activation = ActivationType::SIGMOID;
    } else if (func_name == "cubic_plus_linear") {
        hidden_neurons = 35;  // 35 neurons for cubic function
        learning_rate = 0.1f;
        activation = ActivationType::SIGMOID_SYMMETRIC;
    } else if (func_name == "sinusoid") {
        hidden_neurons = 40;  // 40 neurons for sinusoid function
        learning_rate = 0.1f;
        activation = ActivationType::SIN_SYMMETRIC;
    }

    // Create network with different architectures for each function
    if (func_name == "square") {
        // Single hidden layer for square function
        ann.create_standard(3, 1, hidden_neurons, 1);
    } else if (func_name == "cubic") {
        // Two hidden layers for cubic function (more complex)
        ann.create_standard(4, 1, hidden_neurons, hidden_neurons/2, 1);
    } else if (func_name == "sin") {
        // Three hidden layers for sinusoid (most complex)
        ann.create_standard(5, 1, hidden_neurons, hidden_neurons/2, hidden_neurons/4, 1);
    } else {
        // Fallback to single hidden layer
        ann.create_standard(3, 1, hidden_neurons, 1);
    }
    
    // Set learning rate
    ann.set_learning_rate(learning_rate);

    // Set activation functions for all hidden layers
    FANN::activation_function_enum activation_func = FANN::SIGMOID_SYMMETRIC;
    switch (activation) {
        case ActivationType::SIGMOID:
            activation_func = FANN::SIGMOID;
            break;
        case ActivationType::SIGMOID_SYMMETRIC:
            activation_func = FANN::SIGMOID_SYMMETRIC;
            break;
        case ActivationType::SIN_SYMMETRIC:
            activation_func = FANN::SIN_SYMMETRIC;
            break;
    }
    
    // Set activation function for all hidden layers
    for (int i = 0; i < ann.get_num_layers() - 1; i++) {
        ann.set_activation_function_hidden(activation_func);
    }
    
    // Set output layer activation
    ann.set_activation_function_output(FANN::LINEAR);
    ann.set_training_algorithm(FANN::TRAIN_INCREMENTAL);

    FANN::training_data data;
    if (!data.read_train_from_file(train_file)) {
        cerr << "Failed to read training file: " << train_file << endl;
        return;
    }

    // Тренування з відслідковуванням найкращого MSE
    float best_mse = 1.0f;  // Початкове значення
    for (unsigned int i = 1; i <= 3000; ++i) {
        float mse = ann.train_epoch(data);
        best_mse = min(best_mse, mse);
        if (i % 500 == 0 || i == 1 || i == 3000)
            cout << "Epoch " << i << ", MSE = " << mse << endl;
    }

    // Кінець заміру часу тренування
    auto train_end = chrono::high_resolution_clock::now();
    chrono::duration<double> train_time = train_end - train_start;

    // Початок заміру часу тестування
    auto test_start = chrono::high_resolution_clock::now();

    // Тестування та збір метрик MSE
    float test_mse = 0.0f;
    int test_count = 0;

    std::string results_dir = "results";
    fs::create_directory(results_dir);

    std::string csv_filename = results_dir + "/" +
        get_function_name_from_file(train_file) + "_" +
        get_activation_name(activation) + ".csv";

    std::ofstream fout(csv_filename);
    if (!fout.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
        return;
    }

    fout << "x,predicted,expected\n";
    std::cout << "\nPredictions \n";

    std::cout << std::fixed;
    for (int i = 0; i <= 10; ++i)
    {
        float x = static_cast<float>(i) / 5.0f - 1.0f;

        fann_type input[1] = {x};
        float predicted = ann.run(input)[0];
        float expected  = target_function(x);

        float error = predicted - expected;
        test_mse += error * error;
        ++test_count;

        // акуратний вивід: «x = …   predicted = …   expected = …»
        std::cout << "x = "
                  << std::setw(6)  << std::right << std::setprecision(2) << x
                  << "   predicted = "
                  << std::setw(10) << std::setprecision(6) << predicted
                  << "   expected = "
                  << std::setw(10) << expected
                  << '\n';

        fout << x << ',' << predicted << ',' << expected << '\n';
    }


    auto test_end = chrono::high_resolution_clock::now();
    chrono::duration<double> test_time = test_end - test_start;


    size_t num_connections = ann.get_total_connections();
    size_t memory_size = num_connections * sizeof(float);
    memory_size += sizeof(FANN::neural_net);

    fout.close();


    std::cout << "\nMetrics \n";
    std::cout << "Training MSE: " << best_mse << std::endl;
    std::cout << "Testing MSE: " << (test_mse / test_count) << std::endl;
    std::cout << "Training Time: " << train_time.count() << " seconds" << std::endl;
    std::cout << "Testing Time: " << test_time.count() << " seconds" << std::endl;
    std::cout << "Model Memory Size: " << memory_size << " bytes ("
              << (memory_size / 1024.0) << " KB)" << std::endl;
    std::cout << "\n";

    ann.save(model_file);
}

std::string get_function_name_from_file(const std::string& filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) dot_pos = filename.length();
    return filename.substr(slash_pos == std::string::npos ? 0 : slash_pos + 1,
                           dot_pos - (slash_pos == std::string::npos ? 0 : slash_pos + 1));
}
