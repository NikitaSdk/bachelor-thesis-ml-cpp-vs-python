#ifndef TRAINER_H
#define TRAINER_H

#include <string>
#include <functional>
#include <chrono>    // Для вимірювання часу
#include <algorithm> // Для std::min та std::max

enum class FunctionType {
    SQUARE,
    CUBIC_PLUS_LINEAR,
    SINUSOID
};

enum class ActivationType {
    SIGMOID,
    SIGMOID_SYMMETRIC,
    SIN_SYMMETRIC
};

std::function<float(float)> get_function(FunctionType type);
std::string get_function_name(FunctionType type);
std::string get_activation_name(ActivationType type);

void train_and_test(const std::string& train_file,
                   const std::string& model_file,
                   std::function<float(float)> target_function,
                   ActivationType activation);

std::string get_function_name_from_file(const std::string& filename);

#endif
